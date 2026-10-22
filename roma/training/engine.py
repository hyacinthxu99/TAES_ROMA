"""Pretraining, adaptation and reconstruction-score evaluation."""
import hashlib
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from roma import ROMAModel, build_backbone, correlation_graph
from roma.data.telemetry import adaptation_data, pretraining_data, select_features, read_matrix
from roma.evaluation.metrics import get_metrics
from roma.evaluation.point_scores import PointScores
from .baselines import DirectPretrainAligner, TrainablePretrainAligner

METHODS = ('lora', 'full', 'pretrained_full_finetune', 'direct_pretrain_align')


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def task_seed(seed, dataset, model, entity, method):
    text = f'{seed}|{dataset}|{model}|{entity}|{method}|full'
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def loader(dataset, training):
    return DataLoader(dataset, batch_size=training['batch_size'], shuffle=False, num_workers=0, drop_last=False)


def forward(model, x):
    return model(x) if isinstance(model, ROMAModel) else model(x, None, None, None)


def fit(model, dataset, training, device, specific_loss=False):
    """Match original Adam, type1 LR schedule, and minimum mean minibatch train loss."""
    model.to(device)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=training['learning_rate'])
    best, best_epoch, best_loss = None, None, float('inf')
    losses, durations = [], []
    train_loader = loader(dataset, training)
    for epoch in range(training['epochs']):
        start = time.perf_counter()
        model.train()
        batch_losses = []
        for batch, _ in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            if specific_loss and hasattr(model, 'compute_loss'):
                loss = model.compute_loss(batch, None, None, None)
            else:
                loss = nn.functional.mse_loss(forward(model, batch), batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite training loss at epoch {epoch+1}')
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        mean = float(np.mean(batch_losses))
        losses.append(mean)
        durations.append(time.perf_counter()-start)
        print(f'epoch={epoch+1}/{training["epochs"]} train_loss={mean:.8g}', flush=True)
        if mean < best_loss:
            best_loss, best_epoch = mean, epoch+1
            best = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        for group in optimizer.param_groups:
            group['lr'] = training['learning_rate']*.5**epoch
    model.load_state_dict(best)
    return dict(train_loss_by_epoch=losses, best_epoch=best_epoch, epoch_times_s=durations,
                training_time_total_s=sum(durations), trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
                total_params=sum(p.numel() for p in model.parameters()))


def evaluate(model, windows, labels, mask, training, device, esa=False):
    model.to(device).eval()
    point = PointScores(len(labels), 1) if esa else None
    scores, repeated_labels = [], []
    start = time.perf_counter()
    with torch.no_grad():
        for x, y in loader(windows, training):
            x = x.to(device)
            score = (forward(model, x)-x).square().mean(dim=-1).cpu().numpy()
            if esa:
                point.add(score)
            else:
                scores.append(score.reshape(-1))
                repeated_labels.append(y.numpy().reshape(-1))
    if esa:
        energy, truth, _ = point.finish(labels, mask)
    else:
        energy, truth = np.concatenate(scores), np.concatenate(repeated_labels)
    if not np.isfinite(energy).all() or len(np.unique(truth)) != 2:
        raise ValueError('Evaluation requires finite scores and both binary classes')
    metrics = get_metrics(energy, truth, version='opt_mem' if esa else 'opt')
    metrics = {key: float(value) for key, value in metrics.items()}
    metrics.update(evaluation_points=len(energy), inference_and_metrics_time_s=time.perf_counter()-start,
                   f1_threshold_policy='legacy_oracle', score_aggregation='mean_overlapping_windows' if esa else 'flattened_overlapping_windows',
                   evaluation_protocol='custom_esa_temporal_not_official_esa_adb' if esa else 'legacy_reproduction')
    return metrics


def run_group(config, config_dir, output, tag, dataset, model_name, seed, device):
    """One independent seed/model/dataset group, resumable at entity/method granularity."""
    config_dir, output = Path(config_dir), Path(output)
    spec, training = config['datasets'][dataset], config['training']
    if not spec['history'] or not spec['new'] or set(spec['history']) & set(spec['new']):
        raise ValueError('history and new must be nonempty, disjoint entity groups')
    root = Path(config['data_root'])/dataset
    seq_len = training['seq_len']
    base_indices = select_features(spec, training['pretrain_ratio'], training['dimension_seed'], config_dir)
    base_width = len(base_indices)
    def build(width):
        return build_backbone(model_name, enc_in=width, seq_len=seq_len, **config.get('backbone', {}))

    checkpoint = output/'checkpoints'/f'seed{seed}'/model_name/f'{dataset}.pt'
    seed_everything(seed)
    if checkpoint.exists():
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
        base_state = payload['state_dict']
    else:
        base = build(base_width)
        pretrain = pretraining_data(root, spec, base_indices, seq_len)
        details = fit(base, pretrain, training, device, specific_loss=True)
        base_state = {key: value.detach().cpu().clone() for key, value in base.state_dict().items()}
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        temp = checkpoint.with_suffix('.tmp')
        torch.save(dict(state_dict=base_state, seed=seed, selected_indices=base_indices, training=details), temp)
        temp.replace(checkpoint)
        del base, pretrain
    def pretrained():
        backbone = build(base_width)
        backbone.load_state_dict(base_state)
        return backbone
    methods = METHODS + (('patchtsad_channel_independent',) if model_name == 'PatchTST' else ())
    for entity in spec['new']:
        indices = select_features(spec, spec['ratios'][entity], training['dimension_seed'], config_dir)
        train, test, labels, mask = adaptation_data(root, spec, entity, indices, seq_len)
        for method in methods:
            path = output/'results'/f'seed{seed}'/f'anomaly_detection_{dataset}_{model_name}_match'/f'{Path(entity).stem}__{method}.json'
            if path.exists():
                row = json.loads(path.read_text())
                if all(isinstance(row.get(key), (int, float)) and np.isfinite(row[key]) for key in ['AUC-ROC','VUS-ROC']):
                    print(f'[Resume] {entity}/{method}', flush=True)
                    continue
            seed_everything(task_seed(seed, dataset, model_name, entity, method))
            print(f'[Task] seed={seed} {dataset}/{model_name}/{entity}/{method} dims={base_width}->{len(indices)}', flush=True)
            if method == 'full':
                model = build(len(indices))
            elif method == 'pretrained_full_finetune':
                model = TrainablePretrainAligner(pretrained(), base_width, len(indices))
            elif method == 'direct_pretrain_align':
                model = DirectPretrainAligner(pretrained(), base_width, len(indices))
            elif method == 'patchtsad_channel_independent':
                model = pretrained()
            else:
                raw = read_matrix(root/'train'/entity, spec['csv_header'])[:, indices]
                length = training['graph_sample_points']
                if length > 0 and len(raw) > length:
                    offset = (len(raw)-length)//2
                    raw = raw[offset:offset+length]
                graph = correlation_graph(raw, training['graph_threshold'])
                model = ROMAModel(pretrained(), base_width, len(indices), seq_len, graph,
                                  rank=training['rank'], alpha=training['alpha'], device=device)
            detail = {}
            if method != 'direct_pretrain_align':
                detail = fit(model, train, training, device, specific_loss=method == 'full')
            metrics = evaluate(model, test, labels, mask, training, device, esa=bool(spec.get('manifest')))
            atomic_json(path, dict(data=dataset, model=model_name, file_name=entity, method=method,
                entity_role='simulation_full_retraining' if method=='full' else method, adapter_variant='full',
                experiment_tag=f'{tag}_s{seed}', experiment_stage='stage01_main', training_seed=seed,
                base_dims=base_width, new_dims=len(indices), base_selected_indices=base_indices,
                selected_indices=indices, pretraining_entities=spec['history'], **detail, **metrics))
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
