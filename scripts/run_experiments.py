#!/usr/bin/env python3
"""Run the configured experiments with resumable parallel workers."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import signal
import sys
import traceback

from experiment_config import resolve_datasets

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            result.update(chunk)
    return result.hexdigest()


def parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT/'configs/main.json')
    p.add_argument('--data-root', type=Path)
    p.add_argument('--output', type=Path, default=ROOT/'results/main')
    p.add_argument('--tag', default='roma_main')
    p.add_argument('--seeds', help='Comma-separated; default 2021,...,2030')
    p.add_argument('--models', help='Comma-separated backbone names')
    p.add_argument('--datasets', help='Comma-separated dataset names')
    p.add_argument('--pretrain-ratio', type=float, help='History variable fraction in (0, 1]; default from config (0.8)')
    p.add_argument('--target-ratio-min', type=float, help='Minimum deployment variable fraction; default from config (0.6)')
    p.add_argument('--target-ratio-max', type=float, help='Maximum deployment variable fraction; default from config (1.0)')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--jobs', type=int, default=1)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--max-retries', type=int, default=3)
    p.add_argument('--retry-delay', type=float, default=30)
    p.add_argument('--worker-timeout', type=float, default=7200)
    p.add_argument('--dry-run', action='store_true', help='Validate required data and show plan, without training')
    p.add_argument('--plan-only', action='store_true', help='Show counts without requiring data or scientific dependencies')
    p.add_argument('--report-only', action='store_true')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--seed', type=int, help=argparse.SUPPRESS)
    p.add_argument('--dataset', help=argparse.SUPPRESS)
    p.add_argument('--model', help=argparse.SUPPRESS)
    return p.parse_args()


def main():
    args = parse()
    output = args.output.resolve()
    if args.report_only:
        from roma.evaluation.statistics import report
        plan = json.loads((output/'plan.json').read_text())
        if len(plan['seeds']) < 2:
            raise ValueError('At least two seeds are required for sample standard deviation')
        report(plan, output)
        return
    path = args.config.resolve()
    config = json.loads(path.read_text())
    config['data_root'] = str((args.data_root or path.parent/config['data_root']).resolve())
    if args.worker:
        config = json.loads((output/'plan.json').read_text())['config']
        import torch
        from roma.training.engine import run_group
        torch.set_num_threads(args.threads)
        try:
            run_group(config, path.parent, output, args.tag, args.dataset, args.model, args.seed, args.device)
        except Exception as error:
            traceback.print_exc()
            from roma.training.scheduler import is_memory_error
            sys.exit(86 if is_memory_error(str(error)) else 87)
        return
    training = config['training']
    if args.pretrain_ratio is not None:
        training['pretrain_ratio'] = args.pretrain_ratio
    lower, upper = training['target_ratio']
    training['target_ratio'] = [
        lower if args.target_ratio_min is None else args.target_ratio_min,
        upper if args.target_ratio_max is None else args.target_ratio_max,
    ]
    seeds = [int(x) for x in args.seeds.split(',')] if args.seeds else config['seeds']
    models = args.models.split(',') if args.models else config['models']
    datasets = args.datasets.split(',') if args.datasets else list(config['datasets'])
    if not seeds or min(seeds)<0 or len(seeds)!=len(set(seeds)):
        raise ValueError('Seeds must be unique nonnegative integers')
    if not models or len(models)!=len(set(models)) or any(x not in config['models'] for x in models):
        raise ValueError('Models must be unique configured backbones')
    if not datasets or len(datasets)!=len(set(datasets)) or any(x not in config['datasets'] for x in datasets):
        raise ValueError('Datasets must be unique configured datasets')
    resolve_datasets(config, datasets, path.parent)
    entities = {key: config['datasets'][key]['new'] for key in datasets}
    groups = len(seeds)*len(models)*len(datasets)
    count = len(seeds)*sum(len(entities[d])*(5 if m=='PatchTST' else 4) for d in datasets for m in models)
    print(f'{groups} groups; {count} entity/method evaluations; seeds={seeds}', flush=True)
    if args.plan_only:
        print(json.dumps({name: {role: config['datasets'][name][role] for role in ['history', 'new', 'ratios']} for name in datasets},indent=2))
        return
    inputs, missing = {}, []
    for name in datasets:
        spec = config['datasets'][name]
        required = {Path('train')/e for e in spec['history']}
        required.update(Path(split)/e for e in spec['new'] for split in ['train','test','label'])
        if spec.get('manifest'):
            required.update(Path('metadata')/(Path(e).stem+'.test.npz') for e in spec['new'])
            manifest = path.parent/spec['manifest']
            inputs[f'config/{spec["manifest"]}'] = digest(manifest)
        for relative in sorted(required):
            source = Path(config['data_root'])/name/relative
            if not source.is_file():
                missing.append(str(source))
            else:
                inputs[f'{name}/{relative}'] = digest(source)
    if missing:
        print(f'Missing {len(missing)} data/metadata files. First 12:\n'+'\n'.join(missing[:12]),file=sys.stderr)
        sys.exit(2)
    from roma.data.telemetry import load_targets, select_features
    invalid = []
    for name in datasets:
        spec = config['datasets'][name]
        for entity in spec['new']:
            indices = select_features(spec, spec['ratios'][entity], config['training']['dimension_seed'], path.parent)
            try:
                load_targets(Path(config['data_root'])/name, spec, entity, indices)
            except ValueError as error:
                invalid.append(f'{name}: {error}')
    if invalid:
        raise ValueError('Invalid new entities:\n' + '\n'.join(invalid))
    if args.dry_run:
        print('Required files and target anomaly coverage checked; SHA256 fingerprints calculated. No training started.')
        return
    if args.jobs < 1 or args.threads < 1:
        raise ValueError('jobs and threads must be positive')
    code = {str(p.relative_to(ROOT)):digest(p) for folder in ['roma','scripts'] for p in sorted((ROOT/folder).rglob('*.py'))}
    plan = dict(tag=args.tag,seeds=seeds,models=models,datasets=datasets,entities=entities,
                config=config,input_sha256=inputs,code_sha256=code,device=args.device)
    output.mkdir(parents=True,exist_ok=True)
    with (output/'.run.lock').open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f'Another launcher is using {output}')
        plan_path = output/'plan.json'
        if plan_path.exists():
            if json.loads(plan_path.read_text()) != plan:
                raise ValueError('Configuration, code, data, device or seeds changed. Choose a new --output; refusing mixed results.')
        else:
            plan_path.write_text(json.dumps(plan,indent=2)+'\n')
        jobs = []
        for seed in seeds:
            for model in models:
                for dataset in datasets:
                    command=[sys.executable,str(Path(__file__).resolve()),'--worker','--config',str(path),
                             '--data-root',config['data_root'],'--output',str(output),'--tag',args.tag,
                             '--seed',str(seed),'--model',model,'--dataset',dataset,'--device',args.device,'--threads',str(args.threads)]
                    jobs.append(dict(id=f's{seed}_{model}_{dataset}',command=command))
        def interrupt(signum,frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM,interrupt)
        from roma.training.scheduler import run_jobs
        failed = run_jobs(jobs, output, concurrency=args.jobs, threads=args.threads, cwd=ROOT,
                          max_retries=args.max_retries,retry_delay=args.retry_delay,worker_timeout=args.worker_timeout)
        if failed:
            raise RuntimeError(f'{len(failed)} worker groups failed. Resume identical command. See progress.json and workers/.')
        if len(seeds)>=2:
            from roma.evaluation.statistics import report
            report(plan,output)
        else:
            from roma.evaluation.statistics import collect
            collect(plan,output)
            print('Single-seed run audited; statistical inference requires at least two seeds.')


if __name__=='__main__':
    main()
