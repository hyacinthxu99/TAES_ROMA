"""Train-only normalization and windows for the fixed reproduction protocol."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


def read_matrix(path, header):
    values = pd.read_csv(path, header=header).fillna(0).to_numpy()
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError(f'Invalid finite telemetry matrix: {path}')
    return values


def select_features(spec, ratio, seed=2025, config_dir=None):
    """Legacy random feature choice; ESA nested physical channels, including one-hot groups."""
    rng = np.random.default_rng(seed)
    if spec.get('manifest'):
        manifest = json.loads((Path(config_dir)/spec['manifest']).read_text())
        source = manifest['source_channels']
        selected = [source[i] for i in rng.permutation(len(source))[:int(round(len(source)*ratio, 10))]]
        result = [i for channel in selected for i, original in enumerate(manifest['feature_to_channel'])
                  if original == channel]
    else:
        result = rng.choice(spec['dimensions'], size=int(round(spec['dimensions']*ratio, 10)), replace=False).tolist()
    if not result or len(set(result)) != len(result):
        raise ValueError('Empty or duplicated feature selection')
    return result


class Windows(Dataset):
    """Windows never cross entries of arrays (entities); no validation split."""
    def __init__(self, arrays, seq_len, labels=None):
        self.arrays = [np.asarray(x, dtype=np.float32) for x in arrays]
        self.seq_len = int(seq_len)
        self.labels = labels
        counts = [len(x)-self.seq_len+1 for x in self.arrays]
        if not counts or min(counts) < 1:
            raise ValueError('Each sequence must contain at least seq_len points')
        self.ends = np.cumsum(counts)

    def __len__(self):
        return int(self.ends[-1])

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        entity = int(np.searchsorted(self.ends, index, side='right'))
        offset = index-(int(self.ends[entity-1]) if entity else 0)
        x = self.arrays[entity][offset:offset+self.seq_len]
        y = np.zeros(self.seq_len, np.float32) if self.labels is None else self.labels[offset:offset+self.seq_len]
        return x, np.asarray(y, dtype=np.float32)


def pretraining_data(root, spec, indices, seq_len):
    """Pool history training data for scaling; keep windows within entities."""
    root = Path(root)
    arrays = []
    scaler = StandardScaler()
    for name in spec['history']:
        values = read_matrix(root/'train'/name, spec['csv_header'])
        if values.shape[1] != spec['dimensions']:
            raise ValueError(f'{name}: expected {spec["dimensions"]} features')
        selected = values[:, indices].astype(np.float32)
        scaler.partial_fit(selected)
        arrays.append(selected)
    arrays = [scaler.transform(x).astype(np.float32) for x in arrays]
    return Windows(arrays, seq_len)


def adaptation_data(root, spec, name, indices, seq_len):
    root = Path(root)
    train = read_matrix(root/'train'/name, spec['csv_header'])
    test = read_matrix(root/'test'/name, spec['csv_header'])
    if train.shape[1] != spec['dimensions'] or test.shape[1] != spec['dimensions']:
        raise ValueError(f'{name}: expected {spec["dimensions"]} features')
    labels, mask = load_targets(root, spec, name, indices)
    if len(labels) != len(test):
        raise ValueError(f'{name}: test features and labels have different lengths')
    scaler = StandardScaler().fit(train)
    train = scaler.transform(train)[:, indices]
    test = scaler.transform(test)[:, indices]
    return Windows([train], seq_len), Windows([test], seq_len, labels), labels, mask


def load_targets(root, spec, name, indices):
    root = Path(root)
    labels = read_matrix(root/'label'/name, spec['csv_header']).reshape(-1)
    if not np.isin(labels, [0, 1]).all():
        raise ValueError(f'{name}: expected binary test labels')
    mask = np.ones(len(labels), dtype=bool)
    if spec.get('manifest'):
        with np.load(root/'metadata'/f'{Path(name).stem}.test.npz', allow_pickle=False) as metadata:
            flags = metadata['channel_flags'][:, indices]
        valid = (flags & 28) == 0
        labels = (((flags & 1) != 0) & valid).any(axis=1).astype(np.float32)
        mask = valid.all(axis=1)
    if len(mask) != len(labels) or len(np.unique(labels[mask])) != 2:
        raise ValueError(f'{name}: selected channels need evaluable normal and anomalous test samples')
    return labels, mask
