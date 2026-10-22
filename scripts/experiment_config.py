"""Resolve entity splits and target widths from the local datasets."""
import json
import math
from pathlib import Path
import random


def resolve_datasets(config, names, config_dir):
    training = config['training']
    lower, upper = training['target_ratio']
    if not 0 < lower <= upper <= 1:
        raise ValueError('target_ratio must satisfy 0 < lower <= upper <= 1')
    if not 0 < training['pretrain_ratio'] <= 1:
        raise ValueError('pretrain_ratio must be in (0, 1]')
    for name in names:
        spec = config['datasets'][name]
        folder = Path(config['data_root'])/name/'test'
        excluded = set(spec.get('exclude', []))
        entities = sorted(p.name for p in folder.glob('*.csv') if p.is_file() and p.name not in excluded)
        if len(entities) < 2:
            raise ValueError(f'{name}: found {len(entities)} entities in {folder}; at least 2 are needed for disjoint history/new groups')
        random.Random(f'{config["split_seed"]}:{name}').shuffle(entities)
        cut = (len(entities)+1)//2
        spec['history'], spec['new'] = entities[:cut], entities[cut:]
        variables = spec['dimensions']
        if spec.get('manifest'):
            manifest = json.loads((Path(config_dir)/spec['manifest']).read_text())
            variables = len(manifest['source_channels'])
        if int(round(variables*training['pretrain_ratio'], 10)) < 1:
            raise ValueError(f'{name}: pretrain_ratio selects no variables')
        minimum = math.ceil(round(variables*lower, 10))
        maximum = math.floor(round(variables*upper, 10))
        if minimum > maximum:
            raise ValueError(f'{name}: no integer variable count fits target_ratio')
        spec['ratios'] = {}
        for entity in spec['new']:
            rng = random.Random(f'{training["dimension_seed"]}:{name}:{entity}')
            spec['ratios'][entity] = rng.randint(minimum, maximum)/variables
