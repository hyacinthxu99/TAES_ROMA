"""Seed-level paired inference; standard library only, no training imports."""
import csv
import itertools
import json
import math
from pathlib import Path
import random
import statistics as st

METRICS = ('AUC-ROC', 'VUS-ROC')
ESA_MISSIONS = ('ESA-M1', 'ESA-M2', 'ESA-M3')
METHODS = ('lora', 'full', 'pretrained_full_finetune', 'direct_pretrain_align')
NAMES = {'lora': 'ROMA', 'full': 'Scratch', 'pretrained_full_finetune': 'Pretrained-FT',
         'direct_pretrain_align': 'Direct-align', 'patchtsad_channel_independent': 'PatchCI'}


def methods(model):
    return METHODS + (('patchtsad_channel_independent',) if model in ('PatchTST', 'PatchTSTAD') else ())


def sign_flip_p(differences):
    """Two-sided sign randomization; exact for <=16 seeds, otherwise Monte Carlo."""
    n = len(differences)
    observed = abs(sum(differences))
    tolerance = 1e-12 * max(1., sum(abs(x) for x in differences))
    if n <= 16:
        sums = (sum(s*x for s, x in zip(signs, differences))
                for signs in itertools.product((-1, 1), repeat=n))
        return sum(abs(x) >= observed-tolerance for x in sums) / (2**n)
    rng = random.Random(74123)
    extreme = sum(abs(sum(rng.choice((-1, 1))*x for x in differences)) >= observed-tolerance
                  for _ in range(50000))
    return (extreme+1)/50001


def bootstrap_ci(values, draws=10000):
    rng = random.Random(19237)
    n = len(values)
    samples = sorted(st.fmean(rng.choices(values, k=n)) for _ in range(draws))
    return samples[int(.025*(draws-1))], samples[int(.975*(draws-1))]


def holm(pvalues):
    adjusted = [None] * len(pvalues)
    running = 0.
    for rank, index in enumerate(sorted(range(len(pvalues)), key=pvalues.__getitem__)):
        running = max(running, min(1., (len(pvalues)-rank)*pvalues[index]))
        adjusted[index] = running
    return adjusted


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect(plan, output):
    """Require every predeclared seed/entity/method; never silently drop failures."""
    records, errors, missing = {}, [], []
    for seed in plan['seeds']:
        root = Path(output)/'results'/f'seed{seed}'
        for path in sorted(root.glob('anomaly_detection_*_match/*.json')):
            data = json.loads(path.read_text())
            entries = data.values() if data and all(isinstance(x, dict) for x in data.values()) else [data]
            for row in entries:
                if row.get('experiment_tag') != f"{plan['tag']}_s{seed}":
                    continue
                if row.get('experiment_stage') != 'stage01_main':
                    continue
                dataset = row.get('data', row.get('dataset'))
                key = (seed, dataset, row.get('model'), row.get('file_name'), row.get('method'))
                if row.get('method') == 'lora' and row.get('adapter_variant', 'full') != 'full':
                    continue
                if row.get('method') == 'full' and row.get('entity_role') != 'simulation_full_retraining':
                    continue
                if all(isinstance(row.get(metric), (int, float)) and math.isfinite(row[metric])
                       for metric in METRICS):
                    records[key] = row
                elif row.get('error') or 'error' in str(row.get('method', '')):
                    errors.append(dict(seed=seed, file=str(path), method=row.get('method')))
    ordered = []
    for seed in plan['seeds']:
        for dataset in plan['datasets']:
            for model in plan['models']:
                for entity in plan['entities'][dataset]:
                    for method in methods(model):
                        key = (seed, dataset, model, entity, method)
                        if key not in records:
                            missing.append(dict(seed=seed, dataset=dataset, model=model, entity=entity, method=method))
                        else:
                            ordered.append(dict(records[key], training_seed=seed))
    audit = dict(expected=len(ordered)+len(missing), completed=len(ordered), missing=len(missing),
                 missing_tasks=missing, historical_error_records=errors, complete=not missing)
    (Path(output)/'completion_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    # Keep all scalar metrics/resources for later analysis and retain the original JSON files.
    columns = sorted({key for row in ordered for key, value in row.items()
                      if value is None or isinstance(value, (str, int, float, bool))})
    write_csv(Path(output)/'all_results.csv', [{key: row.get(key) for key in columns} for row in ordered])
    if missing:
        raise RuntimeError(f'{len(missing)} results missing; inference withheld. See completion_audit.json; resume same tag.')
    return records


def report(plan, output):
    output = Path(output)
    records = collect(plan, output)
    seed_rows, summaries, comparisons = [], [], []
    groups = {dataset: [dataset] for dataset in plan['datasets']}
    if all(mission in groups for mission in ESA_MISSIONS):
        groups['ESA-ADB'] = ESA_MISSIONS
    for dataset, members in groups.items():
        aggregation = 'equal_missions' if dataset == 'ESA-ADB' else 'equal_entities'
        for model in plan['models']:
            values = {}
            for method in methods(model):
                for metric in METRICS:
                    vals = []
                    for seed in plan['seeds']:
                        value = 100 * st.fmean(
                            st.fmean(records[(seed, mission, model, entity, method)][metric]
                                     for entity in plan['entities'][mission])
                            for mission in members)
                        vals.append(value)
                        seed_rows.append(dict(dataset=dataset, model=model, method=NAMES[method], metric=metric,
                                              seed=seed, entity_count=sum(len(plan['entities'][m]) for m in members),
                                              aggregation=aggregation,
                                              value_percent=value))
                    values[method, metric] = vals
                    lo, hi = bootstrap_ci(vals)
                    summaries.append(dict(dataset=dataset, model=model, method=NAMES[method], metric=metric,
                                          aggregation=aggregation, n_seeds=len(vals), mean=st.fmean(vals), std=st.stdev(vals),
                                          ci95_low=lo, ci95_high=hi))
            family = []
            for baseline in methods(model)[1:]:
                for metric in METRICS:
                    differences = [a-b for a, b in zip(values['lora', metric], values[baseline, metric])]
                    lo, hi = bootstrap_ci(differences)
                    row = dict(dataset=dataset, model=model, baseline=NAMES[baseline], metric=metric,
                               aggregation=aggregation,
                               n_pairs=len(differences), delta_pp=st.fmean(differences), delta_ci95_low=lo,
                               delta_ci95_high=hi, p_raw=sign_flip_p(differences),
                               wins=sum(x > 0 for x in differences), ties=sum(x == 0 for x in differences))
                    family.append(row)
            for row, p in zip(family, holm([row['p_raw'] for row in family])):
                row.update(p_holm_family=p, significant_family=p < .05,
                           roma_better_family=p < .05 and row['delta_pp'] > 0)
            comparisons.extend(family)
    for row, p in zip(comparisons, holm([row['p_raw'] for row in comparisons])):
        row.update(p_holm_global=p, significant_global=p < .05)
    write_csv(output/'seed_means.csv', seed_rows)
    write_csv(output/'mean_std_ci.csv', summaries)
    write_csv(output/'paired_significance.csv', comparisons)
    lines = [f"Experiment: {plan['tag']}", f"Seeds: {plan['seeds']}",
             'Units: percent. Entities are averaged within each seed. Values are mean ± sample SD, with 95% bootstrap CI (10000 resamples).',
             'ESA-ADB: equal-weight average of M1/M2/M3 entity means within each seed; reported only when all three missions are present.',
             'Tests: seed-level paired two-sided sign flips; Holm correction within dataset/backbone and globally.',
             'Intervals are not multiplicity-adjusted. Inference concerns training randomness on the fixed datasets.', '',
             'Mean ± standard deviation:']
    for row in summaries:
        lines.append(f"{row['dataset']} / {row['model']} / {row['method']} / {row['metric']}: "
                     f"{row['mean']:.2f} ± {row['std']:.2f} [{row['ci95_low']:.2f}, {row['ci95_high']:.2f}]")
    lines += ['', 'ROMA comparisons (differences in percentage points; family Holm p):']
    for row in comparisons:
        lines.append(f"{row['dataset']} / {row['model']} / {row['baseline']} / {row['metric']}: "
                     f"Δ={row['delta_pp']:+.2f}, p={row['p_holm_family']:.6g}, "
                     f"{'significant' if row['significant_family'] else 'not significant'}")
    (output/'statistical_report.txt').write_text('\n'.join(lines)+'\n')
    print(f'Statistical report: {output}/statistical_report.txt', flush=True)
