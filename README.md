# ROMA

ROMA adapts time-series anomaly detectors to changing telemetry variables using
LoRA and channel adapters. It includes five backbone implementations.

## Directory layout

```text
adcs-simulation/     ADCS dataset simulation and anomaly injection
configs/            Experiment settings, dataset configuration and ESA channel mappings
data/               Dataset folders: train, test and label
docs/               Experiment protocol and validation records
roma/
   adapters/        Channel adapters
   backbones/       TranAD, TimesNet, DTAAD, AnomalyTransformer and PatchTST
   data/            Data loading, normalization and sliding windows
   evaluation/      Evaluation metrics and multiseed statistics
   layers/          Shared network layers
   training/        Pretraining, adaptation, baselines and parallel scheduling
   config.py        Model configuration
   graph.py         Correlation graph construction
   lora.py          LoRA injection and adapter weight saving/loading
   model.py         ROMA model
scripts/            Experiment and result aggregation scripts
tests/              Tests
```

`scripts/run_experiments.py` is the main script for running experiments.
`scripts/summarize_results.py` summarizes existing experiment results.

## ADCS dataset generation

`adcs-simulation/` generates the ADCS simulation dataset. It is based on
[gavincmartin/adcs-simulation](https://github.com/gavincmartin/adcs-simulation).
The [inject_anomaly.py](adcs-simulation/inject_anomaly.py) script is our own anomaly
injection code, used to generate anomalous test data and corresponding labels.

## Dataset sources

| Dataset | Source |
| --- | --- |
| MSL | NASA telemetry data from [Telemanom](https://github.com/khundman/telemanom) |
| SMAP | NASA telemetry data from [Telemanom](https://github.com/khundman/telemanom) |
| ESA-ADB | [Official project](https://github.com/kplabs-pl/ESA-ADB); [dataset download](https://zenodo.org/records/12528696) |

Dataset copies may also be obtained from other open-source anomaly detection
projects. Arrange them in the expected `train/`, `test/` and `label/` folders
before running experiments. 

## Installation

```bash
pip install -e ".[experiments]"
```

Datasets and model weights are not included in this repository.
