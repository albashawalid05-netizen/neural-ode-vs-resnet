# Notes

## Toy study
- src/data.py: synthetic damped-oscillator dataset with irregular sampling and controlled missingness
- src/models.py: ResNetStep and Neural ODE (torchdiffeq) implementations for the toy study
- src/train.py: training runner that writes per-run JSON results
- src/eval.py: aggregates JSON runs into summary tables
- src/plots.py: generates toy-study figures under results/figures/

## PhysioNet 2012 (Set A)
- src/physionet2012.py: parses PhysioNet patient files into (t, x, m) with normalization and missingness mask, and creates deterministic train/val/test splits (2800/600/600, D=23)
- src/train_physio_gru.py: GRU baseline training and evaluation
- src/train_physio_lastmlp.py: LastObsMLP baseline
- src/models_physio_grud.py and src/train_physio_grud.py: GRU-D implementation and training
- src/models_physio_odernn.py and src/train_physio_odernn.py: ODE-RNN implementation and training
- src/run_physio_suite.py: runs the PhysioNet suite across seeds/models
- src/eval_physio_models.py: aggregates results/physio_runs/physio_*_seed*.json and computes mean ± 95% CI for AUROC/AUPRC, writing results/tables/physio_summary.csv
- src/plots_physio.py and src/plots_physio_compare.py: PhysioNet plots, including AUROC by observation-count quartile

## Reproducibility
- RUN_PHYSIONET.ps1 reproduces the PhysioNet suite on Windows given the dataset under data/physionet2012_raw/
- Reported PhysioNet metrics use mean ± 95% CI across 10 seeds (0–9)
