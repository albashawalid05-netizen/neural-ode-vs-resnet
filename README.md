# Neural ODEs vs ResNets under Irregular Sampling (Toy Study)
## PhysioNet 2012 (Windows PowerShell)

### Data layout (not included in repo)
Place PhysioNet 2012 Set A here:
data/physionet2012_raw/
  Outcomes-a.txt
  set-a/set-a/*.txt

### Install
pip install -r requirements.txt

### Run
powershell -ExecutionPolicy Bypass -File .\RUN_PHYSIONET.ps1

### Outputs
results\tables\physio_summary.csv
results\figures\physio_auroc_by_obs_quartile_models.png
results\tables\physio_auroc_by_obs_quartile_models.csv
### Results (PhysioNet 2012, 10 seeds: 0–9)
Metric format: mean ± 95% CI across seeds

| Model      | AUROC          | AUPRC          |
|-----------|-----------------|----------------|
| GRU D     | 0.824 ± 0.015   | 0.458 ± 0.042  |
| GRU       | 0.815 ± 0.005   | 0.471 ± 0.006  |
| ODE RNN   | 0.702 ± 0.055   | 0.320 ± 0.061  |
| LastObsMLP| 0.666 ± 0.002   | 0.270 ± 0.003  |

Note: ODE RNN shows higher variance across seeds under the same training budget, while the simple LastObsMLP baseline underperforms as expected.

The summary table is saved at results/tables/physio_summary.csv
A quartile analysis plot is saved at results/figures/physio_auroc_by_obs_quartile_models.png


A small, reproducible research artifact comparing a discrete-time **ResNetStep** baseline vs a continuous-time **Neural ODE** under **irregular sampling** and **controlled missingness** on a synthetic damped oscillator dataset.

- **GitHub:** https://github.com/albashawalid05-netizen/neural-ode-vs-resnet  
- **Mini-paper (PDF):** https://github.com/albashawalid05-netizen/neural-ode-vs-resnet/blob/main/paper/mini_paper_clean.pdf

---

## Repo structure

- `configs/` - base run + ablation configs  
- `src/`
  - `data.py` - synthetic damped oscillator + missingness mask
  - `models.py` - ResNetStep + NeuralODE
  - `train.py` - trains from a config; saves JSON results per run
  - `eval.py` - aggregates results; writes `summary.csv`
  - `plots.py` - generates PNG figures
- `results/`
  - `tables/summary.csv`
  - `figures/mse_by_run.png`
  - `figures/mse_vs_missing_rate.png`
- `paper/mini_paper_clean.pdf`

---

## Reproduce (Windows / PowerShell)

### Train (base + ablations)
```bash
python -m src.train --config .\configs\base.yaml
python -m src.train --config .\configs\ablation_solver_dopri5.yaml
python -m src.train --config .\configs\ablation_missing_0.yaml
python -m src.train --config .\configs\ablation_missing_60.yaml

---

## Real data upgrade: PhysioNet 2012 (irregular ICU time series)

This repo now includes a real-world irregularly-sampled + missing clinical time-series benchmark (PhysioNet/CinC 2012, Set A) for **binary mortality classification**.

### How to download the dataset (Windows / PowerShell)
```powershell
cd data\physionet2012_raw
$setA = "https://archive.physionet.org/pn3/challenge/2012/set-a.zip"
$outA = "https://archive.physionet.org/pn3/challenge/2012/Outcomes-a.txt"
Invoke-WebRequest -Uri $setA -OutFile "set-a.zip"
Invoke-WebRequest -Uri $outA -OutFile "Outcomes-a.txt"
Expand-Archive -Path ".\set-a.zip" -DestinationPath ".\set-a" -Force
# If files end up nested in set-a\set-a, move them one level up:
Move-Item .\set-a\set-a\*.txt .\set-a\ -ErrorAction SilentlyContinue
Remove-Item .\set-a\set-a -Recurse -Force -ErrorAction SilentlyContinue
## PhysioNet 2012 (Set A) suite (GRU / LastObsMLP / GRU-D)

```powershell
python -m src.run_physio_suite --data_root data/physionet2012_raw --seeds 0 1 2 3 4 5 6 7 8 9 --epochs 10 --batch_size 64 --hidden 128

## Paper
mini paper (includes toy study and PhysioNet section): paper/mini_paper_clean.pdf
Source: paper/mini_paper_clean.tex

