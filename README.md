# Neural ODEs vs ResNets under Irregular Sampling (Toy Study)

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
