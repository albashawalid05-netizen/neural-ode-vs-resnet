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

