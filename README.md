# Neural ODEs vs ResNets under Irregular Sampling — A Toy Benchmark (PyTorch)

A small, reproducible experimental study comparing a discrete-time ResNet baseline against a continuous-time Neural ODE on synthetic damped-oscillator trajectories with missing/irregular observations.

This project is structured as a research artifact: **baselines + metrics + ablations + plots + reproducibility**.

---

## Problem
Neural ODEs provide a continuous-time parameterization of dynamics and are often motivated by irregularly sampled time series. However, it is not always clear when they outperform simpler discrete-time baselines (e.g., residual networks) under a controlled compute budget.

We run a controlled benchmark on synthetic trajectories where we can explicitly vary **missingness** and **solver choice**.

---

## Method
### Dataset (Synthetic ODE)
We generate trajectories from a damped oscillator:
- state: \(x_t = (position, velocity)\)
- Gaussian observation noise
- an observation mask to simulate missing/irregular measurements

### Models (Baselines)
- **ResNetStep** (discrete-time): predicts next step with a residual update  
  \(x_{t+1} = x_t + f(x_t)\)
- **Neural ODE** (continuous-time): learns \(dx/dt = f(x)\) and integrates with `torchdiffeq.odeint`

### Objective
Masked validation **MSE** computed only on observed (non-missing) points.

---

## Experiments
We report masked validation MSE and run the following:
- **Base:** `missing_rate=0.3`, `solver=rk4`
- **Ablation 1 (Solver):** `rk4` vs `dopri5`
- **Ablation 2 (Missingness):** `missing_rate ∈ {0.0, 0.6}` (solver fixed to rk4)

### Outputs
- Metrics summary: `results/tables/summary.csv`
- Per-run JSON logs: `results/tables/*_metrics.json`
- Figures:
  - `results/figures/mse_by_run.png`
  - `results/figures/mse_vs_missing_rate.png`

---

## Quickstart
### Install
```bash
pip install -r requirements.txt
