\# Neural ODEs vs ResNets under Irregular Sampling: A Small Controlled Study



\*\*Author:\*\* <YOUR NAME>  

\*\*Code:\*\* <GITHUB LINK>  

\*\*Date:\*\* 2026-01-28



\## Abstract

We compare a discrete-time residual network (ResNet) baseline against a continuous-time Neural Ordinary Differential Equation (Neural ODE) model on a controlled synthetic dynamical system (damped oscillator) with irregular/missing observations. We report masked validation MSE and perform two simple ablations: (i) ODE solver choice (rk4 vs dopri5) and (ii) observation missingness. Under our setup, the Neural ODE achieves substantially lower masked MSE than the ResNet baseline and is robust to solver choice.



\## 1. Problem

Continuous-time models are often motivated by irregularly sampled time series. However, it is not always clear when they outperform simpler discrete-time baselines under a controlled compute budget. We design a toy benchmark where missingness and solver settings can be varied while keeping the learning problem simple and reproducible.



\## 2. Methods

\### Dataset

We generate trajectories from a damped oscillator with state \\(x\_t = (position, velocity)\\). Observations are corrupted with Gaussian noise (\\(\\sigma = 0.05\\)) and a binary observation mask simulates missing/irregular measurements.



\### Models (Baselines)

\- \*\*ResNetStep (discrete-time):\*\* iterated residual update \\(x\_{t+1} = x\_t + f\_\\theta(x\_t)\\).

\- \*\*Neural ODE (continuous-time):\*\* learns \\(dx/dt = f\_\\theta(x)\\) and integrates with `torchdiffeq.odeint`.



\### Training objective

We minimize \*\*masked MSE\*\*, computed only over observed points.



\## 3. Experimental Setup

\- Sequences: 3000, length 50, \\(dt=0.1\\)

\- Train/val split: 80/20

\- Epochs: 15, Adam, lr=1e-3

\- Hardware: NVIDIA RTX 4070 Laptop GPU

\- Primary metric: masked validation MSE



\## 4. Results

Table 1 summarizes the main results (masked validation MSE). Neural ODE significantly outperforms the ResNet baseline across all runs.



\*\*Table 1: Masked validation MSE (lower is better)\*\*



| Run | Missing rate | Solver | ResNet MSE | Neural ODE MSE |

|---|---:|---|---:|---:|

| base | 0.3 | rk4 | 0.111581 | 0.011062 |

| solver ablation | 0.3 | dopri5 | 0.111581 | 0.011063 |

| missing ablation | 0.0 | rk4 | 0.145135 | 0.011394 |

| missing ablation | 0.6 | rk4 | 0.109933 | 0.010556 |



Figures are provided in `results/figures/`:

\- `mse\_by\_run.png`

\- `mse\_vs\_missing\_rate.png`



\## 5. Ablations

\### 5.1 Solver choice (rk4 vs dopri5)

Changing the ODE solver from rk4 to dopri5 yields negligible difference in masked validation MSE for the Neural ODE (0.011062 vs 0.011063). This suggests robustness to solver choice under the current tolerances and time grid.



\### 5.2 Missingness (0.0 vs 0.6)

Varying missingness changes the baseline behavior more than the Neural ODE in this configuration. Neural ODE remains stable around ~0.010–0.011 MSE.



\## 6. Error Analysis (qualitative)

We observe:

\- The ResNet baseline accumulates error over longer horizons due to step-wise rollout.

\- The Neural ODE produces smoother trajectories and maintains lower error under missingness.

A more complete error analysis would visualize individual trajectories and report error vs. time horizon (planned extension).



\## 7. Limitations

\- Synthetic toy system; results may not transfer directly to real-world irregular time series.

\- Limited hyperparameter tuning and only a small set of ablations.

\- No multi-seed confidence intervals; future work should average results over multiple seeds and add additional baselines (GRU-D / Neural CDE).



\## References

1\. Chen et al., \*Neural Ordinary Differential Equations\*, NeurIPS 2018.

2\. torchdiffeq: differentiable ODE solvers in PyTorch.



