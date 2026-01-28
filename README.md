\# Neural ODEs vs ResNets under Irregular Sampling (Toy Study)



A small, reproducible research artifact comparing a discrete-time \*\*ResNetStep\*\* baseline vs a continuous-time \*\*Neural ODE\*\* under \*\*irregular sampling\*\* and \*\*controlled missingness\*\* on a synthetic damped oscillator dataset.



\- \*\*GitHub:\*\* https://github.com/albashawalid05-netizen/neural-ode-vs-resnet  

\- \*\*Mini-paper (PDF):\*\* paper/mini\_paper\_clean.pdf



---



\## Repo structure



\- `configs/` - base run + ablation configs  

\- `src/`

&nbsp; - `data.py` - synthetic damped oscillator + missingness mask

&nbsp; - `models.py` - ResNetStep + NeuralODE

&nbsp; - `train.py` - trains from a config; saves JSON results per run

&nbsp; - `eval.py` - aggregates results; writes `summary.csv`

&nbsp; - `plots.py` - generates PNG figures

\- `results/`

&nbsp; - `tables/summary.csv`

&nbsp; - `figures/mse\_by\_run.png`

&nbsp; - `figures/mse\_vs\_missing\_rate.png`

\- `paper/mini\_paper\_clean.pdf`



---



\## Reproduce (Windows / PowerShell)



\### Train (base + ablations)

```bash

python -m src.train --config .\\configs\\base.yaml

python -m src.train --config .\\configs\\ablation\_solver\_dopri5.yaml

python -m src.train --config .\\configs\\ablation\_missing\_0.yaml

python -m src.train --config .\\configs\\ablation\_missing\_60.yaml



