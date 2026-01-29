from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.metrics import roc_auc_score


def mean_ci95(x: np.ndarray):
    x = np.asarray(x, dtype=float)
    n = x.size
    m = float(x.mean())
    if n <= 1:
        return m, float("nan")
    s = float(x.std(ddof=1))
    h = float(t.ppf(0.975, df=n - 1) * s / np.sqrt(n))
    return m, h


def auroc_by_quartile(y_true, y_prob, obs_count):
    q = np.quantile(obs_count, [0.25, 0.5, 0.75])
    bins = np.digitize(obs_count, q, right=True)
    vals = []
    for k in range(4):
        idx = bins == k
        yt = y_true[idx]
        yp = y_prob[idx]
        if yt.size == 0 or yt.min() == yt.max():
            vals.append(np.nan)
        else:
            vals.append(float(roc_auc_score(yt, yp)))
    return np.array(vals, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_root", type=str, default=r".\results\preds\physio")
    ap.add_argument("--models", nargs="+", default=["gru", "lastmlp", "grud"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--out_png", type=str, default=r".\results\figures\physio_auroc_by_obs_quartile_models.png")
    ap.add_argument("--out_csv", type=str, default=r".\results\tables\physio_auroc_by_obs_quartile_models.csv")
    args = ap.parse_args()

    pred_root = Path(args.pred_root)
    rows = []
    for model in args.models:
        per_seed = []
        for s in args.seeds:
            f = pred_root / model / f"seed{s}.npz"
            d = np.load(f)
            y_true = d["y_true"].astype(int).reshape(-1)
            y_prob = d["y_prob"].astype(float).reshape(-1)
            obs_count = d["obs_count"].astype(float).reshape(-1)
            per_seed.append(auroc_by_quartile(y_true, y_prob, obs_count))
        per_seed = np.stack(per_seed, axis=0)
        for q in range(4):
            m, h = mean_ci95(per_seed[:, q])
            rows.append({"model": model, "quartile": q + 1, "auroc_mean": m, "auroc_ci95": h})

    df = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    plt.figure(figsize=(7.5, 4.5))
    for model in args.models:
        g = df[df["model"] == model].sort_values("quartile")
        x = g["quartile"].values
        y = g["auroc_mean"].values
        yerr = g["auroc_ci95"].values
        plt.errorbar(x, y, yerr=yerr, marker="o", capsize=3, label=model)

    plt.xticks([1, 2, 3, 4], ["Q1", "Q2", "Q3", "Q4"])
    plt.xlabel("Observation-count quartile")
    plt.ylabel("Test AUROC (mean ± 95% CI over seeds)")
    plt.title("PhysioNet 2012: AUROC vs missingness")
    plt.grid(True, alpha=0.3)
    plt.legend()
    Path(args.out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=200)
    print(f"Wrote {args.out_png}")
    print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
