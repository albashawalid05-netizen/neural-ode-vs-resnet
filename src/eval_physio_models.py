from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t


def mean_ci95(x: np.ndarray):
    x = np.asarray(x, dtype=float)
    n = x.size
    m = float(x.mean())
    if n <= 1:
        return m, float("nan")
    s = float(x.std(ddof=1))
    h = float(t.ppf(0.975, df=n - 1) * s / np.sqrt(n))
    return m, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", type=str, default=r".\results\physio_runs")
    ap.add_argument("--out_csv", type=str, default=r".\results\tables\physio_summary.csv")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    rows = []
    for p in sorted(runs_dir.glob("physio_*_seed*.json")):
        obj = json.loads(p.read_text())
        rows.append(
            {
                "model": obj["model"],
                "seed": int(obj["seed"]),
                "auroc": float(obj.get("test_auroc", obj.get("test", {}).get("auroc", np.nan))),
                "auprc": float(obj.get("test_auprc", obj.get("test", {}).get("auprc", np.nan))),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No run json files found in {runs_dir}")

    out = []
    for model, g in df.groupby("model"):
        auroc_m, auroc_h = mean_ci95(g["auroc"].values)
        auprc_m, auprc_h = mean_ci95(g["auprc"].values)
        out.append(
            {
                "model": model,
                "n_seeds": int(g.shape[0]),
                "auroc_mean": auroc_m,
                "auroc_ci95": auroc_h,
                "auprc_mean": auprc_m,
                "auprc_ci95": auprc_h,
            }
        )

    out_df = pd.DataFrame(out).sort_values("auroc_mean", ascending=False)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv}")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
