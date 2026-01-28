import os
import json
import glob
import pandas as pd


def main():
    paths = sorted(glob.glob("results/tables/*_metrics.json"))
    rows = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        cfg = d["config"]
        rows.append({
            "run": os.path.basename(p).replace("_metrics.json", ""),
            "device": d.get("device", ""),
            "missing_rate": cfg["data"]["missing_rate"],
            "noise_std": cfg["data"]["noise_std"],
            "solver": cfg["ode"]["solver"],
            "epochs": cfg["train"]["epochs"],
            "resnet_val_mse": d["resnet_val_mse"],
            "neuralode_val_mse": d["neuralode_val_mse"],
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(by=["missing_rate", "solver", "run"]).reset_index(drop=True)
    out_csv = "results/tables/summary.csv"
    df.to_csv(out_csv, index=False)
    print("Wrote", out_csv)
    print(df)


if __name__ == "__main__":
    main()
