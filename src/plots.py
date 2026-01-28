import pandas as pd
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv("results/tables/summary.csv")

    # Plot 1: Compare models across runs (bar-style via simple lines)
    plt.figure()
    for model_col in ["resnet_val_mse", "neuralode_val_mse"]:
        plt.plot(df["run"], df[model_col], marker="o", label=model_col)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Validation MSE (masked)")
    plt.xlabel("Run")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/figures/mse_by_run.png", dpi=200)
    plt.close()

    # Plot 2: Effect of missing_rate for each model (filter rk4 runs)
    df2 = df[df["solver"] == "rk4"].copy()
    df2 = df2.sort_values("missing_rate")

    plt.figure()
    plt.plot(df2["missing_rate"], df2["resnet_val_mse"], marker="o", label="ResNet")
    plt.plot(df2["missing_rate"], df2["neuralode_val_mse"], marker="o", label="Neural ODE")
    plt.xlabel("Missing rate")
    plt.ylabel("Validation MSE (masked)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/figures/mse_vs_missing_rate.png", dpi=200)
    plt.close()

    print("Saved figures to results/figures/")


if __name__ == "__main__":
    main()
