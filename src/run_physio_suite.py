from __future__ import annotations

import argparse
import subprocess
import sys


def run(cmd):
    print("\n> " + " ".join(cmd))
    subprocess.check_call(cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default=r"data/physionet2012_raw")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=128)
    args = ap.parse_args()

    py = sys.executable

    for s in args.seeds:
        run([py, "-m", "src.train_physio_gru", "--data_root", args.data_root, "--seed", str(s),
             "--epochs", str(args.epochs), "--batch_size", str(args.batch_size), "--hidden", str(args.hidden)])
        run([py, "-m", "src.train_physio_lastmlp", "--data_root", args.data_root, "--seed", str(s),
             "--epochs", str(args.epochs), "--batch_size", str(args.batch_size), "--hidden", str(args.hidden)])
        run([py, "-m", "src.train_physio_grud", "--data_root", args.data_root, "--seed", str(s),
             "--epochs", str(args.epochs), "--batch_size", str(args.batch_size), "--hidden", str(args.hidden)])

    run([py, "-m", "src.eval_physio_models"])
    run([py, "-m", "src.plots_physio_compare", "--seeds"] + [str(s) for s in args.seeds])


if __name__ == "__main__":
    main()
