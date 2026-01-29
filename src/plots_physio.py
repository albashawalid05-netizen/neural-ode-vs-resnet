# src/plots_physio.py
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score

from src.physionet2012 import make_splits, PhysioNet2012Dataset, collate_fn
import torch
from torch.utils.data import DataLoader
from src.models import GRUClassifier


@torch.no_grad()
def predict_probs_gru(model, loader, device):
    model.eval()
    probs_all = []
    y_all = []
    obs_counts = []

    for times, x, mask, lengths, y in loader:
        times = times.to(device)
        x = x.to(device)
        mask = mask.to(device)
        lengths = lengths.to(device)

        logits = model(times, x, mask, lengths).squeeze(-1)
        probs = torch.sigmoid(logits).cpu().numpy()
        y_np = y.squeeze(-1).numpy()

        # count observed entries per patient (sum mask across time/features)
        obs = mask.cpu().numpy().sum(axis=(1, 2))

        probs_all.append(probs)
        y_all.append(y_np)
        obs_counts.append(obs)

    probs_all = np.concatenate(probs_all)
    y_all = np.concatenate(y_all)
    obs_counts = np.concatenate(obs_counts)
    return probs_all, y_all, obs_counts


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("results/figures", exist_ok=True)

    # fixed split
    train_s, val_s, test_s, D = make_splits("data/physionet2012_raw", seed=0)

    # Load GRU seed0 weights (we already have it saved as json only; so we just re-train fast here for plot)
    # To keep it simple/robust: train 3 epochs only and plot. (Good enough for the slice plot.)
    train_ds = PhysioNet2012Dataset(train_s)
    test_ds = PhysioNet2012Dataset(test_s)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, collate_fn=collate_fn)

    model = GRUClassifier(d_in=D, hidden=128).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(3):
        for times, x, mask, lengths, y in train_loader:
            times, x, mask, lengths = times.to(device), x.to(device), mask.to(device), lengths.to(device)
            y = y.to(device).squeeze(-1)
            logits = model(times, x, mask, lengths).squeeze(-1)
            loss = loss_fn(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

    probs, y_true, obs = predict_probs_gru(model, test_loader, device)

    # bucket by obs quartiles
    qs = np.quantile(obs, [0.25, 0.5, 0.75])
    bins = [-np.inf, qs[0], qs[1], qs[2], np.inf]
    labels = ["Q1 (few obs)", "Q2", "Q3", "Q4 (many obs)"]

    aurocs = []
    for i in range(4):
        m = (obs > bins[i]) & (obs <= bins[i+1])
        if m.sum() < 10:
            aurocs.append(np.nan)
        else:
            aurocs.append(roc_auc_score(y_true[m], probs[m]))

    plt.figure()
    plt.plot(range(1, 5), aurocs, marker="o")
    plt.xticks(range(1, 5), labels, rotation=15, ha="right")
    plt.ylim(0.5, 0.95)
    plt.ylabel("AUROC (test)")
    plt.title("GRU performance vs observation count (PhysioNet 2012)")
    plt.tight_layout()

    out_path = "results/figures/physio_auroc_by_obs_quartile.png"
    plt.savefig(out_path, dpi=200)
    print("saved:", out_path)


if __name__ == "__main__":
    main()
