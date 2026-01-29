# src/train_physio_lastmlp.py
import os, json, argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import roc_auc_score, average_precision_score

from src.physionet2012 import make_splits, PhysioNet2012Dataset, collate_fn
from src.models import LastObsMLP


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_probs = []
    all_y = []
    for times, x, mask, lengths, y in loader:
        times = times.to(device)
        x = x.to(device)
        mask = mask.to(device)
        lengths = lengths.to(device)
        y = y.to(device)

        logits = model(times, x, mask, lengths).squeeze(-1)  # [B]
        probs = torch.sigmoid(logits).detach().cpu()
        all_probs.append(probs)
        all_y.append(y.squeeze(-1).detach().cpu())

    probs = torch.cat(all_probs).numpy()
    y_true = torch.cat(all_y).numpy()

    auroc = roc_auc_score(y_true, probs)
    auprc = average_precision_score(y_true, probs)
    return auroc, auprc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="data/physionet2012_raw")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # split seed ثابت عشان المقارنات عادلة
    train_s, val_s, test_s, D = make_splits(args.data_root, seed=0)

    train_ds = PhysioNet2012Dataset(train_s)
    val_ds = PhysioNet2012Dataset(val_s)
    test_ds = PhysioNet2012Dataset(test_s)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = LastObsMLP(d_in=D, hidden=args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n = 0

        for times, x, mask, lengths, y in train_loader:
            times = times.to(device)
            x = x.to(device)
            mask = mask.to(device)
            lengths = lengths.to(device)
            y = y.to(device).squeeze(-1)  # [B]

            logits = model(times, x, mask, lengths).squeeze(-1)  # [B]
            loss = loss_fn(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * y.shape[0]
            n += y.shape[0]

        train_loss = total_loss / max(1, n)
        val_auroc, val_auprc = evaluate(model, val_loader, device)
        print(f"epoch {epoch:02d} | train_loss={train_loss:.4f} | val_auroc={val_auroc:.4f} | val_auprc={val_auprc:.4f}")

        if val_auroc > best_val:
            best_val = val_auroc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test_auroc, test_auprc = evaluate(model, test_loader, device)

    os.makedirs("results/physio_runs", exist_ok=True)
    out = {
        "model": "lastmlp",
        "seed": args.seed,
        "device": device,
        "D": D,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "hidden": args.hidden,
        "best_val_auroc": float(best_val),
        "test_auroc": float(test_auroc),
        "test_auprc": float(test_auprc),
    }

    path = f"results/physio_runs/lastmlp_seed{args.seed}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    print("saved:", path)
    print("TEST:", "auroc=", test_auroc, "auprc=", test_auprc)


if __name__ == "__main__":
    main()
