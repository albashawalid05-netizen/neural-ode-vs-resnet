import os, json, math, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

from src.physionet2012 import make_splits, PhysioNet2012Dataset, collate_fn

class ODEFunc(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, h):
        return self.net(h)

def ode_step(h, dt, f, method="rk4"):
    dt = dt.clamp_min(0.0)
    if method == "euler":
        return h + dt * f(h)
    k1 = f(h)
    k2 = f(h + 0.5 * dt * k1)
    k3 = f(h + 0.5 * dt * k2)
    k4 = f(h + dt * k3)
    return h + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

class ODERNNClassifier(nn.Module):
    def __init__(self, D, hidden=128, method="rk4"):
        super().__init__()
        self.D = D
        self.hidden = hidden
        self.method = method
        self.f = ODEFunc(hidden)
        self.gru = nn.GRUCell(input_size=2*D + 1, hidden_size=hidden)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 1)
        )

    def forward(self, t, x, m):
        B, T, D = x.shape
        h = torch.zeros(B, self.hidden, device=x.device, dtype=x.dtype)
        t = t.to(x.dtype)

        for i in range(T):
            xi = x[:, i, :]
            mi = m[:, i, :]
            obs_any = (mi.sum(dim=1, keepdim=True) > 0)

            if i == 0:
                dt = torch.zeros(B, 1, device=x.device, dtype=x.dtype)
            else:
                dt = (t[:, i] - t[:, i-1]).unsqueeze(1)
                h = ode_step(h, dt, self.f, method=self.method)

            inp = torch.cat([xi * mi, mi, dt], dim=1)
            h_new = self.gru(inp, h)
            h = torch.where(obs_any, h_new, h)

        return self.head(h).squeeze(1)

@torch.no_grad()
def eval_loader(model, loader, device):
    model.eval()
    ys, ps = [], []
    for batch in loader:
        t, x, m, rid, y = batch
        t = t.to(device)
        x = x.to(device)
        m = m.to(device)
        y = y.to(device).float().view(-1)

        logits = model(t, x, m)
        prob = torch.sigmoid(logits).view(-1)

        ys.append(y.detach().cpu().numpy())
        ps.append(prob.detach().cpu().numpy())

    y = np.concatenate(ys).reshape(-1)
    p = np.concatenate(ps).reshape(-1)

    if len(np.unique(y)) < 2:
        auroc = float("nan")
    else:
        auroc = float(roc_auc_score(y, p))
    auprc = float(average_precision_score(y, p))
    return auroc, auprc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="data/physionet2012_raw")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--method", type=str, default="rk4", choices=["rk4","euler"])
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_s, val_s, test_s, D = make_splits(args.data_root, seed=args.seed)

    train_ds = PhysioNet2012Dataset(train_s)
    val_ds = PhysioNet2012Dataset(val_s)
    test_ds = PhysioNet2012Dataset(test_s)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ODERNNClassifier(D, hidden=args.hidden, method=args.method).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        n = 0

        for batch in train_loader:
            t, x, m, rid, y = batch
            t = t.to(device)
            x = x.to(device)
            m = m.to(device)
            y = y.to(device).float().view(-1)

            opt.zero_grad(set_to_none=True)
            logits = model(t, x, m)
            loss = loss_fn(logits.view(-1), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total += float(loss.item()) * y.shape[0]
            n += y.shape[0]

        train_loss = total / max(1, n)
        val_auroc, val_auprc = eval_loader(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_auroc": val_auroc, "val_auprc": val_auprc})

        print(f"epoch {epoch:02d} | train_loss={train_loss:.4f} | val_auroc={val_auroc:.4f} | val_auprc={val_auprc:.4f}")

        score = val_auroc if not math.isnan(val_auroc) else -1.0
        if score > best_val:
            best_val = score
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_auroc, test_auprc = eval_loader(model, test_loader, device)
    print("[ODE-RNN] seed=", args.seed, "AUROC=", test_auroc, "AUPRC=", test_auprc)

    os.makedirs("results/physio_runs", exist_ok=True)
    out = {
        "model": "ODE-RNN",
        "seed": int(args.seed),
        "device": device,
        "D": int(D),
        "hidden": int(args.hidden),
        "method": args.method,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "history": history,
        "test_auroc": float(test_auroc) if not math.isnan(test_auroc) else None,
        "test_auprc": float(test_auprc),
    }
    path = f"results/physio_runs/physio_odernn_seed{args.seed}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("Wrote:", path)

if __name__ == "__main__":
    main()
