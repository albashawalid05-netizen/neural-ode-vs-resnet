from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from src.models_physio_grud import GRUDClassifier
from src.physionet2012 import make_splits, PhysioNet2012Dataset, collate_fn


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


@torch.no_grad()
def compute_train_feature_mean(train_loader, device):
    # mean over all observed entries (mask==1) across (B,T)
    sum_x = None
    sum_m = None
    for times, x, mask, lengths, y in train_loader:
        x = x.to(device)          # (B,T,D)
        mask = mask.to(device)    # (B,T,D)
        xm = (x * mask).sum(dim=(0, 1))   # (D,)
        mm = mask.sum(dim=(0, 1))         # (D,)

        if sum_x is None:
            sum_x = xm
            sum_m = mm
        else:
            sum_x = sum_x + xm
            sum_m = sum_m + mm

    mean = sum_x / (sum_m.clamp_min(1.0))
    mean = torch.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
    return mean  # (D,)


def build_deltas(times, mask, lengths):
    """
    times:  (B,T) in hours
    mask:   (B,T,D) 1 if observed else 0
    lengths:(B,)
    returns:
      d_t: (B,T,1) time gap since previous step
      d_x: (B,T,D) time since last observation per feature
      mask2: (B,T,D) mask with padding forced to 0
    """
    B, T, D = mask.shape
    device = times.device

    # valid timestep mask (B,T)
    t_idx = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
    valid = t_idx < lengths.unsqueeze(1)

    # force padding mask to 0
    mask2 = mask * valid.unsqueeze(-1).float()

    # d_t
    d_t = torch.zeros(B, T, 1, device=device)
    if T > 1:
        dt = times[:, 1:] - times[:, :-1]              # (B,T-1)
        dt = dt * valid[:, 1:].float()                 # padding -> 0
        d_t[:, 1:, 0] = dt

    # d_x
    d_x = torch.zeros(B, T, D, device=device)
    last_time = torch.zeros(B, D, device=device)       # init at 0 hours
    for t in range(T):
        v_t = valid[:, t]                              # (B,)
        if not torch.any(v_t):
            continue
        cur = times[:, t].unsqueeze(-1)                # (B,1)

        # only for valid rows
        cur_v = cur[v_t]
        last_v = last_time[v_t]
        m_v = mask2[:, t, :][v_t]                      # (Bv,D)

        d_x[v_t, t, :] = cur_v.expand(-1, D) - last_v
        # update last_time where observed
        last_time[v_t] = torch.where(m_v.bool(), cur_v.expand(-1, D), last_v)

    return d_t, d_x, mask2


@torch.no_grad()
def evaluate(model, loader, device, x_mean):
    model.eval()
    all_probs = []
    all_y = []
    all_obs = []

    for times, x, mask, lengths, y in loader:
        times = times.to(device)      # (B,T)
        x = x.to(device)              # (B,T,D)
        mask = mask.to(device)        # (B,T,D)
        lengths = lengths.to(device)  # (B,)
        y = y.to(device)              # (B,)

        d_t, d_x, mask2 = build_deltas(times, mask, lengths)
        logits = model(x, mask2, d_x, d_t, x_mean=x_mean)
        probs = torch.sigmoid(logits)

        all_probs.append(probs.detach().cpu().numpy())
        all_y.append(y.detach().cpu().numpy())

        # obs_count proxy = total observed entries
        all_obs.append(mask2.detach().cpu().numpy().sum(axis=(1, 2)))

    y_true = np.concatenate(all_y).astype(int)
    y_prob = np.concatenate(all_probs).astype(float)
    obs_count = np.concatenate(all_obs).astype(float)

    auroc = float(roc_auc_score(y_true, y_prob))
    auprc = float(average_precision_score(y_true, y_prob))
    return auroc, auprc, y_true, y_prob, obs_count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="data/physionet2012_raw")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # same split pipeline as your GRU baseline
    train_s, val_s, test_s, var_to_idx = make_splits(args.data_root, seed=args.seed)
    train_ds = PhysioNet2012Dataset(train_s)
    val_ds = PhysioNet2012Dataset(val_s)
    test_ds = PhysioNet2012Dataset(test_s)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # infer D from first batch
    times0, x0, mask0, lengths0, y0 = next(iter(train_loader))
    D = x0.shape[-1]

    # compute feature mean in the same space as x returned by dataset
    x_mean = compute_train_feature_mean(train_loader, device=device)  # (D,)

    model = GRUDClassifier(input_dim=D, hidden_dim=args.hidden, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit = nn.BCEWithLogitsLoss()

    best_val = -1.0
    best_state = None
    patience = 5
    bad = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []

        for times, x, mask, lengths, y in train_loader:
            times = times.to(device)
            x = x.to(device)
            mask = mask.to(device)
            lengths = lengths.to(device)
            y = y.float().to(device)

            d_t, d_x, mask2 = build_deltas(times, mask, lengths)

            opt.zero_grad(set_to_none=True)
            logits = model(x, mask2, d_x, d_t, x_mean=x_mean)
            loss = crit(logits, y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            train_losses.append(loss.item())

        val_auroc, val_auprc, _, _, _ = evaluate(model, val_loader, device, x_mean=x_mean)
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        print(f"epoch {epoch:02d} | train_loss={train_loss:.4f} | val_auroc={val_auroc:.4f} | val_auprc={val_auprc:.4f}")

        if val_auroc > best_val:
            best_val = val_auroc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_auroc, test_auprc, y_true, y_prob, obs_count = evaluate(model, test_loader, device, x_mean=x_mean)

    # match your GRU script folder pattern (it already creates results/physio_runs)
    os.makedirs("results/physio_runs", exist_ok=True)
    out_json = f"results/physio_runs/physio_grud_seed{args.seed}.json"
    out = {
        "model": "GRU-D",
        "seed": args.seed,
        "test_auroc": test_auroc,
        "test_auprc": test_auprc,
    }
    Path(out_json).write_text(json.dumps(out, indent=2))
    print(f"[GRU-D] seed={args.seed} AUROC={test_auroc:.4f} AUPRC={test_auprc:.4f}")
    print(f"Wrote: {out_json}")

    # save preds for the new comparison plot
    pred_dir = Path("results/preds/physio/grud")
    pred_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(pred_dir / f"seed{args.seed}.npz", y_true=y_true, y_prob=y_prob, obs_count=obs_count)


if __name__ == "__main__":
    main()
