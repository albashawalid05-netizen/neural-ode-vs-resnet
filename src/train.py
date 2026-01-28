import os
import json
import argparse
import yaml
import torch
from tqdm import tqdm

from src.data import set_seed, make_loaders
from src.models import ResNetStep, NeuralODEModel


def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_device(cfg):
    dev = cfg.get("device", "auto")
    if dev == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(dev)


def masked_mse(pred, target, mask):
    """
    pred/target: [B, T, 2]
    mask: [B, T, 1] where 1 observed, 0 missing
    """
    m = mask
    diff2 = (pred - target) ** 2  # [B, T, 2]
    diff2 = diff2 * m  # broadcast on last dim
    denom = torch.clamp(m.sum() * pred.shape[-1], min=1.0)
    return diff2.sum() / denom


@torch.no_grad()
def evaluate(model, val_loader, t, device):
    model.eval()
    losses = []
    for x, m in val_loader:
        x = x.to(device)
        m = m.to(device)
        x0 = x[:, 0, :]  # [B,2]
        T = x.shape[1]
        if isinstance(model, NeuralODEModel):
            pred = model(x0, t.to(device))  # [B,T,2]
        else:
            pred = torch.cat([x0.unsqueeze(1), model(x0, T - 1)], dim=1)  # include t0
        loss = masked_mse(pred, x, m)
        losses.append(loss.item())
    return float(sum(losses) / max(len(losses), 1))


def train_one(model, train_loader, val_loader, t, device, epochs=15, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_val = 1e9
    history = {"train": [], "val": []}

    for ep in range(1, epochs + 1):
        model.train()
        train_losses = []
        for x, m in tqdm(train_loader, desc=f"epoch {ep}/{epochs}", leave=False):
            x = x.to(device)
            m = m.to(device)
            x0 = x[:, 0, :]
            T = x.shape[1]

            if isinstance(model, NeuralODEModel):
                pred = model(x0, t.to(device))  # [B,T,2]
            else:
                pred = torch.cat([x0.unsqueeze(1), model(x0, T - 1)], dim=1)

            loss = masked_mse(pred, x, m)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

        train_loss = float(sum(train_losses) / max(len(train_losses), 1))
        val_loss = evaluate(model, val_loader, t, device)

        history["train"].append(train_loss)
        history["val"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss

    return best_val, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/base.yaml")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    set_seed(cfg["seed"])
    device = get_device(cfg)

    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    train_loader, val_loader, t = make_loaders(cfg)

    # Baseline 1: ResNet step model
    resnet = ResNetStep(hidden_dim=cfg["model"]["hidden_dim"]).to(device)
    res_best, res_hist = train_one(
        resnet, train_loader, val_loader, t, device,
        epochs=cfg["train"]["epochs"], lr=cfg["train"]["lr"]
    )

    # Baseline 2: Neural ODE
    ode = NeuralODEModel(
        hidden_dim=cfg["model"]["hidden_dim"],
        solver=cfg["ode"]["solver"],
        rtol=cfg["ode"]["rtol"],
        atol=cfg["ode"]["atol"],
    ).to(device)
    ode_best, ode_hist = train_one(
        ode, train_loader, val_loader, t, device,
        epochs=cfg["train"]["epochs"], lr=cfg["train"]["lr"]
    )

    out = {
        "config_path": args.config,
        "config": cfg,
        "device": str(device),
        "resnet_val_mse": res_best,
        "neuralode_val_mse": ode_best,
        "history": {"resnet": res_hist, "neuralode": ode_hist},
    }

    cfg_name = os.path.splitext(os.path.basename(args.config))[0]
    out_path = f"results/tables/{cfg_name}_metrics.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"Saved results to {out_path}")
    print("ResNet val MSE:", res_best)
    print("Neural ODE val MSE:", ode_best)


if __name__ == "__main__":
    main()
