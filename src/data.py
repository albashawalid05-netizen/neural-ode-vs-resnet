import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def simulate_damped_oscillator(x0, v0, dt, seq_len, gamma=0.15, omega=1.0):
    """
    x'' + 2*gamma*x' + omega^2*x = 0
    State: [x, v]
    """
    x, v = x0, v0
    states = []
    for _ in range(seq_len):
        a = -2.0 * gamma * v - (omega ** 2) * x
        x = x + dt * v
        v = v + dt * a
        states.append([x, v])
    return np.array(states, dtype=np.float32)


def make_mask(seq_len: int, missing_rate: float):
    """
    Returns mask shape [seq_len, 1], where 1 means observed, 0 means missing.
    Keep the first point observed to anchor the sequence.
    """
    mask = np.ones((seq_len, 1), dtype=np.float32)
    if missing_rate <= 0:
        return mask
    drop = np.random.rand(seq_len) < missing_rate
    drop[0] = False
    mask[drop, 0] = 0.0
    return mask


def make_dataset(n_sequences=3000, seq_len=50, dt=0.1, noise_std=0.05, missing_rate=0.3):
    """
    Returns:
      x: [N, T, 2] noisy observations (with missingness still present via mask)
      m: [N, T, 1] mask
      t: [T] time vector
    """
    t = np.arange(seq_len, dtype=np.float32) * dt
    X = np.zeros((n_sequences, seq_len, 2), dtype=np.float32)
    M = np.zeros((n_sequences, seq_len, 1), dtype=np.float32)

    for i in range(n_sequences):
        x0 = np.random.uniform(-2.0, 2.0)
        v0 = np.random.uniform(-2.0, 2.0)
        traj = simulate_damped_oscillator(x0, v0, dt, seq_len)

        noise = np.random.randn(seq_len, 2).astype(np.float32) * noise_std
        obs = traj + noise

        mask = make_mask(seq_len, missing_rate)

        X[i] = obs
        M[i] = mask

    return X, M, t


class ODEDataset(Dataset):
    def __init__(self, X, M):
        self.X = torch.from_numpy(X)  # [N, T, 2]
        self.M = torch.from_numpy(M)  # [N, T, 1]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        # Predict next-step for simplicity: input is full sequence, target is same sequence shifted
        x = self.X[idx]  # [T, 2]
        m = self.M[idx]  # [T, 1]
        return x, m


def make_loaders(cfg):
    X, M, t = make_dataset(
        n_sequences=cfg["data"]["n_sequences"],
        seq_len=cfg["data"]["seq_len"],
        dt=cfg["data"]["dt"],
        noise_std=cfg["data"]["noise_std"],
        missing_rate=cfg["data"]["missing_rate"],
    )

    N = X.shape[0]
    n_train = int(N * cfg["data"]["train_split"])

    train_ds = ODEDataset(X[:n_train], M[:n_train])
    val_ds = ODEDataset(X[n_train:], M[n_train:])

    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False)

    t_tensor = torch.from_numpy(t)  # [T]
    return train_loader, val_loader, t_tensor
