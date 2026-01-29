import torch
import torch.nn as nn
from torchdiffeq import odeint


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, depth=3):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth - 1):
            layers += [nn.Linear(d, hidden_dim), nn.ReLU()]
            d = hidden_dim
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ResNetStep(nn.Module):
    """
    Discrete-time baseline: predicts next state using residual MLP:
      x_{t+1} = x_t + f(x_t)
    """
    def __init__(self, state_dim=2, hidden_dim=64):
        super().__init__()
        self.f = MLP(state_dim, hidden_dim, state_dim, depth=3)

    def forward(self, x0, steps):
        # x0: [B, 2], returns [B, steps, 2]
        xs = []
        x = x0
        for _ in range(steps):
            x = x + self.f(x)
            xs.append(x)
        return torch.stack(xs, dim=1)


class ODEFunc(nn.Module):
    """
    Continuous dynamics: dx/dt = f(x, t) but we ignore t in f for simplicity.
    """
    def __init__(self, state_dim=2, hidden_dim=64):
        super().__init__()
        self.f = MLP(state_dim, hidden_dim, state_dim, depth=3)

    def forward(self, t, x):
        return self.f(x)


class NeuralODEModel(nn.Module):
    def __init__(self, state_dim=2, hidden_dim=64, solver="rk4", rtol=1e-4, atol=1e-4):
        super().__init__()
        self.func = ODEFunc(state_dim, hidden_dim)
        self.solver = solver
        self.rtol = rtol
        self.atol = atol

    def forward(self, x0, t):
        """
        x0: [B, 2]
        t:  [T] increasing times
        returns: [B, T, 2]
        """
        # odeint returns [T, B, 2]
        traj = odeint(self.func, x0, t, method=self.solver, rtol=self.rtol, atol=self.atol)
        return traj.permute(1, 0, 2)
import torch
import torch.nn as nn

class GRUClassifier(nn.Module):
    def __init__(self, d_in: int, hidden: int = 128, layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=2 * d_in + 1,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, times, x, mask, lengths):
        dt = torch.zeros_like(times)
        dt[:, 1:] = times[:, 1:] - times[:, :-1]
        dt = dt.unsqueeze(-1)

        inp = torch.cat([x, mask, dt], dim=-1)

        packed = nn.utils.rnn.pack_padded_sequence(
            inp, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.rnn(packed)
        h_last = h[-1]
        return self.head(h_last)
class LastObsMLP(nn.Module):
    """
    Baseline بسيط: ناخذ آخر قياس لكل feature (مع mask) -> MLP للتصنيف.
    """
    def __init__(self, d_in: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * d_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, times, x, mask, lengths):
        # x/mask: [B,T,D]
        # استخدم آخر timestep فعلي حسب lengths
        B, T, D = x.shape
        idx = (lengths - 1).clamp_min(0)  # [B]
        last_x = x[torch.arange(B), idx]       # [B,D]
        last_m = mask[torch.arange(B), idx]    # [B,D]
        feat = torch.cat([last_x, last_m], dim=-1)  # [B,2D]
        return self.net(feat)
