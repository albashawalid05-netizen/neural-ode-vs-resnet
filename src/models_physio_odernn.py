import torch
import torch.nn as nn
from torchdiffeq import odeint

class ODEFunc(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
        )
        self._dt = None

    def set_dt(self, dt: torch.Tensor):
        if dt.dim() == 1:
            dt = dt.unsqueeze(1)
        self._dt = dt

    def forward(self, t, h):
        return self._dt * self.net(h)

class ODERNNClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128, method: str = "rk4"):
        super().__init__()
        self.hidden = hidden
        self.method = method
        self.gru = nn.GRUCell(input_dim * 2, hidden)  # concat(x, mask)
        self.odefunc = ODEFunc(hidden)
        self.head = nn.Linear(hidden, 1)
        self._t = torch.tensor([0.0, 1.0])

    def forward(self, t, x, m):
        B, T, D = x.shape
        device = x.device
        h = torch.zeros(B, self.hidden, device=device)
        m_f = m.float()
        t_f = t.float()

        for i in range(T):
            inp = torch.cat([x[:, i], m_f[:, i]], dim=-1)
            h = self.gru(inp, h)

            if i < T - 1:
                dt = (t_f[:, i + 1] - t_f[:, i]).clamp(min=0.0)
                self.odefunc.set_dt(dt)
                tt = self._t.to(device)
                h = odeint(self.odefunc, h, tt, method=self.method)[-1]

        logits = self.head(h).squeeze(-1)
        return logits
