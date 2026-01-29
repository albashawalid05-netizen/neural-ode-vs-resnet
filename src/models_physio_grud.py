from __future__ import annotations

import torch
import torch.nn as nn


class GRUDCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.D = input_dim
        self.H = hidden_dim

        # Input decay per feature
        self.w_x = nn.Parameter(torch.zeros(input_dim))
        self.b_x = nn.Parameter(torch.zeros(input_dim))

        # Hidden decay (scalar per step)
        self.w_h = nn.Parameter(torch.zeros(1))
        self.b_h = nn.Parameter(torch.zeros(1))

        # GRU gates use [x_hat, mask]
        self.gru = nn.GRUCell(input_size=input_dim * 2, hidden_size=hidden_dim)

    def forward(
        self,
        x_t: torch.Tensor,     # (B, D)
        m_t: torch.Tensor,     # (B, D)
        d_x_t: torch.Tensor,   # (B, D)
        d_t: torch.Tensor,     # (B, 1)
        x_mean: torch.Tensor,  # (D,)
        x_prev: torch.Tensor,  # (B, D)
        h_prev: torch.Tensor,  # (B, H)
    ):
        # Decay inputs toward feature means
        gamma_x = torch.exp(-torch.relu(self.w_x * d_x_t + self.b_x))  # (B, D)
        x_decay = gamma_x * x_prev + (1.0 - gamma_x) * x_mean          # (B, D)
        x_hat = m_t * x_t + (1.0 - m_t) * x_decay                      # (B, D)

        # Decay hidden state
        gamma_h = torch.exp(-torch.relu(self.w_h * d_t + self.b_h))    # (B, 1)
        h_prev = gamma_h * h_prev                                      # (B, H)

        # GRU update with mask concatenated
        inp = torch.cat([x_hat, m_t], dim=-1)                          # (B, 2D)
        h_t = self.gru(inp, h_prev)

        # Update carried forward last-observed values
        x_prev = m_t * x_t + (1.0 - m_t) * x_prev
        return h_t, x_prev


class GRUDClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.cell = GRUDCell(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,        # (B, T, D)
        m: torch.Tensor,        # (B, T, D)
        d_x: torch.Tensor,      # (B, T, D)
        d_t: torch.Tensor,      # (B, T, 1)
        x_mean: torch.Tensor,   # (D,)
    ):
        B, T, _ = x.shape
        device = x.device
        x_mean = x_mean.to(device)

        h = torch.zeros(B, self.cell.H, device=device)
        x_prev = x_mean.unsqueeze(0).repeat(B, 1)  # init at mean

        for t in range(T):
            h, x_prev = self.cell(
                x_t=x[:, t],
                m_t=m[:, t],
                d_x_t=d_x[:, t],
                d_t=d_t[:, t],
                x_mean=x_mean,
                x_prev=x_prev,
                h_prev=h,
            )

        h = self.dropout(h)
        logits = self.head(h).squeeze(-1)
        return logits
