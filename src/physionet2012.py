# src/physionet2012.py
from __future__ import annotations

import os
import csv
import glob
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import torch
from torch.utils.data import Dataset


# A compact variable set (common vitals/labs). You can expand later.
DEFAULT_VARS = [
    "Heart Rate", "Respiratory Rate", "Temperature", "Weight",
    "Systolic BP", "Diastolic BP", "Mean BP",
    "SpO2", "GCS", "pH", "PaO2", "PaCO2",
    "Glucose", "Creatinine", "BUN", "Sodium", "Potassium",
    "HCO3", "Chloride", "WBC", "Hgb", "Platelets", "Lactate",
]


def _parse_time_to_hours(t: str) -> float:
    # format: HH:MM
    hh, mm = t.split(":")
    return float(int(hh)) + float(int(mm)) / 60.0


def load_outcomes(outcomes_path: str) -> Dict[int, int]:
    # RecordID -> In-hospital_death (0/1)
    out: Dict[int, int] = {}
    with open(outcomes_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = int(row["RecordID"])
            y = int(row["In-hospital_death"])
            out[rid] = y
    return out


def list_patient_files(set_a_dir: str) -> List[str]:
    # files named like 132539.txt
    files = sorted(glob.glob(os.path.join(set_a_dir, "*.txt")))
    return files


def parse_patient_file(path: str, var_to_idx: Dict[str, int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      times: [N] float (hours)
      values: [N, D] float (0 where missing at that timestamp)
      mask: [N, D] float (1 where observed)
    We treat each row as an observation time; multiple variables can appear at same time,
    so we aggregate by time (last value wins).
    """
    rows: List[Tuple[float, str, float]] = []
    with open(path, "r") as f:
        _ = f.readline()  # header: Time,Parameter,Value
        for line in f:
            line = line.strip()
            if not line:
                continue
            t_s, param, val_s = line.split(",")
            if param not in var_to_idx:
                continue
            try:
                val = float(val_s)
            except ValueError:
                continue
            t = _parse_time_to_hours(t_s)
            rows.append((t, param, val))

    if len(rows) == 0:
        # edge case: no vars in our set
        times = torch.zeros(1, dtype=torch.float32)
        D = len(var_to_idx)
        values = torch.zeros((1, D), dtype=torch.float32)
        mask = torch.zeros((1, D), dtype=torch.float32)
        return times, values, mask

    # group by time
    rows.sort(key=lambda x: x[0])
    uniq_times: List[float] = []
    time_to_row: Dict[float, int] = {}

    D = len(var_to_idx)
    values_list: List[torch.Tensor] = []
    mask_list: List[torch.Tensor] = []

    for t, param, val in rows:
        if t not in time_to_row:
            time_to_row[t] = len(uniq_times)
            uniq_times.append(t)
            values_list.append(torch.zeros(D, dtype=torch.float32))
            mask_list.append(torch.zeros(D, dtype=torch.float32))
        j = time_to_row[t]
        k = var_to_idx[param]
        values_list[j][k] = float(val)
        mask_list[j][k] = 1.0

    times = torch.tensor(uniq_times, dtype=torch.float32)
    values = torch.stack(values_list, dim=0)  # [N, D]
    mask = torch.stack(mask_list, dim=0)      # [N, D]
    return times, values, mask


def compute_normalization(stats_records: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], eps: float = 1e-6):
    # stats_records: list of (times, values, mask)
    # compute mean/std per feature using observed entries only
    D = stats_records[0][1].shape[1]
    sum_x = torch.zeros(D)
    sum_x2 = torch.zeros(D)
    count = torch.zeros(D)

    for _, x, m in stats_records:
        # x: [N, D], m: [N, D]
        sum_x += (x * m).sum(dim=0)
        sum_x2 += ((x * x) * m).sum(dim=0)
        count += m.sum(dim=0)

    mean = sum_x / (count.clamp_min(1.0))
    var = (sum_x2 / (count.clamp_min(1.0))) - mean * mean
    std = torch.sqrt(var.clamp_min(eps))
    return mean, std


def normalize(values: torch.Tensor, mask: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
    # values: [N, D], mask: [N, D], mean/std: [D]
    x = values.clone()
    x = (x - mean) / std
    x = x * mask  # keep missing entries at 0
    return x



@dataclass
class PhysioSample:
    times: torch.Tensor      # [N]
    x: torch.Tensor          # [N, D]
    mask: torch.Tensor       # [N, D]
    y: torch.Tensor          # [1]


class PhysioNet2012Dataset(Dataset):
    def __init__(self, samples: List[PhysioSample]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        return s.times, s.x, s.mask, s.y


def collate_fn(batch):
    # batch: list of (times [Ni], x [Ni,D], mask [Ni,D], y [1])
    times_list, x_list, m_list, y_list = zip(*batch)
    B = len(batch)
    D = x_list[0].shape[1]
    lengths = torch.tensor([t.shape[0] for t in times_list], dtype=torch.long)
    T = int(lengths.max().item())

    times_pad = torch.zeros((B, T), dtype=torch.float32)
    x_pad = torch.zeros((B, T, D), dtype=torch.float32)
    m_pad = torch.zeros((B, T, D), dtype=torch.float32)

    for i in range(B):
        n = times_list[i].shape[0]
        times_pad[i, :n] = times_list[i]
        x_pad[i, :n] = x_list[i]
        m_pad[i, :n] = m_list[i]

    y = torch.stack(y_list, dim=0).float()  # [B,1]
    return times_pad, x_pad, m_pad, lengths, y


def make_splits(
    root: str,
    var_list: List[str] = DEFAULT_VARS,
    seed: int = 0,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
):
    set_a_dir = os.path.join(root, "set-a")
    outcomes_path = os.path.join(root, "Outcomes-a.txt")

    outcomes = load_outcomes(outcomes_path)
    files = list_patient_files(set_a_dir)

    # build (rid, path, y)
    triples = []
    for p in files:
        rid = int(os.path.splitext(os.path.basename(p))[0])
        if rid in outcomes:
            triples.append((rid, p, outcomes[rid]))

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(triples), generator=g).tolist()
    triples = [triples[i] for i in perm]

    n = len(triples)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    train_t = triples[:n_train]
    val_t = triples[n_train:n_train + n_val]
    test_t = triples[n_train + n_val:]

    var_to_idx = {v: i for i, v in enumerate(var_list)}

    # parse train first for stats
    train_parsed = [parse_patient_file(p, var_to_idx) for _, p, _ in train_t]
    mean, std = compute_normalization(train_parsed)

    def build_samples(trip_list):
        samples: List[PhysioSample] = []
        for (rid, p, y) in trip_list:
            t, x, m = parse_patient_file(p, var_to_idx)
            x = normalize(x, m, mean, std)
            samples.append(PhysioSample(t, x, m, torch.tensor([y], dtype=torch.float32)))
        return samples

    train_s = build_samples(train_t)
    val_s = build_samples(val_t)
    test_s = build_samples(test_t)

    return train_s, val_s, test_s, len(var_list)
