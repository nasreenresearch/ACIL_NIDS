# strategies/srr.py
from __future__ import annotations
from typing import List, Tuple, Any, Optional
import math
import random

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, Dataset


def _to_tensor(x) -> torch.Tensor:
    if torch.is_tensor(x):
        return x.detach().cpu()
    return torch.as_tensor(x).detach().cpu()


def _ensure_min1d(x: torch.Tensor) -> torch.Tensor:
    return x.unsqueeze(0) if x.dim() == 0 else x


def _stack_pairs(pairs: List[Tuple[torch.Tensor, int]]) -> TensorDataset | list:
    if not pairs:
        return []
    xs = [_ensure_min1d(_to_tensor(x)) for x, _ in pairs]
    try:
        X = torch.stack(xs)
    except Exception:
        xs = [t if t.dim() >= 1 else t.unsqueeze(0) for t in xs]
        max_len = max(t.shape[-1] if t.dim() > 0 else 1 for t in xs)
        xs = [t if (t.dim() > 0 and t.shape[-1] == max_len)
              else F.pad(t, (0, max_len - (t.shape[-1] if t.dim() > 0 else 1)))
              for t in xs]
        X = torch.stack(xs)
    y = torch.tensor([int(t) for _, t in pairs], dtype=torch.long)
    return TensorDataset(X, y)


class SRRBuffer:
    """
    Score-Reservoir Replay (SRR): class-agnostic priority reservoir.
    Keeps the top-k items by key k = u^(1/score), where u~Uniform(0,1).

    Required by Avalanche ReplayPlugin:
      - .max_size (int)
      - .update(...) or .update_from_dataset(...)
      - .get_memory() -> Dataset or list[(x,y)]
    """
    def __init__(self, capacity: int, scoring: str = "loss", temperature: float = 1.0):
        self.capacity = int(capacity)
        self.max_size = self.capacity
        self.scoring = scoring  # "loss" | "entropy" | "margin"
        self.temperature = float(temperature)
        # store as (key, x, y)
        self._buf: List[Tuple[float, torch.Tensor, int]] = []

    def __len__(self): return len(self._buf)

    # ----- public API used by ReplayPlugin -----
    def update(self, strategy=None, **kwargs):
        """
        Called each iteration by Avalanche (with 'strategy'), or
        with a dataset via kwargs['dataset'].
        """
        dataset = kwargs.get("dataset")
        if dataset is not None:
            # fallback path: stream in a dataset (no scores)
            # default to uniform reservoir when scores not available
            for x, y in dataset:
                self._maybe_insert_uniform(x, y)
            return

        if strategy is None:
            return

        # Use current minibatch and model outputs to score items
        x_mb, y_mb = strategy.mb_x, strategy.mb_y
        logits = strategy.mb_output  # computed by forward
        with torch.no_grad():
            # per-sample scores
            score = self._compute_scores(logits.detach(), y_mb.detach())
            # move to CPU for storage
            x_mb_cpu = x_mb.detach().cpu()
            y_mb_cpu = y_mb.detach().cpu()
            for i in range(y_mb_cpu.shape[0]):
                s = float(score[i].clamp(min=1e-8))  # ensure >0
                u = random.random()
                key = u ** (1.0 / s)
                self._maybe_insert_keyed(key, x_mb_cpu[i], int(y_mb_cpu[i].item()))

    def update_from_dataset(self, dataset, **kwargs):
        # If someone explicitly calls this: uniform reservoir fallback
        for x, y in dataset:
            self._maybe_insert_uniform(x, y)

    def get_memory(self):
        pairs = [(x, y) for _, x, y in sorted(self._buf, key=lambda t: t[0], reverse=True)]
        return _stack_pairs(pairs)

    # ----- internal scoring & reservoir logic -----
    def _compute_scores(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # logits: (B, C)
        if self.scoring == "loss":
            # per-sample cross-entropy (temperature optional)
            if self.temperature != 1.0:
                logits = logits / self.temperature
            return F.cross_entropy(logits, y, reduction="none")
        elif self.scoring == "entropy":
            p = F.softmax(logits / self.temperature, dim=1)
            return -(p * (p.clamp_min(1e-12).log())).sum(dim=1)  # higher = more uncertain
        elif self.scoring == "margin":
            p = F.softmax(logits / self.temperature, dim=1)
            top2 = torch.topk(p, k=2, dim=1).values  # (B, 2)
            margin = (top2[:, 0] - top2[:, 1]).clamp_min(1e-12)
            return 1.0 / margin  # smaller margin -> higher score
        else:
            # default: uniform
            return torch.ones(logits.shape[0], device=logits.device)

    def _maybe_insert_keyed(self, key: float, x, y: int):
        x = _ensure_min1d(_to_tensor(x)); y = int(y)
        if len(self._buf) < self.capacity:
            self._buf.append((key, x, y))
            return
        # replace if key is larger than the worst (min key)
        min_idx, min_key = min(enumerate(self._buf), key=lambda t: t[1][0])
        if key > min_key[0]:
            self._buf[min_idx] = (key, x, y)

    def _maybe_insert_uniform(self, x, y: int):
        """Fallback: classic uniform reservoir (no scores)."""
        x = _ensure_min1d(_to_tensor(x)); y = int(y)
        n = len(self._buf)
        if n < self.capacity:
            self._buf.append((1.0, x, y))
        else:
            j = random.randint(0, n)
            if j < self.capacity:
                self._buf[j] = (1.0, x, y)
    def get_memory(self):
        pairs = [(x, y) for _, x, y in sorted(self._buf, key=lambda t: t[0], reverse=True)]
        return _stack_pairs(pairs)   # -> TensorDataset or []

    @property
    def buffer(self):
        """
        Avalanche expects a length-checkable object here.
        Returning the same thing as get_memory() is enough:
        - TensorDataset: len() works
        - [] when empty: len() works
        """
        return self.get_memory()