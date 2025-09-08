# strategies/cbrs.py
from collections import defaultdict, deque
import random
from typing import List, Tuple, Any

import torch
from torch.utils.data import Dataset, TensorDataset


# ---- helpers ----------------------------------------------------------------
def _to_tensor(x) -> torch.Tensor:
    """Coerce features to a detached CPU tensor."""
    if torch.is_tensor(x):
        return x.detach().cpu()
    return torch.as_tensor(x)  # dtype inferred; fine for tabular


def _ensure_min1d(x: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is at least 1-D so stacking works."""
    return x.unsqueeze(0) if x.dim() == 0 else x


class _ListDataset(Dataset):
    """Fallback dataset view around a list of (x, y) pairs."""
    def __init__(self, pairs: List[Tuple[torch.Tensor, int]]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


# ---- CBRS buffer -------------------------------------------------------------
class CBRSBuffer:
    """
    Class-Balanced Replacement Strategy buffer compatible with Avalanche ReplayPlugin.

    - Ensures all X are torch.Tensors (CPU, detached)
    - Exposes .max_size == capacity (asserted by ReplayPlugin)
    - Provides update/update_from_dataset/get_memory APIs
    - Returns a TensorDataset from get_memory() so DataLoader collation is safe
    """
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.max_size = self.capacity  # Avalanche expects this
        self.buf: List[Tuple[torch.Tensor, int]] = []
        self.per_class = defaultdict(deque)  # class_id -> deque of buffer indices
        self.count = defaultdict(int)        # class_id -> count

    def __len__(self) -> int:
        return len(self.buf)

    # ---------- APIs used by ReplayPlugin ----------
    def update_from_dataset(self, dataset, **kwargs):
        """Ingest items from a dataset that yields (x, y)."""
        for x, y in dataset:
            y = int(y.item()) if torch.is_tensor(y) else int(y)
            self.add(x, y)

    def update(self, strategy=None, **kwargs):
        """
        Fallback updater. If plugin calls update(strategy),
        try current minibatch; or a provided 'dataset' kwarg.
        """
        ds = kwargs.get("dataset")
        if ds is not None:
            self.update_from_dataset(ds, **kwargs)
            return
        if strategy is not None and hasattr(strategy, "mb_x") and hasattr(strategy, "mb_y"):
            x_mb, y_mb = strategy.mb_x, strategy.mb_y
            bsz = y_mb.shape[0] if torch.is_tensor(y_mb) else len(y_mb)
            for i in range(bsz):
                x_i = x_mb[i]
                y_i = int(y_mb[i].item()) if torch.is_tensor(y_mb) else int(y_mb[i])
                self.add(x_i, y_i)

    def get_memory(self):
        """
        Return a TensorDataset so collation is always safe.
        If empty, return an empty list (ReplayPlugin handles it).
        """
        if not self.buf:
            return []
        xs = [_ensure_min1d(_to_tensor(x)) for x, _ in self.buf]
        # Try to stack directly; if ragged, pad last dim.
        try:
            X = torch.stack(xs)
        except Exception:
            # Make all at least 1D and pad to max length along the last dim
            xs = [t if t.dim() >= 1 else t.unsqueeze(0) for t in xs]
            max_len = max(t.shape[-1] if t.dim() > 0 else 1 for t in xs)
            xs = [
                t if (t.dim() > 0 and t.shape[-1] == max_len)
                else torch.nn.functional.pad(t, (0, max_len - (t.shape[-1] if t.dim() > 0 else 1)))
                for t in xs
            ]
            X = torch.stack(xs)
        y = torch.tensor([int(t) for _, t in self.buf], dtype=torch.long)
        return TensorDataset(X, y)

    @property
    def buffer(self) -> Dataset:
        """Dataset view of the memory (uses the same TensorDataset as get_memory)."""
        mem = self.get_memory()
        return mem if isinstance(mem, Dataset) else _ListDataset(self.buf)

    # ---------- internal logic ----------
    def add(self, x, y: int):
        x = _ensure_min1d(_to_tensor(x))
        y = int(y)

        if len(self.buf) < self.capacity:
            self.buf.append((x, y))
            self.per_class[y].append(len(self.buf) - 1)
            self.count[y] += 1
            return

        # choose class most over-represented vs. average
        avg = self.capacity / max(1, len(self.count))
        over = max(self.count.keys(), key=lambda c: (self.count[c] - avg, -c))

        # pick index to evict
        if self.per_class[over]:
            idx = self.per_class[over].popleft()
        else:
            idx = random.randrange(self.capacity)

        oldx, oldc = self.buf[idx]
        # tidy: remove idx from old class deque if present
        try:
            self.per_class[oldc].remove(idx)
        except ValueError:
            pass
        self.count[oldc] -= 1

        # place the new sample
        self.buf[idx] = (x, y)
        self.count[y] += 1
        self.per_class[y].append(idx)

        if self.count[oldc] <= 0:
            self.count.pop(oldc, None)
