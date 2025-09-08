# benchmarks/make_benchmark.py
import math
import torch
from torch.utils.data import Dataset
from avalanche.benchmarks import class_incremental_benchmark
from avalanche.benchmarks.utils import as_classification_dataset


class _TensorXY(Dataset):
    """Minimal dataset that exposes `.targets` so Avalanche can read labels."""
    def __init__(self, X, y):
        self.X = torch.as_tensor(X)
        self.y = torch.as_tensor(y).long()
        self.targets = self.y  # Avalanche will look for this

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def _from_xy(X, y):
    """Create a ClassificationDataset from (X, y) without non-exported helpers."""
    base = _TensorXY(X, y)  # has `.targets`
    return as_classification_dataset(base)  # no kwargs; reads .targets


def _ensure_classification_dataset(ds):
    """
    Ensure ds is a ClassificationDataset exposing `.targets`.
    Accepts:
      - tuple (X, y) tensors/arrays
      - dataset with `.targets`
      - dataset with `.X` and `.y`
      - generic indexable dataset (slow fallback)
    """
    # (X, y)
    if isinstance(ds, tuple) and len(ds) == 2:
        X, y = ds
        return _from_xy(X, y)

    # Already has targets? Wrap it.
    if hasattr(ds, "targets"):
        return as_classification_dataset(ds)

    # Common custom attrs
    if hasattr(ds, "X") and hasattr(ds, "y"):
        return _from_xy(ds.X, ds.y)

    # Slow fallback: infer y by indexing
    ys = []
    for i in range(len(ds)):
        item = ds[i]
        if isinstance(item, tuple) and len(item) >= 2:
            ys.append(item[1])
        elif isinstance(item, dict) and "y" in item:
            ys.append(item["y"])
        else:
            raise TypeError("Cannot infer targets from dataset items")
    y = torch.as_tensor(ys).long()
    X_dummy = torch.arange(len(y)).float().unsqueeze(1)  # (N,1) placeholder
    return _from_xy(X_dummy, y)


def _normalize_num_experiences(n_classes, num_classes_per_exp):
    """Compute num experiences from either an int or a list like [3,3,4]."""
    if isinstance(num_classes_per_exp, int):
        return math.ceil(n_classes / max(1, num_classes_per_exp))
    if isinstance(num_classes_per_exp, (list, tuple)):
        return len(num_classes_per_exp)
    raise TypeError(
        f"num_classes_per_exp must be int or list/tuple, got {type(num_classes_per_exp)}"
    )


def make_cil(train_ds, test_ds, classes, num_classes_per_exp=2, seed=0):
    """
    If `classes` is provided as a list of ints and matches dataset targets, we use it.
    Otherwise, we compute a FIXED class order = descending frequency from TRAIN targets.
    """
    train_clf = _ensure_classification_dataset(train_ds)
    test_clf  = _ensure_classification_dataset(test_ds)

    # True target IDs in train
    t = train_clf.targets
    # Handle both tensor and list
    t = torch.as_tensor(t).long()
    uniq, counts = torch.unique(t, return_counts=True)
    uniq = uniq.tolist()
    counts = counts.tolist()

    # Desired: descending by frequency
    order_by_freq = [c for _, c in sorted(zip(counts, uniq), key=lambda z: z[0], reverse=True)]

    # If user passed valid int classes that exist, respect them; else use frequency order
    use_order = None
    if classes is not None and all(isinstance(c, int) for c in classes):
        missing = [c for c in classes if c not in uniq]
        if not missing:
            use_order = classes  # user-specified valid integers
    if use_order is None:
        use_order = order_by_freq  # fixed order by descending frequency

    n_exp = _normalize_num_experiences(len(uniq), num_classes_per_exp)

    # Build args for Avalanche.
    # When class_order is fixed, DO NOT pass seed.
    kwargs = {
        "datasets_dict": {"train": train_clf, "test": test_clf},
        "num_experiences": n_exp,
        "class_order": use_order,
    }

    scenario = class_incremental_benchmark(**kwargs)
    return scenario
