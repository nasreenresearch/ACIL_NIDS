# strategies/papa.py
from __future__ import annotations
from typing import Callable, List, Tuple, Any, Optional
import random
import torch
from torch.utils.data import Dataset
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin


class _PairsDataset(Dataset):
    """Tiny dataset wrapping a list of (x,y) pairs."""
    def __init__(self, pairs: List[Tuple[Any, int]]):
        self.pairs = pairs
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx): return self.pairs[idx]


class PAPAPlugin(SupervisedPlugin):
    """
    PAPA-style plugin:
      - selector(model, dataset, old_memory, mem_size, device) -> list[(x,y)]
      - optional mixing of K memory samples per minibatch.
    """
    def __init__(
        self,
        mem_size: int = 2000,
        *,
        selector: Optional[Callable[[torch.nn.Module, Dataset, List[Tuple[Any, int]], int, torch.device], List[Tuple[Any, int]]]] = None,
        mix_per_mb: int = 0,
        shuffle_memory: bool = True,
    ):
        super().__init__()
        self.mem_size = int(mem_size)
        self.memory: List[Tuple[Any, int]] = []
        self.selector = selector
        self.mix_per_mb = int(mix_per_mb)
        self.shuffle_memory = bool(shuffle_memory)

    def before_training_exp(self, strategy, **kwargs):
        """Refresh memory using user-provided selector."""
        if self.selector is None:
            return
        dataset = strategy.experience.dataset
        new_mem = self.selector(
            strategy.model, dataset, self.memory, self.mem_size, strategy.device
        )
        if not isinstance(new_mem, list):
            raise TypeError("PAPA selector must return a list of (x, y) pairs")
        self.memory = new_mem[: self.mem_size]
        if self.shuffle_memory:
            random.shuffle(self.memory)

    def before_forward(self, strategy, **kwargs):
        """Optionally mix a few memory samples into the current minibatch."""
        if not (strategy.is_training and self.mix_per_mb > 0 and self.memory):
            return
        k = min(self.mix_per_mb, len(self.memory))
        mem_samples = random.sample(self.memory, k)

        mem_x, mem_y = [], []
        for x, y in mem_samples:
            y = y.to(strategy.device).long() if torch.is_tensor(y) else torch.tensor(int(y), device=strategy.device, dtype=torch.long)
            x = x if torch.is_tensor(x) else torch.as_tensor(x)
            mem_x.append(x.to(strategy.device))
            mem_y.append(y)

        # ensure shape consistency
        if mem_x[0].dim() == strategy.mb_x.dim():
            mem_x = torch.stack(mem_x)
        else:
            mem_x = torch.cat([t.unsqueeze(0) for t in mem_x], dim=0)
        mem_y = torch.stack(mem_y)

        strategy.mb_x = torch.cat([strategy.mb_x, mem_x], dim=0)
        strategy.mb_y = torch.cat([strategy.mb_y, mem_y], dim=0)

    @property
    def memory_dataset(self) -> Dataset:
        return _PairsDataset(self.memory)
