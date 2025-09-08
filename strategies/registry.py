# strategies/registry.py
import torch
from avalanche.training.supervised import Naive, JointTraining
from avalanche.training.plugins import ReplayPlugin, EWCPlugin, LwFPlugin, GEMPlugin

def make_strategy(name, model, optimizer, criterion, device,
                  mem_size=2000, ewc_lambda=0.001, lwf_alpha=1.0, lwf_temp=2.0,
                  gem_patterns=256, gem_strength=0.5,
                  train_mb_size=128, eval_mb_size=128, train_epochs=1,
                  extra_plugins=None):
    """
    Build a supervised continual learning strategy with optional plugins.

    Pass custom plugins via `extra_plugins`:
      - CBRS: ReplayPlugin(mem_size=..., storage_policy=CBRSBuffer(...))
      - PAPA: PAPAPlugin(...)
    """
    name = name.lower()
    plugins = [] if extra_plugins is None else list(extra_plugins)

    if name == "naive":
        return Naive(model, optimizer, criterion,
                     device=device, train_mb_size=train_mb_size,
                     train_epochs=train_epochs, eval_mb_size=eval_mb_size)

    if name == "replay":
        # If caller didn't provide a ReplayPlugin (e.g., with CBRS), add default.
        if not any(isinstance(p, ReplayPlugin) for p in plugins):
            plugins.append(ReplayPlugin(mem_size=mem_size))
        return Naive(model, optimizer, criterion, device=device,
                     train_mb_size=train_mb_size, train_epochs=train_epochs,
                     eval_mb_size=eval_mb_size, plugins=plugins)

    if name == "ewc":
        plugins.append(EWCPlugin(ewc_lambda=ewc_lambda))
        return Naive(model, optimizer, criterion, device=device,
                     train_mb_size=train_mb_size, train_epochs=train_epochs,
                     eval_mb_size=eval_mb_size, plugins=plugins)

    if name == "lwf":
        plugins.append(LwFPlugin(alpha=lwf_alpha, temperature=lwf_temp))
        return Naive(model, optimizer, criterion, device=device,
                     train_mb_size=train_mb_size, train_epochs=train_epochs,
                     eval_mb_size=eval_mb_size, plugins=plugins)

    if name == "gem":
        plugins.append(GEMPlugin(patterns_per_experience=gem_patterns,
                                 memory_strength=gem_strength))
        return Naive(model, optimizer, criterion, device=device,
                     train_mb_size=train_mb_size, train_epochs=train_epochs,
                     eval_mb_size=eval_mb_size, plugins=plugins)

    if name in ("cumulative", "offline", "joint"):
        return JointTraining(model, optimizer, criterion,
                             device=device, train_mb_size=train_mb_size,
                             train_epochs=train_epochs, eval_mb_size=eval_mb_size)

    if name == "papa":
        # Expect caller to pass a PAPAPlugin via extra_plugins
        return Naive(model, optimizer, criterion, device=device,
                     train_mb_size=train_mb_size, train_epochs=train_epochs,
                     eval_mb_size=eval_mb_size, plugins=plugins)

    raise ValueError(f"Unknown strategy {name}")
