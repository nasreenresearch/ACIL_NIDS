import argparse
from pathlib import Path

import torch
import torch.optim as optim
import torch.nn as nn

from avalanche.evaluation.metrics import (
    accuracy_metrics,
    forgetting_metrics,
    bwt_metrics,
    timing_metrics,
)
from avalanche.logging import InteractiveLogger, CSVLogger
from avalanche.training.plugins import EvaluationPlugin, ReplayPlugin

from models.mlp import make_mlp
from strategies.registry import make_strategy
from benchmarks.make_benchmark import make_cil

from datasets.cicids2017 import load_cicids2017
from datasets.cicids2018 import load_cicids2018
from datasets.nslkdd import load_nslkdd
from strategies.cbrs import CBRSBuffer
from strategies.papa import PAPAPlugin  # you can keep selector=None for now


def load_ds(name, train_path, test_path):
    if name == "cicids2017":
        tr, classes = load_cicids2017(train_path)
        te, _ = load_cicids2017(test_path)
    elif name == "cicids2018":
        tr, classes = load_cicids2018(train_path)
        te, _ = load_cicids2018(test_path)
    elif name == "nslkdd":
        tr, classes, enc, scaler, lab2id = load_nslkdd(train_path, fit=True)
        te, _, _, _, _ = load_nslkdd(test_path, fit=False, enc=enc, scaler=scaler, lab2id=lab2id)
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return tr, te, classes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["cicids2017", "cicids2018", "nslkdd"])
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument(
        "--strategy",
        required=True,
        choices=["naive", "replay", "ewc", "gem", "lwf", "cumulative", "cbrs", "papa"],
    )
    ap.add_argument("--mem_size", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--per_exp", type=int, nargs="+", default=None,
                    help="num classes per experience, e.g. --per_exp 1 3 3 3")
    ap.add_argument("--outdir", type=str, default="results",
                    help="Where to save logs (CSV) and artifacts")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    try:
        import numpy as np
        np.random.seed(args.seed)
    except Exception:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr, te, classes = load_ds(args.dataset, args.train, args.test)

    x0 = tr[0][0]
    in_dim = x0.numel() if torch.is_tensor(x0) else int(x0.shape[0])
    n_classes = len(classes)

    model = make_mlp(in_dim, n_classes).to(device)
    opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # CIL schedule: default 1 + 3 + 3 ... if not provided
    num_per_exp = args.per_exp if args.per_exp is not None else [1] + [3] * 5

    scenario = make_cil(tr, te, classes, num_classes_per_exp=num_per_exp, seed=args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    log_dir = outdir / f"{args.strategy}_{args.dataset}"  # CSVLogger treats path as a folder
    log_dir.mkdir(parents=True, exist_ok=True)

    csv_logger = CSVLogger(log_dir)
    int_logger = InteractiveLogger()
    evalp = EvaluationPlugin(
        accuracy_metrics(epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        bwt_metrics(experience=True, stream=True),
        timing_metrics(epoch=True, experience=True),
        loggers=[csv_logger, int_logger],
    )

    extra_plugins = []
    if args.strategy == "cbrs":
        cbrs = CBRSBuffer(args.mem_size)
        extra_plugins = [ReplayPlugin(mem_size=args.mem_size, storage_policy=cbrs)]
        strategy = make_strategy(
            "replay", model, opt, criterion, device,
            mem_size=args.mem_size, train_epochs=args.epochs,
            extra_plugins=extra_plugins,
        )
    elif args.strategy == "papa":
        # Plug in your selector when ready: selector=your_callable
        papa = PAPAPlugin(mem_size=args.mem_size, selector=None, mix_per_mb=0)
        extra_plugins = [papa]
        strategy = make_strategy(
            "papa", model, opt, criterion, device,
            mem_size=args.mem_size, train_epochs=args.epochs,
            extra_plugins=extra_plugins,
        )
    else:
        strategy = make_strategy(
            args.strategy, model, opt, criterion, device,
            mem_size=args.mem_size, train_epochs=args.epochs,
        )

    # Attach evaluator AFTER construction (your wrapper doesn’t forward it)
    strategy.evaluator = evalp

    # Train / Evaluate
    for exp in scenario.train_stream:
        strategy.train(exp, eval_streams=[scenario.test_stream])

    print(f"✅ DONE. Logs in: {log_dir}")


if __name__ == "__main__":
    main()
