# tools/auto_train.py
import argparse
import json
from pathlib import Path
from datetime import datetime

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
from strategies.papa import PAPAPlugin  # selector can be wired later


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


def run_one(dataset, train_csv, test_csv, strategy_name, out_root, mem_size, epochs, seed, per_exp):
    torch.manual_seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr, te, classes = load_ds(dataset, train_csv, test_csv)

    x0 = tr[0][0]
    in_dim = x0.numel() if torch.is_tensor(x0) else int(x0.shape[0])
    n_classes = len(classes)

    model = make_mlp(in_dim, n_classes).to(device)
    opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    num_per_exp = per_exp if per_exp is not None else [1] + [3] * 5
    scenario = make_cil(tr, te, classes, num_classes_per_exp=num_per_exp, seed=seed)

    run_dir = Path(out_root) / dataset / strategy_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = {
        "dataset": dataset,
        "train": str(train_csv),
        "test": str(test_csv),
        "strategy": strategy_name,
        "mem_size": mem_size,
        "epochs": epochs,
        "seed": seed,
        "per_exp": per_exp,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    csv_logger = CSVLogger(run_dir)
    int_logger = InteractiveLogger()
    evalp = EvaluationPlugin(
        accuracy_metrics(epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        bwt_metrics(experience=True, stream=True),
        timing_metrics(epoch=True, experience=True),
        loggers=[csv_logger, int_logger],
    )

    extra_plugins = []
    if strategy_name == "cbrs":
        cbrs = CBRSBuffer(mem_size)
        extra_plugins = [ReplayPlugin(mem_size=mem_size, storage_policy=cbrs)]
        strategy = make_strategy(
            "replay", model, opt, criterion, device,
            mem_size=mem_size, train_epochs=epochs,
            extra_plugins=extra_plugins,
        )
    elif strategy_name == "papa":
        # Wire your selector when available: selector=your_callable
        papa = PAPAPlugin(mem_size=mem_size, selector=None, mix_per_mb=0)
        extra_plugins = [papa]
        strategy = make_strategy(
            "papa", model, opt, criterion, device,
            mem_size=mem_size, train_epochs=epochs,
            extra_plugins=extra_plugins,
        )
    else:
        strategy = make_strategy(
            strategy_name, model, opt, criterion, device,
            mem_size=mem_size, train_epochs=epochs,
        )

    strategy.evaluator = evalp

    for exp in scenario.train_stream:
        strategy.train(exp, eval_streams=[scenario.test_stream])

    # Auto-summarize
    try:
        from tools.summarize_metrics import main as summarize_main
        summarize_main(str(run_dir))
    except Exception as e:
        print(f"⚠️  Summarizer skipped for {strategy_name} due to: {e}")

    print(f"✅ Finished {strategy_name} on {dataset}. Logs in: {run_dir}")
    return run_dir


def main():
    ap = argparse.ArgumentParser(description="Automate CL strategy runs and save results.")
    ap.add_argument("--datasets", nargs="+", required=True, choices=["cicids2017", "cicids2018", "nslkdd"])
    ap.add_argument("--train", nargs="+", required=True, help="Training CSV(s) aligned with --datasets.")
    ap.add_argument("--test", nargs="+", required=True, help="Test CSV(s) aligned with --datasets.")
    ap.add_argument("--strategies", nargs="+", required=True,
                    choices=["naive", "replay", "ewc", "gem", "lwf", "cumulative", "cbrs", "papa"])
    ap.add_argument("--mem_size", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--per_exp", type=int, nargs="+", default=None)
    ap.add_argument("--outdir", type=str, default="results")
    args = ap.parse_args()

    if len(args.train) != len(args.datasets) or len(args.test) != len(args.datasets):
        raise ValueError("Provide the same number of --train and --test paths as --datasets.")

    train_map = {ds: Path(p).resolve() for ds, p in zip(args.datasets, args.train)}
    test_map = {ds: Path(p).resolve() for ds, p in zip(args.datasets, args.test)}

    all_run_dirs = []
    for ds in args.datasets:
        for strat in args.strategies:
            run_dir = run_one(
                dataset=ds,
                train_csv=train_map[ds],
                test_csv=test_map[ds],
                strategy_name=strat,
                out_root=args.outdir,
                mem_size=args.mem_size,
                epochs=args.epochs,
                seed=args.seed,
                per_exp=args.per_exp,
            )
            all_run_dirs.append(run_dir)

    # try combining accuracy across strategies per dataset
    try:
        from tools.combine_runs import main as combine_main
        for ds in args.datasets:
            acc_csvs = []
            for strat in args.strategies:
                cand = Path(args.outdir) / ds / strat / "eval_results_accuracy_per_experience.csv"
                if cand.exists():
                    acc_csvs.append(str(cand))
            if len(acc_csvs) >= 2:
                combine_main(acc_csvs)
    except Exception as e:
        print(f"ℹ️ Combine step skipped: {e}")

    print("🎉 All requested runs completed.")


if __name__ == "__main__":
    main()
