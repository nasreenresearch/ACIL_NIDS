# Continual NIDS (ACIL Benchmark Framework)

This repository provides a benchmark framework for evaluating continual learning (CL) strategies in the context of network intrusion detection systems (NIDS).  
The supported datasets are CICIDS2017, CICIDS2018, and NSL-KDD.

---

## Project Structure

```
CL_ANIDS/
├─ run.py                        # Entry point to run a single experiment
├─ tools/
│  ├─ summarize_metrics.py       # Summarize raw logs into tidy CSVs and Excel
│  ├─ plot_metrics.py            # Plot accuracy per experience
│  ├─ combine_runs.py            # Combine results from multiple strategies
│  └─ auto_train.py              # Automate experiments across datasets/strategies
├─ benchmarks/
│  └─ make_benchmark.py          # Class-incremental benchmark builder
├─ datasets/                     # Loader for Datasets
│  ├─ cicids2017.py
│  ├─ cicids2018.py
│  └─ nslkdd.py                  
├─ models/
│  └─ mlp.py
├─ strategies/
│  ├─ registry.py
│  ├─ cbrs.py
│  └─ papa.py
└─ Data/                         # All dataset CSVs here
```

---

## 1. Environment Setup

### Install
```bash
cd CL_ANIDS
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip

# Core dependencies
pip install torch torchvision torchaudio
pip install avalanche-lib pandas numpy scikit-learn matplotlib xlsxwriter tqdm
```

---

## 2. Dataset Preparation

Place the dataset CSVs under the `Data/` directory (or specify custom paths):

```
Data/
├─ CICIDS2017_train.csv
├─ CICIDS2017_test.csv
├─ CICIDS2018_train.csv
├─ CICIDS2018_test.csv
├─ KDDTrain.csv
└─ KDDTest.csv
```

The `nslkdd.py` loader automatically detects categorical columns, scales numerical features, and ensures consistent label encoding between training and test sets.

---

## 3. Supported Strategies

| Strategy | Description |
|----------|-------------|
| **Replay** | Stores a subset of past samples in memory and replays them during training to reduce forgetting. |
| **GEM (Gradient Episodic Memory)** | Constrains gradient updates to avoid increasing loss on previous tasks. |
| **EWC (Elastic Weight Consolidation)** | Uses Fisher information to regularize important weights and preserve old knowledge. |
| **CBRS (Class-Balanced Reservoir Sampling)** | Maintains a balanced memory buffer across classes to ensure fair replay. |
| **PAPA (Parameter-Allocation Per-Attribute)** | Dynamically allocates parameters per task/attribute for better knowledge retention. |

---

## 4. Running an Experiment

### Example: GEM on CICIDS2017
```bash
python run.py   --dataset cicids2017   --train Data/CICIDS2017_train.csv   --test  Data/CICIDS2017_test.csv   --strategy gem   --epochs 5   --mem_size 2000   --seed 1   --outdir results
```

### Example: Replay on NSL-KDD
```bash
python run.py   --dataset nslkdd   --train Data/KDDTrain.csv   --test  Data/KDDTest.csv   --strategy replay   --epochs 5   --mem_size 2000   --seed 1   --outdir results
```

By default, the benchmark uses a fixed class order sorted by descending frequency in the training set.

---

## 5. Summarizing Results

After a run, Avalanche writes logs into a directory such as `results/gem_cicids2017.csv/`.

Summarize the raw logs:
```bash
python tools/summarize_metrics.py results/gem_cicids2017.csv
```

This creates:

- `eval_results_summary.xlsx`
- `eval_results_accuracy_per_experience.csv`
- `eval_results_forgetting_per_experience.csv`
- `eval_results_bwt_per_experience.csv`

---

## 6. Plotting Results

### Accuracy per experience (single run)
```bash
python tools/plot_metrics.py   results/gem_cicids2017.csv/eval_results_accuracy_per_experience.csv   "GEM on CIC-IDS-2017: Accuracy per Experience"
```

### Compare strategies

1. Combine results:
```bash
python tools/combine_runs.py   results/gem_cicids2017.csv/eval_results_accuracy_per_experience.csv   results/ewc_cicids2017.csv/eval_results_accuracy_per_experience.csv   results/replay_cicids2017.csv/eval_results_accuracy_per_experience.csv
```

2. Plot comparison:
```bash
python tools/plot_combined.py results/combined_accuracy_per_experience.csv   "CIC-IDS-2017: Accuracy per Experience (GEM vs EWC vs Replay)"
```

---

## 7. Automating Multiple Runs

To run several strategies in one go, use:

```bash
python -m tools.auto_train   --datasets cicids2017   --train Data/CICIDS2017_train.csv   --test  Data/CICIDS2017_test.csv   --strategies gem ewc replay   --epochs 5 --mem_size 2000 --seed 1   --outdir results
```

This creates:

```
results/
└─ cicids2017/
   ├─ gem/
   │  ├─ eval_results.csv
   │  ├─ eval_results_summary.xlsx
   │  └─ eval_results_accuracy_per_experience.csv
   ├─ ewc/
   └─ replay/
```

---

## 8. Notes

- Replay buffer size is set with `--mem_size`.
- For CBRS: use `--strategy cbrs`.
- For PAPA: use `--strategy papa` (requires `strategies/papa.py`).
- Logs are reproducible with `--seed`.
- Each automated run also saves a `config.json` with all parameters.


---

## 9. Citing this Work

