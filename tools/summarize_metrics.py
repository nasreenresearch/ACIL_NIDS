# tools/summarize_metrics.py
import sys, re
from pathlib import Path
import pandas as pd

PREFERRED_FILENAMES = ["eval_results.csv", "training_results.csv", "log.csv", "metrics.csv"]

def _resolve_csv_input(path_like) -> Path:
    p = Path(path_like)
    if p.is_dir():
        for name in PREFERRED_FILENAMES:
            cand = p / name
            if cand.exists():
                return cand
        csvs = sorted(p.glob("*.csv"))
        if not csvs:
            raise FileNotFoundError(f"No CSV files found in directory: {p}")
        return csvs[0]
    if not p.exists():
        raise FileNotFoundError(f"Path not found: {p}")
    return p

def _guess_phase_and_stream_from_filename(csv_path: Path):
    stem = csv_path.name.lower()
    if "eval" in stem:
        return "eval_phase", "test_stream"
    if "train" in stem:
        return "train_phase", "train_stream"
    return None, None

def _melt_if_wide(df: pd.DataFrame) -> pd.DataFrame:
    # If it already looks "long", don't melt
    if ("metric_name" in df.columns and "metric_value" in df.columns) or \
       ("name" in df.columns and "value" in df.columns):
        return df

    meta_candidates = [
        "experience", "eval_exp", "train_exp", "epoch", "iteration", "step",
        "phase", "stream", "timestamp", "exp", "run", "dataset", "split"
    ]
    meta_cols = [c for c in df.columns if c in meta_candidates]
    metric_cols = [c for c in df.columns if c not in meta_cols]

    # If ALL metric_cols are numeric-like and there are multiple -> wide table
    if any(mc for mc in metric_cols) and len(metric_cols) > 1:
        m = df.melt(id_vars=meta_cols, value_vars=metric_cols,
                    var_name="name", value_name="value")
        return m
    return df

def load_log(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Try to infer phase/stream from filename if missing
    phase_guess, stream_guess = _guess_phase_and_stream_from_filename(csv_path)
    if "phase" not in df.columns and phase_guess is not None:
        df["phase"] = phase_guess
    if "stream" not in df.columns and stream_guess is not None:
        df["stream"] = stream_guess

    # Handle wide format (many metric columns) -> long
    df = _melt_if_wide(df)

    # Normalize columns across Avalanche versions
    if "metric_name" in df.columns and "metric_value" in df.columns:
        df = df.rename(columns={"metric_name": "name", "metric_value": "value"})
    elif "metric" in df.columns and "value" in df.columns:
        df = df.rename(columns={"metric": "name"})

    # If still no 'name', synthesize from the first non-meta column
    if "name" not in df.columns:
        candidates = [c for c in df.columns if c not in ("value", "train_exp", "eval_exp", "experience")]
        if candidates:
            df = df.rename(columns={candidates[0]: "name"})
        else:
            df["name"] = "metric"

    if "value" not in df.columns:
        if "metric_value" in df.columns:
            df = df.rename(columns={"metric_value": "value"})
        else:
            raise ValueError(f"No 'value' column found. Columns: {df.columns.tolist()}")

    # Normalize experience column
    if "experience" not in df.columns:
        if "eval_exp" in df.columns:
            df["experience"] = df["eval_exp"]
        elif "train_exp" in df.columns:
            df["experience"] = df["train_exp"]
        else:
            exp_col = df["name"].astype(str).str.extract(r"/exp(\d+)", expand=False)
            df["experience"] = pd.to_numeric(exp_col, errors="coerce")

    # Coerce numeric
    for col in ("experience", "value"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

def tidy_tables(df: pd.DataFrame) -> dict:
    # Prefer evaluation on test stream if we have those hints
    mask_eval = df["phase"].eq("eval_phase") if "phase" in df.columns else True
    mask_test = df["stream"].eq("test_stream") if "stream" in df.columns else True
    dfe = df[mask_eval & mask_test].copy()
    if dfe.empty:
        # Fall back to the whole DF (e.g., training file)
        dfe = df.copy()

    # Case-insensitive patterns
    acc_patterns = r"(Top1_Acc|accuracy|acc|top1)"
    fog_patterns = r"(Forgetting|forget|forgetting)"
    bwt_patterns = r"(BWT|Backward)"

    # Ensure 'name' is str
    dfe["name"] = dfe["name"].astype(str)

    tables = {}

    # Accuracy per experience
    acc_mask = dfe["name"].str.contains(acc_patterns, case=False, na=False)
    acc = dfe[acc_mask].copy()
    if not acc.empty and "experience" in acc.columns:
        tables["accuracy_per_experience"] = (
            acc.groupby("experience")["value"].last().rename("accuracy").to_frame()
        )

    # Forgetting per experience
    fog_mask = dfe["name"].str.contains(fog_patterns, case=False, na=False)
    fog = dfe[fog_mask].copy()
    if not fog.empty and "experience" in fog.columns:
        tables["forgetting_per_experience"] = (
            fog.groupby("experience")["value"].last().rename("forgetting").to_frame()
        )

    # BWT per experience
    bwt_mask = dfe["name"].str.contains(bwt_patterns, case=False, na=False)
    bwt = dfe[bwt_mask].copy()
    if not bwt.empty and "experience" in bwt.columns:
        tables["bwt_per_experience"] = (
            bwt.groupby("experience")["value"].last().rename("bwt").to_frame()
        )

    # Stream (overall) accuracy if present anywhere
    if "name" in df.columns:
        stream_acc = df[df["name"].astype(str).str.contains(r"(Top1_Acc_Stream|accuracy_stream)", case=False, na=False)]
        if not stream_acc.empty:
            tables["accuracy_stream"] = stream_acc[["value"]].rename(columns={"value": "accuracy_stream"})

    # Fallback: if we still didn't get accuracy, guess a numeric "best" column
    if "accuracy_per_experience" not in tables and "experience" in dfe.columns:
        # pick the most "accuracy-like" first matching column from the wide form if present
        numeric_cols = []
        for c in df.columns:
            if c in ("value", "name", "experience", "phase", "stream", "eval_exp", "train_exp"):
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                numeric_cols.append(c)
        # Heuristic: prefer columns with 'acc' in their name
        best = None
        for c in numeric_cols:
            if re.search(r"(acc|accuracy|top1)", c, flags=re.I):
                best = c
                break
        if best is None and numeric_cols:
            best = numeric_cols[0]
        if best is not None:
            # Recompute per-experience "accuracy" from wide data
            if "value" not in df.columns or "name" not in df.columns:
                # wide path: group last by experience
                if "experience" in df.columns:
                    tables["accuracy_per_experience"] = (
                        df.groupby("experience")[best].last().rename("accuracy").to_frame()
                    )

    return tables

def main(csv_or_dir: str, outxlsx: str | None = None):
    csv_path = _resolve_csv_input(csv_or_dir)
    df = load_log(csv_path)
    tables = tidy_tables(df)

    outdir = csv_path.parent
    stem = csv_path.stem  # e.g., eval_results
    for name, t in tables.items():
        t.sort_index().to_csv(outdir / f"{stem}_{name}.csv")

    if outxlsx is None:
        outxlsx = outdir / f"{stem}_summary.xlsx"

    with pd.ExcelWriter(outxlsx, engine="xlsxwriter") as xl:
        df.to_excel(xl, sheet_name="raw_log", index=False)
        for name, t in tables.items():
            t.to_excel(xl, sheet_name=name)

    print("✅ Wrote:")
    print(f"  - Excel summary: {outxlsx}")
    for name in tables:
        print(f"  - {outdir / f'{stem}_{name}.csv'}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/summarize_metrics.py <results_dir_or_csv> [optional_output.xlsx]")
        sys.exit(1)
    main(*sys.argv[1:])
