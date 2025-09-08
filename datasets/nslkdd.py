import pandas as pd, numpy as np, torch
from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from typing import Optional, Tuple, Dict


class TabDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


def _find_label_and_diff_cols(df: pd.DataFrame) -> Tuple[str, Optional[str]]:
    """Return (label_col, difficulty_col_or_None). Handles CSVs with/without difficulty column."""
    # Common names seen in NSL-KDD CSVs
    name_candidates = ["label", "Label", "class", "Class", "attack", "Attack"]
    for c in name_candidates:
        if c in df.columns:
            # difficulty sometimes present with these names
            diff_candidates = ["difficulty", "difficulty_level", "difficulty score", "Difficulty"]
            diff_col = next((d for d in diff_candidates if d in df.columns), None)
            return c, diff_col

    # No obvious name: infer by types at the end
    # If last col is numeric and second-last is stringy -> assume [-2] is label and [-1] is difficulty
    last = df.columns[-1]
    second_last = df.columns[-2] if len(df.columns) >= 2 else None
    if second_last is not None:
        if pd.api.types.is_object_dtype(df[second_last]) and pd.api.types.is_numeric_dtype(df[last]):
            return second_last, last

    # Otherwise assume last column is the label
    return last, None


def _get_categorical_columns(df: pd.DataFrame) -> list:
    """Prefer names (protocol_type, service, flag). Fall back to indices [1,2,3]."""
    by_name = []
    for cand in ["protocol_type", "protocol", "service", "flag"]:
        if cand in df.columns:
            by_name.append(cand)
    # If we found all three by name, return them
    if len(set(by_name) & {"protocol_type", "service", "flag"}) == 3:
        return ["protocol_type", "service", "flag"]

    # Fallback: indices 1,2,3 if they exist
    if df.shape[1] >= 4:
        return [df.columns[1], df.columns[2], df.columns[3]]
    return []


def load_nslkdd(
    csv_path: str,
    header: Optional[int] = "infer",
    *,
    fit: bool = True,
    enc: Optional[OneHotEncoder] = None,
    scaler: Optional[StandardScaler] = None,
    lab2id: Optional[Dict[str, int]] = None,
) -> Tuple[TabDataset, list, OneHotEncoder, StandardScaler, Dict[str, int]]:
    """
    Robust loader for KDDTrain.csv / KDDTest.csv (NSL-KDD).
    If fit=True: fits OneHotEncoder, StandardScaler, and label map on this file.
    If fit=False: uses provided enc, scaler, and lab2id to transform consistently.

    Returns:
      dataset, labels(list in id order), enc, scaler, lab2id
    """
    df = pd.read_csv(csv_path, header=header)

    # Identify label & optional difficulty columns
    label_col, diff_col = _find_label_and_diff_cols(df)

    # Identify categorical columns
    cat_cols = _get_categorical_columns(df)
    cats = df[cat_cols].astype(str).values if cat_cols else np.empty((len(df), 0), dtype=str)

    # Numeric features: drop cat + label + difficulty
    drop_cols = set(cat_cols + [label_col])
    if diff_col is not None:
        drop_cols.add(diff_col)
    X_num = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    # Keep only numeric
    X_num = X_num.select_dtypes(include=[np.number]).astype(float).values

    # Encoder: support both new/old scikit-learn kwargs
    if fit:
        try:
            enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        except TypeError:
            enc = OneHotEncoder(sparse=False, handle_unknown="ignore")
        X_cat = enc.fit_transform(cats) if cats.size else np.empty((len(df), 0), dtype=float)
    else:
        if enc is None:
            raise ValueError("enc must be provided when fit=False")
        X_cat = enc.transform(cats) if cats.size else np.empty((len(df), 0), dtype=float)

    X = np.hstack([X_num, X_cat])

    # Labels
    y_text = df[label_col].astype(str)
    if fit:
        # Order labels deterministically: sorted by name (you can change to frequency if you prefer)
        labels = sorted(y_text.unique().tolist())
        lab2id = {l: i for i, l in enumerate(labels)}
    else:
        if lab2id is None:
            raise ValueError("lab2id must be provided when fit=False")
        labels = [None] * len(lab2id)
        for k, v in lab2id.items():
            if v < len(labels):
                labels[v] = k

    # Map labels; unknowns will raise a KeyError to signal mismatch early
    try:
        y = y_text.map(lab2id).values
        if np.any(pd.isna(y)):
            unknown = sorted(set(y_text[pd.isna(y)].tolist()))
            raise KeyError(f"Found unseen labels in '{csv_path}': {unknown}")
    except Exception as e:
        raise

    # Scale numeric+oh features
    if fit:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        if scaler is None:
            raise ValueError("scaler must be provided when fit=False")
        X = scaler.transform(X)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    return TabDataset(X, y), labels, enc, scaler, lab2id
