import pandas as pd, numpy as np, torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class TabDataset(Dataset):
    def __init__(self, X, y): self.X, self.y = X, y
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

def load_cicids2017(csv_path: str, label_col: str = "Label"):
    df = pd.read_csv(csv_path)
    # numeric features only (preprocessed CSVs typically numeric + Label)
    feat_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feat_cols = [c for c in feat_cols if c != label_col]
    labels = df[label_col].astype(str).unique().tolist()
    # put BENIGN first to match many IDS conventions
    labels = ["BENIGN"] + sorted([l for l in labels if l != "BENIGN"])
    lab2id = {l:i for i,l in enumerate(labels)}
    X = df[feat_cols].astype(float).values
    y = df[label_col].astype(str).map(lab2id).values
    X = torch.tensor(StandardScaler().fit_transform(X), dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    return TabDataset(X, y), labels
