import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


class TactileDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_loaders(df, batch_size=128, existing_scaler=None):
    """
    Orchestrates the continuous split (80/20), scaling, and loader creation.
    Validation is the last 20% (continuous).
    Training is the first 80% (shuffled during loading).
    """
    # 1. Identify Columns
    target_cols = [c for c in df.columns if "target" in c]
    feature_cols = [c for c in df.columns if "target" not in c]

    # 2. Sequential Split (NO SHUFFLE YET)
    split_idx = int(len(df) * 0.8)

    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]

    X_train = train_df[feature_cols].values
    y_train = train_df[target_cols].values

    X_val = val_df[feature_cols].values
    y_val = val_df[target_cols].values

    # 3. Scaling (Resuming logic preserved)
    if existing_scaler is not None:
        print("Using provided scaler for normalization...")
        scaler = existing_scaler
    else:
        print("Fitting new scaler on training data...")
        scaler = StandardScaler()
        scaler.fit(X_train)  # Fit only on training data

    # Transform both sets
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)

    # 4. Create PyTorch Objects
    train_ds = TactileDataset(X_train, y_train)
    val_ds = TactileDataset(X_val, y_val)

    # 5. Create Loaders
    # shuffle=True here only shuffles the training samples per epoch
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # shuffle=False for validation to keep it as a continuous sequence replay
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    input_dim = len(feature_cols)
    output_dim = len(target_cols)

    print(f"✅ Data Ready: {len(train_ds)} train samples, {len(val_ds)} val samples.")

    # return None for test_loader
    return train_loader, val_loader, None, scaler, input_dim, output_dim
