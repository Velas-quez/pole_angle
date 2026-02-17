import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Simple Dataset class to convert Pandas to PyTorch Tensors
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
    Orchestrates the split, scaling, and loader creation.
    """
    # 1. Identify Columns
    # We assume 'target_' columns are outputs, everything else is input
    target_cols = [c for c in df.columns if "target" in c]
    feature_cols = [c for c in df.columns if c not in target_cols]

    
    # feature_cols = [c for c in df.columns if "target" not in c and "base" not in c]
    
    X = df[feature_cols].values
    y = df[target_cols].values

    # 2. Split Data (70% Train, 15% Val, 15% Test)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # 3. Scaling (The important part for Resuming)
    if existing_scaler is not None:
        print("Using provided scaler for normalization...")
        scaler = existing_scaler
        X_train = scaler.transform(X_train) # Don't fit! Use the old one
    else:
        print("Fitting new scaler on training data...")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

    # Always transform Val and Test using the scaler from Train
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # 4. Create PyTorch Objects
    train_ds = TactileDataset(X_train, y_train)
    val_ds = TactileDataset(X_val, y_val)
    test_ds = TactileDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    input_dim = len(feature_cols)
    
    return train_loader, val_loader, test_loader, scaler, input_dim