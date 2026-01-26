import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import math

# --- 1. Model Definition ---
class TactileMLP(nn.Module):
    def __init__(self, input_size):
        super(TactileMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # Output: [roll, pitch]
        )
        
    def forward(self, x):
        return self.network(x)

# --- 2. Custom Dataset ---
class TactileDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

# --- 3. Data Preparation Function ---
def prepare_data(parquet_path, batch_size=64):
    # Load data (works for parquet or csv)
    df = pd.read_parquet(parquet_path)
    
    # Define features and targets
    feature_cols = [c for c in df.columns if 'target' not in c]
    target_cols = ['target_roll', 'target_pitch']
    
    X = df[feature_cols].values
    y = df[target_cols].values
    
    # Split: 70% Train, 15% Val, 15% Test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Scale features (Crucial for MLPs!)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Create DataLoaders
    train_loader = DataLoader(TactileDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TactileDataset(X_val, y_val), batch_size=batch_size)
    test_loader = DataLoader(TactileDataset(X_test, y_test), batch_size=batch_size)
    
    return train_loader, val_loader, test_loader, len(feature_cols), scaler

# --- 4. Training Loop ---
def train_model(model, train_loader, val_loader, epochs=200):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Train Loss: {train_loss/len(train_loader):.6f} | Val Loss: {val_loss/len(val_loader):.6f}")

# --- Execution ---
# --- Execution ---
if __name__ == "__main__":
   

    filename = 'training_data.parquet' 
    train_ld, val_ld, test_ld, input_dim, scaler = prepare_data(filename)
    
    model = TactileMLP(input_size=input_dim)
    print(f"Model initialized with input dimension: {input_dim}")
    
    # --- 1. Train ---
    train_model(model, train_ld, val_ld, epochs=100)
    
    # --- 2. Test (Final Evaluation) ---
    model.eval()
    test_mse = 0
    test_mae_deg = 0 # Now directly in degrees
    
    with torch.no_grad():
        for inputs, targets in test_ld:
            outputs = model(inputs)
            
            # MSE in degrees^2
            loss = nn.MSELoss()(outputs, targets)
            test_mse += loss.item()
            
            # Mean Absolute Error directly in degrees
            test_mae_deg += torch.abs(outputs - targets).mean().item()

    # Calculate averages based on the number of batches
    avg_mse = test_mse / len(test_ld)
    avg_mae_final = test_mae_deg / len(test_ld)
    
    print("\n" + "="*30)
    print("      FINAL TEST RESULTS      ")
    print("="*30)
    print(f"Test MSE (Loss): {avg_mse:.6f}")
    print(f"Avg Error:       {avg_mae_final:.4f}°")
    print("="*30)
