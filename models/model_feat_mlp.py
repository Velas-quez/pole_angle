import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.pytorch

# --- 0. Hyperparameters ---
PARAMS = {
    "batch_size": 128,
    "learning_rate": 0.01,
    "epochs": 200,
    "hidden_size_1": 64,
    "hidden_size_2": 32,
    "scheduler_patience": 5,
    "scheduler_factor": 0.5,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

print(f"Using device: {PARAMS['device']}")


# --- 1. Model Definition ---
class TactileMLP(nn.Module):
    def __init__(self, input_size):
        super(TactileMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, PARAMS["hidden_size_1"]),
            nn.ReLU(),
            nn.Linear(PARAMS["hidden_size_1"], PARAMS["hidden_size_2"]),
            nn.ReLU(),
            nn.Linear(PARAMS["hidden_size_2"], 2),  # Output: [roll, pitch]
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
def prepare_data(parquet_path, batch_size=128):
    df = pd.read_parquet(parquet_path)
    feature_cols = [c for c in df.columns if "target" not in c]
    target_cols = ["target_roll", "target_pitch"]

    X = df[feature_cols].values
    y = df[target_cols].values

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    train_loader = DataLoader(
        TactileDataset(X_train, y_train), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(TactileDataset(X_val, y_val), batch_size=batch_size)
    test_loader = DataLoader(TactileDataset(X_test, y_test), batch_size=batch_size)

    return train_loader, val_loader, test_loader, len(feature_cols), scaler


# --- 4. Training Loop ---
def train_model(model, train_loader, val_loader, epochs=100, device="cpu"):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=PARAMS["learning_rate"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=PARAMS["scheduler_factor"],
        patience=PARAMS["scheduler_patience"],
    )

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)  # Push to GPU
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        mae_roll = 0
        mae_pitch = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
                errors = torch.abs(outputs - targets)
                mae_roll += errors[:, 0].mean().item()
                mae_pitch += errors[:, 1].mean().item()

        avg_val_loss = val_loss / len(val_loader)
        avg_mae_roll = mae_roll / len(val_loader)
        avg_mae_pitch = mae_pitch / len(val_loader)

        scheduler.step(avg_val_loss)

        mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
        mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
        mlflow.log_metric("val_mae_roll", avg_mae_roll, step=epoch)
        mlflow.log_metric("val_mae_pitch", avg_mae_pitch, step=epoch)

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch} | Val Loss: {avg_val_loss:.6f} | Roll Err: {avg_mae_roll:.4f}° | Pitch Err: {avg_mae_pitch:.4f}°"
            )


# --- Execution ---
if __name__ == "__main__":
    mlflow.set_experiment("Tactile_Pole_Angle_Estimation")

    device = PARAMS["device"]
    filename = "final_training_data.parquet"
    train_ld, val_ld, test_ld, input_dim, scaler = prepare_data(
        filename, batch_size=PARAMS["batch_size"]
    )

    model = TactileMLP(input_size=input_dim).to(device)  # Push Model to Device

    with mlflow.start_run():
        mlflow.log_params(PARAMS)

        train_model(model, train_ld, val_ld, epochs=PARAMS["epochs"], device=device)

        # --- Final Evaluation ---
        model.eval()
        test_mse = 0
        test_mae_roll = 0
        test_mae_pitch = 0

        with torch.no_grad():
            for inputs, targets in test_ld:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                test_mse += nn.MSELoss()(outputs, targets).item()
                errors = torch.abs(outputs - targets)
                test_mae_roll += errors[:, 0].mean().item()
                test_mae_pitch += errors[:, 1].mean().item()

        avg_mse = test_mse / len(test_ld)
        final_roll_err = test_mae_roll / len(test_ld)
        final_pitch_err = test_mae_pitch / len(test_ld)

        mlflow.log_metric("test_mse", avg_mse)
        mlflow.log_metric("test_mae_roll", final_roll_err)
        mlflow.log_metric("test_mae_pitch", final_pitch_err)

        # Saves the model architecture and weights for reproducibility
        mlflow.pytorch.log_model(model, "model")

        print("\n" + "=" * 30)
        print("      FINAL TEST RESULTS      ")
        print("=" * 30)
        print(f"Test MSE:        {avg_mse:.6f}")
        print(f"Roll Avg Error:  {final_roll_err:.4f}°")
        print(f"Pitch Avg Error: {final_pitch_err:.4f}°")
        print("=" * 30)

