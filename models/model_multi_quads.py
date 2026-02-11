import pandas as pd
import joblib
from os.path import join as pjoin
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.pytorch
from typing import Tuple, List

# --- 0. Hyperparameters ---
PARAMS = {
    "batch_size": 128,
    "learning_rate": 0.005,
    "epochs": 200,
    "hidden_size_1": 128,
    "hidden_size_2": 64,
    "hidden_size_3": 32,
    "dropout": 0.1,
    "pitch_weight": 2.0,  # Give more importance to Pitch accuracy
    "manifold_weight": 0.1,  # Penalty for sin^2 + cos^2 != 1
    "scheduler_patience": 7,
    "scheduler_factor": 0.6,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


# --- 1. Custom Weighted Loss Function ---
class WeightedTrigLoss(nn.Module):
    def __init__(self, pitch_weight: float = 2.0, manifold_weight: float = 0.1):
        super(WeightedTrigLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.p_weight = pitch_weight
        self.m_weight = manifold_weight

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor):
        # predictions/targets: [sin_roll, cos_roll, cos_pitch]

        # 1. Roll MSE (Trigonometric components)
        loss_r_sin = self.mse(predictions[:, 0], targets[:, 0])
        loss_r_cos = self.mse(predictions[:, 1], targets[:, 1])
        roll_loss = loss_r_sin + loss_r_cos

        # 2. Pitch MSE (Single monotonic component)
        pitch_loss = self.mse(predictions[:, 2], targets[:, 2])

        # 3. Unit Circle Penalty (sin^2 + cos^2 should be 1)
        # Forces the model to keep Roll on the unit circle manifold
        manifold_err = self.mse(
            predictions[:, 0] ** 2 + predictions[:, 1] ** 2,
            torch.ones_like(predictions[:, 0]),
        )

        total_loss = (
            (self.p_weight * pitch_loss) + roll_loss + (self.m_weight * manifold_err)
        )

        return total_loss, roll_loss.item(), pitch_loss.item()


# --- 2. Model Definition ---
class TactileMLP(nn.Module):
    def __init__(self, input_size: int):
        super(TactileMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, PARAMS["hidden_size_1"]),
            nn.BatchNorm1d(PARAMS["hidden_size_1"]),
            nn.ReLU(),
            nn.Dropout(PARAMS["dropout"]),
            nn.Linear(PARAMS["hidden_size_1"], PARAMS["hidden_size_2"]),
            nn.BatchNorm1d(PARAMS["hidden_size_2"]),
            nn.ReLU(),
            nn.Linear(PARAMS["hidden_size_2"], PARAMS["hidden_size_3"]),
            nn.BatchNorm1d(PARAMS["hidden_size_3"]),
            nn.ReLU(),
            # 3 outputs: [sin_roll, cos_roll, cos_pitch]
            nn.Linear(PARAMS["hidden_size_3"], 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# --- 3. Data & Helper Functions ---
class TactileDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


def get_angles_from_output(output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reconstructs Roll [0,360) and Pitch [0,180] from trig components."""
    sin_r, cos_r, cos_p = output[:, 0], output[:, 1], output[:, 2]
    roll_deg = torch.rad2deg(torch.atan2(sin_r, cos_r)) % 360
    pitch_deg = torch.rad2deg(torch.acos(torch.clamp(cos_p, -1.0, 1.0)))
    return roll_deg, pitch_deg


def prepare_data(parquet_path: str, batch_size: int):
    df = pd.read_parquet(parquet_path)
    feature_cols = [c for c in df.columns if "target" not in c]

    # Trig conversions
    r_rad, p_rad = (
        np.radians(df["target_roll"].values),
        np.radians(df["target_pitch"].values),
    )
    y = np.stack([np.sin(r_rad), np.cos(r_rad), np.cos(p_rad)], axis=1)

    X = df[feature_cols].values
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    # joblib.dump(scaler, pjoin("preprocessing", "scaler.pkl"))
    # raise ValueError
    X_val, X_test = scaler.transform(X_val), scaler.transform(X_test)

    train_loader = DataLoader(
        TactileDataset(X_train, y_train), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(TactileDataset(X_val, y_val), batch_size=batch_size)
    test_loader = DataLoader(TactileDataset(X_test, y_test), batch_size=batch_size)

    return train_loader, val_loader, test_loader, len(feature_cols)


# --- 4. Training Loop ---
def train_model(model, train_loader, val_loader, device):
    criterion = WeightedTrigLoss(PARAMS["pitch_weight"], PARAMS["manifold_weight"])
    optimizer = optim.Adam(model.parameters(), lr=PARAMS["learning_rate"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=PARAMS["scheduler_patience"]
    )

    for epoch in range(PARAMS["epochs"]):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss, _, _ = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        v_loss, mae_r, mae_p = 0.0, 0.0, 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                l_total, _, _ = criterion(outputs, targets)
                v_loss += l_total.item()

                # Accuracy tracking in degrees
                pr, pp = get_angles_from_output(outputs)
                tr, tp = get_angles_from_output(targets)
                r_err = torch.abs(pr - tr)
                mae_r += torch.min(r_err, 360 - r_err).mean().item()
                mae_p += torch.abs(pp - tp).mean().item()

        avg_loss = v_loss / len(val_loader)
        scheduler.step(avg_loss)
        mlflow.log_metrics(
            {
                "val_loss": avg_loss,
                "mae_roll_deg": mae_r / len(val_loader),
                "mae_pitch_deg": mae_p / len(val_loader),
            },
            step=epoch,
        )

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch} | Roll Err: {mae_r / len(val_loader):.2f}° | Pitch Err: {mae_p / len(val_loader):.2f}°"
            )


if __name__ == "__main__":
    filepath = pjoin("data", "final_training_data.parquet")
    mlflow.set_experiment("Tactile_Weighted_Trig")
    dev = PARAMS["device"]
    train_ld, val_ld, test_ld, in_dim = prepare_data(filepath, PARAMS["batch_size"])

    model = TactileMLP(in_dim).to(dev)
    with mlflow.start_run():
        mlflow.log_params(PARAMS)
        train_model(model, train_ld, val_ld, dev)
        mlflow.pytorch.log_model(model, "model")

