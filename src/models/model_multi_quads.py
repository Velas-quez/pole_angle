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
    "learning_rate": 0.002,  # Reduzido levemente para estabilidade com Cosine Loss
    "epochs": 400,
    "hidden_size_1": 256,
    "hidden_size_2": 128,
    "hidden_size_3": 64,
    "hidden_size_4": 32,
    "dropout": 0.10,
    "scheduler_patience": 8,
    "scheduler_factor": 0.5,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


# --- 2. Model Definition with L2 Normalization ---
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
            nn.Dropout(PARAMS["dropout"]),
            nn.Linear(PARAMS["hidden_size_2"], PARAMS["hidden_size_3"]),
            nn.BatchNorm1d(PARAMS["hidden_size_3"]),
            nn.ReLU(),
            nn.Dropout(PARAMS["dropout"]),
            nn.Linear(PARAMS["hidden_size_3"], PARAMS["hidden_size_4"]),
            nn.BatchNorm1d(PARAMS["hidden_size_4"]),
            nn.ReLU(),
            nn.Linear(PARAMS["hidden_size_4"], 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.network(x)
        # Normalização L2 na última camada: garante que o vetor de saída seja unitário
        return out / (torch.norm(out, p=2, dim=1, keepdim=True) + 1e-8)


# --- 3. Helper Functions ---
def get_angular_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Calculates angular error in degrees between unit vectors."""
    # Como as saídas já estão normalizadas, o dot product é a própria similaridade de cosseno
    cos_theta = torch.sum(pred * target, dim=1)
    cos_theta = torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)
    return torch.rad2deg(torch.acos(cos_theta))


def prepare_data(parquet_path: str, batch_size: int):
    df = pd.read_parquet(parquet_path)

    # Filtro de colunas: removemos os alvos e colunas de debug (base_euler)
    target_cols = ["target_x", "target_y", "target_z"]
    meta_cols = ["source_bag", "base_roll", "base_pitch", "base_yaw"]
    feature_cols = [
        c for c in df.columns if c not in target_cols and c not in meta_cols
    ]
    split_idx1 = int(0.7 * len(df))
    split_idx2 = int(0.85 * len(df))
    df_train = df.iloc[:split_idx1]
    df_val = df.iloc[split_idx1:split_idx2]
    df_test = df.iloc[split_idx2:]
    X_train = df_train[feature_cols].values
    y_train = df_train[target_cols].values
    X_val = df_val[feature_cols].values
    y_val = df_val[target_cols].values
    X_test = df_test[feature_cols].values
    y_test = df_test[target_cols].values

    # X = df[feature_cols].values
    # y = df[target_cols].values
    #
    # X_train, X_temp, y_train, y_temp = train_test_split(
    #     X, y, test_size=0.3, random_state=42
    # )
    # X_val, X_test, y_val, y_test = train_test_split(
    #     X_temp, y_temp, test_size=0.5, random_state=42
    # )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val, X_test = scaler.transform(X_val), scaler.transform(X_test)

    # Salvar scaler para uso futuro em tempo real
    joblib.dump(scaler, "scaler.pkl")
    raise ValueError
    train_loader = DataLoader(
        TactileDataset(X_train, y_train), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(TactileDataset(X_val, y_val), batch_size=batch_size)
    test_loader = DataLoader(TactileDataset(X_test, y_test), batch_size=batch_size)

    return train_loader, val_loader, test_loader, len(feature_cols)


# --- 4. Training Loop ---
def train_model(model, train_loader, val_loader, test_loader, device):
    # CosineEmbeddingLoss minimiza 1 - cos(pred, target)
    criterion = nn.CosineEmbeddingLoss()
    optimizer = optim.Adam(model.parameters(), lr=PARAMS["learning_rate"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=PARAMS["scheduler_patience"]
    )

    # 'ones' é necessário para a CosineEmbeddingLoss saber que queremos os vetores IGUAIS
    # (similaridade = 1)
    target_similarity = torch.ones(PARAMS["batch_size"]).to(device)

    for epoch in range(PARAMS["epochs"]):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)

            # Ajuste de tamanho para o último batch caso seja menor que o batch_size
            curr_batch_size = inputs.size(0)
            flag = torch.ones(curr_batch_size).to(device)

            loss = criterion(outputs, targets, flag)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        v_loss, angular_err = 0.0, 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)

                curr_batch_size = inputs.size(0)
                flag = torch.ones(curr_batch_size).to(device)

                loss = criterion(outputs, targets, flag)
                v_loss += loss.item()

                err_deg = get_angular_error(outputs, targets)
                angular_err += err_deg.mean().item()

        avg_loss = v_loss / len(val_loader)
        avg_ang_err = angular_err / len(val_loader)
        scheduler.step(avg_loss)

        mlflow.log_metrics(
            {
                "val_loss": avg_loss,
                "avg_angular_error_deg": avg_ang_err,
            },
            step=epoch,
        )

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d} | Loss: {avg_loss:.6f} | Ang. Error: {avg_ang_err:.2f}°"
            )
        # Validation
    model.eval()
    v_loss, angular_err = 0.0, 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            curr_batch_size = inputs.size(0)
            flag = torch.ones(curr_batch_size).to(device)

            loss = criterion(outputs, targets, flag)
            v_loss += loss.item()

            err_deg = get_angular_error(outputs, targets)
            angular_err += err_deg.mean().item()

    avg_loss = v_loss / len(val_loader)
    avg_ang_err = angular_err / len(val_loader)
    scheduler.step(avg_loss)

    mlflow.log_metrics(
        {
            "test_loss": avg_loss,
            "test_avg_angular_error_deg": avg_ang_err,
        },
    )


if __name__ == "__main__":
    filepath = pjoin("data", "train", "final_training_data_8quads.parquet")
    mlflow.set_experiment("Tactile_Vector_Cosine_Loss")
    dev = PARAMS["device"]

    train_ld, val_ld, test_ld, in_dim = prepare_data(filepath, PARAMS["batch_size"])
    model = TactileMLP(in_dim).to(dev)

    with mlflow.start_run():
        mlflow.log_params(PARAMS)
        train_model(model, train_ld, val_ld, test_ld, dev)
        mlflow.pytorch.log_model(model, "model")
