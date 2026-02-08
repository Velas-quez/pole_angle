import joblib
import torch
import pandas as pd
import numpy as np
import mlflow
import mlflow.pytorch

# --- 1. MANDATORY: Force the backend BEFORE pyplot is imported ---
import matplotlib

try:
    matplotlib.use("TkAgg")  # Standard for Pop!_OS
except:
    matplotlib.use("Qt5Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Tuple


# --- 1. Angle Reconstruction ---
def get_angles_from_output(output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reconstructs Roll [0,360) and Pitch [0,180] from [sin_r, cos_r, cos_p]."""
    sin_r, cos_r, cos_p = output[:, 0], output[:, 1], output[:, 2]
    roll_deg = torch.rad2deg(torch.atan2(sin_r, cos_r)) % 360
    pitch_deg = torch.rad2deg(torch.acos(torch.clamp(cos_p, -1.0, 1.0)))
    return roll_deg, pitch_deg


# --- 2. Coordinate Mapping ---
def get_pole_endpoint(
    roll_deg: float, pitch_deg: float, length: float = 1.0
) -> np.ndarray:
    """Converts spherical angles to Cartesian (Z-up)."""
    r_rad, p_rad = np.radians(roll_deg), np.radians(pitch_deg)
    x = length * np.sin(p_rad) * np.cos(r_rad)
    y = length * np.sin(p_rad) * np.sin(r_rad)
    z = length * np.cos(p_rad)
    return np.array([x, y, z])


# --- 3. Main Visualization Function ---
def run_pole_visualization(run_id: str, parquet_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # MLflow points to your local mlruns folder by default
    model_uri = f"runs:/{run_id}/model"
    print(f"Loading local artifacts for Run: {run_id}")

    model = mlflow.pytorch.load_model(model_uri).to(device)
    model.eval()

    # Fetch the scaler specifically associated with this Run ID
    # scaler_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="scaler.pkl")
    scaler_path = "preprocessing/scaler.pkl"
    scaler = joblib.load(scaler_path)

    # Prepare Data
    df = pd.read_parquet(parquet_path)
    feature_cols = [c for c in df.columns if "target" not in c]
    X_scaled = scaler.transform(df[feature_cols].values)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        raw_outputs = model(X_tensor)
        pred_rolls, pred_pitches = get_angles_from_output(raw_outputs)

    pred_rolls, pred_pitches = pred_rolls.cpu().numpy(), pred_pitches.cpu().numpy()
    true_rolls, true_pitches = df["target_roll"].values, df["target_pitch"].values

    # Setup Plot with 3 subplots (Real, Pred, Error)
    fig = plt.figure(figsize=(15, 8))
    ax_real = fig.add_subplot(121, projection="3d")
    ax_pred = fig.add_subplot(122, projection="3d")

    # Adding a small text area for real-time error metrics
    error_text = fig.text(
        0.5, 0.05, "", ha="center", fontsize=12, bbox=dict(facecolor="white", alpha=0.5)
    )

    def update(frame):
        ax_real.clear()
        ax_pred.clear()

        for ax, title in zip(
            [ax_real, ax_pred], ["OptiTrack (Ground Truth)", "MLP Prediction"]
        ):
            ax.set_xlim([-0.7, 0.7])
            ax.set_ylim([-0.7, 0.7])
            ax.set_zlim([0, 1.1])
            ax.set_title(title)

        # Endpoints
        p_real = get_pole_endpoint(true_rolls[frame], true_pitches[frame])
        p_pred = get_pole_endpoint(pred_rolls[frame], pred_pitches[frame])

        # Plot Poles
        ax_real.plot(
            [0, p_real[0]], [0, p_real[1]], [0, p_real[2]], color="green", linewidth=4
        )
        ax_pred.plot(
            [0, p_pred[0]], [0, p_pred[1]], [0, p_pred[2]], color="red", linewidth=4
        )

        # Calculate Error
        r_err = abs(true_rolls[frame] - pred_rolls[frame])
        r_err = min(r_err, 360 - r_err)  # Circular error
        p_err = abs(true_pitches[frame] - pred_pitches[frame])

        error_text.set_text(
            f"FRAME: {frame} | Roll Error: {r_err:.2f}° | Pitch Error: {p_err:.2f}°"
        )

    ani = FuncAnimation(fig, update, frames=range(0, len(df), 2), interval=30)
    plt.show()


if __name__ == "__main__":
    # Use the Run ID folder name from your local mlruns directory
    RUN_ID = "884dcf460bd44a50959cefcaff22fe73"
    run_pole_visualization(RUN_ID, "data/final_training_data.parquet")
