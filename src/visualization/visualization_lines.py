import pandas as pd
import torch
import mlflow.pytorch
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def get_angles_from_output(output: torch.Tensor):
    sin_r, cos_r, cos_p = output[:, 0], output[:, 1], output[:, 2]
    roll_deg = torch.rad2deg(torch.atan2(sin_r, cos_r)) % 360
    pitch_deg = torch.rad2deg(torch.acos(torch.clamp(cos_p, -1.0, 1.0)))
    return roll_deg, pitch_deg


def plot_interactive_results(run_id: str, parquet_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load Model & Scaler from local MLflow
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.pytorch.load_model(model_uri).to(device)
    model.eval()

    # scaler_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="scaler.pkl")
    scaler_path = "preprocessing/scaler.pkl"
    scaler = joblib.load(scaler_path)

    # 2. Process Data
    df = pd.read_parquet(parquet_path)
    feature_cols = [c for c in df.columns if "target" not in c]
    X_scaled = scaler.transform(df[feature_cols].values)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        raw_outputs = model(X_tensor)
        p_roll, p_pitch = get_angles_from_output(raw_outputs)

    # 3. Create Plotly Figure
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(f"Roll Tracking - Run {run_id[:8]}", "Pitch Tracking"),
    )

    # Add Roll traces
    fig.add_trace(
        go.Scatter(y=df["target_roll"], name="Actual Roll", line=dict(color="#2ecc71")),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            y=p_roll.cpu().numpy(),
            name="Pred Roll",
            line=dict(color="#e74c3c", dash="dash"),
        ),
        row=1,
        col=1,
    )

    # Add Pitch traces
    fig.add_trace(
        go.Scatter(
            y=df["target_pitch"], name="Actual Pitch", line=dict(color="#3498db")
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            y=p_pitch.cpu().numpy(),
            name="Pred Pitch",
            line=dict(color="#f39c12", dash="dash"),
        ),
        row=2,
        col=1,
    )

    # 4. Interactive Layout Features
    fig.update_layout(
        height=900,
        template="plotly_dark",
        title_text="Tactile Sensor: Actual vs Predicted Orientation",
        hovermode="x unified",  # Shows both values when hovering
    )

    # Enable Range Selector (Slider) to fix the "squeezed" feeling
    fig.update_xaxes(rangeslider_visible=True, row=2, col=1)

    # Show in Browser
    fig.show()


if __name__ == "__main__":
    RUN_ID = "884dcf460bd44a50959cefcaff22fe73"
    plot_interactive_results(RUN_ID, "data/test/processed/final_test_data.parquet")
