import os
import sys
from pathlib import Path
import torch
import hydra
import mlflow
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from omegaconf import DictConfig

# Add the root directory to path to import src
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from src.processing.strategies.base_strategy import BaseStrategy

@hydra.main(version_base=None, config_path="config", config_name="config")
def test(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.use_gpu else "cpu")
    run_id = cfg.test.run_id

    # 1. Instantiate Strategy
    strategy: BaseStrategy = hydra.utils.instantiate(cfg.strategies)
    
    # 2. Process Test Data (if necessary)
    test_processed_dir = Path(cfg.data.processed_dir.test)
    strategy_test_dir = test_processed_dir / strategy.get_dirname()

    if not strategy_test_dir.exists():
        print(f"⚙️ Processing test scenario with: {strategy.name}")
        strategy.transform(
            evtac_path=cfg.data.pre_processed.test.evtac,
            markers_path=cfg.data.pre_processed.test.markers,
            opti_path=cfg.data.pre_processed.test.opti,
            output_base_dir=str(test_processed_dir),
        )

    # 3. Load Processed Data
    data_dict = strategy.load_processed_data(
        base_path=str(test_processed_dir), 
        target_cols=list(cfg.targets.columns)
    )

    # 4. Load MLflow Artifacts
    print(f"📦 Loading artifacts from run: {run_id}")
    mlflow.set_tracking_uri(Path("mlruns").resolve().as_uri())
    
    model = mlflow.pytorch.load_model(f"runs:/{run_id}/model").to(device)
    model.eval()

    uri_evetac = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="scaler_evetac.pkl")
    uri_base = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="scaler_base.pkl")
    scaler_evetac = joblib.load(uri_evetac)
    scaler_base = joblib.load(uri_base)

    # 5. Scaling and Inference
    try:
        evetac_scaled = scaler_evetac.transform(data_dict["evetac"])
        base_scaled = scaler_base.transform(data_dict["base"])
    except ValueError:
        print("⚠️ Dimension mismatch. Reshaping for scaler...")
        expected_feats = scaler_evetac.n_features_in_
        samples, total_feats = data_dict["evetac"].shape
        evetac_scaled = scaler_evetac.transform(data_dict["evetac"].reshape(-1, expected_feats)).reshape(samples, total_feats)
        
        expected_base_feats = scaler_base.n_features_in_
        base_samples, total_base_feats = data_dict["base"].shape
        base_scaled = scaler_base.transform(data_dict["base"].reshape(-1, expected_base_feats)).reshape(base_samples, total_base_feats)

    X_evetac = torch.tensor(evetac_scaled, dtype=torch.float32).to(device)
    X_base = torch.tensor(base_scaled, dtype=torch.float32).to(device)
    y_true = data_dict["target"]

    with torch.no_grad():
        model_input = {"evetac": X_evetac, "base": X_base}
        y_pred = model(model_input).cpu().numpy()

    # 6. Metrics
    dot_product = np.sum(y_pred * y_true, axis=1)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    errors_deg = np.rad2deg(np.arccos(dot_product))

    # 7. Plots and Animation
    output_path = Path(cfg.test.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    generate_static_plots(errors_deg, str(output_path))
    animate_pole_merged(y_true, y_pred, str(output_path))

def generate_static_plots(errors, output_dir):
    time_steps = np.arange(len(errors))
    plt.figure(figsize=(15, 5))

    # Subplot 1: Angular Error over time
    plt.subplot(1, 2, 1)
    plt.plot(time_steps, errors, color="purple", label="Error")
    plt.axhline(y=np.mean(errors), color='gray', linestyle='--', label=f'Mean: {np.mean(errors):.2f}°')
    plt.title("Angular Error over Time")
    plt.xlabel("Step"); plt.ylabel("Degrees (°)"); plt.legend()

    # Subplot 2: Error Distribution
    plt.subplot(1, 2, 2)
    plt.hist(errors, bins=50, color='green', edgecolor='black', alpha=0.7)
    plt.title("Error Distribution")
    plt.xlabel("Degrees (°)"); plt.ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/metrics_dashboard.png")
    print(f"📊 Metrics saved in: {output_dir}")

def animate_pole_merged(y_true, y_pred, output_dir):
    # Setup for 30s GIF @ 20fps (600 frames)
    fps = 20
    target_frames = 600
    step = max(1, len(y_true) // target_frames)
    
    y_true_ds = y_true[::step][:target_frames]
    y_pred_ds = y_pred[::step][:target_frames]
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Pole Tracking: Real (Blue) vs Predicted (Red)")

    # Merged lines in one plot
    line_true, = ax.plot([], [], [], color='blue', linewidth=4, label='Real', alpha=0.6)
    line_pred, = ax.plot([], [], [], color='red', linewidth=3, label='Predicted', linestyle='--')

    ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([0, 1])
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend(loc='upper right')
    ax.view_init(elev=20, azim=45)

    def init():
        line_true.set_data([], []); line_true.set_3d_properties([])
        line_pred.set_data([], []); line_pred.set_3d_properties([])
        return line_true, line_pred

    def update(num):
        line_true.set_data([0, y_true_ds[num, 0]], [0, y_true_ds[num, 1]])
        line_true.set_3d_properties([0, y_true_ds[num, 2]])
        line_pred.set_data([0, y_pred_ds[num, 0]], [0, y_pred_ds[num, 1]])
        line_pred.set_3d_properties([0, y_pred_ds[num, 2]])
        return line_true, line_pred

    ani = animation.FuncAnimation(fig, update, frames=len(y_true_ds), 
                                  init_func=init, blit=True, interval=1000/fps)

    save_path = f"{output_dir}/pole_merged_comparison.gif"
    ani.save(save_path, writer='pillow', fps=fps)
    print(f"✨ Animation saved to: {save_path}")
    plt.close()

if __name__ == "__main__":
    test()