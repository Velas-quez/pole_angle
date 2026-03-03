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

    # 4. Load MLflow Artifacts (Model and Training Scalers)
    print(f"📦 Loading model and scalers from run: {run_id}")
    mlflow.set_tracking_uri(Path("mlruns").resolve().as_uri())
    
    # Load model
    model = mlflow.pytorch.load_model(f"runs:/{run_id}/model").to(device)
    model.eval()

    # Load Scalers used during TRAINING
    uri_evetac = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="scaler_evetac.pkl")
    uri_base = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="scaler_base.pkl")
    scaler_evetac = joblib.load(uri_evetac)
    scaler_base = joblib.load(uri_base)

    # 5. Prepare Input and Inference
    # Handle dimension mismatch if the scaler expects individual frames vs concatenated windows
    try:
        evetac_scaled = scaler_evetac.transform(data_dict["evetac"])
        base_scaled = scaler_base.transform(data_dict["base"])
    except ValueError:
        print("⚠️ Dimension mismatch detected. Reshaping data for the scaler...")
        # Get expected features from scaler (e.g., 320)
        expected_feats = scaler_evetac.n_features_in_
        
        # Flatten and scale evetac
        samples, total_feats = data_dict["evetac"].shape
        evetac_reshaped = data_dict["evetac"].reshape(-1, expected_feats)
        evetac_scaled = scaler_evetac.transform(evetac_reshaped).reshape(samples, total_feats)
        
        # Flatten and scale base
        expected_base_feats = scaler_base.n_features_in_
        base_samples, total_base_feats = data_dict["base"].shape
        base_reshaped = data_dict["base"].reshape(-1, expected_base_feats)
        base_scaled = scaler_base.transform(base_reshaped).reshape(base_samples, total_base_feats)

    X_evetac = torch.tensor(evetac_scaled, dtype=torch.float32).to(device)
    X_base = torch.tensor(base_scaled, dtype=torch.float32).to(device)
    y_true = data_dict["target"] # already numpy

    with torch.no_grad():
        tactile_in = {"evetac": X_evetac, "base": X_base}
        y_pred = model(tactile_in).cpu().numpy()

    # 6. Calculate Angular Error (in degrees)
    # Error = arccos(dot_product(pred, true))
    dot_product = np.sum(y_pred * y_true, axis=1)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    errors_deg = np.rad2deg(np.arccos(dot_product))

    # 7. Visualization
    generate_plots(y_true, y_pred, errors_deg, cfg.test.output_dir)

def generate_plots(y_true, y_pred, errors, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    time_steps = np.arange(len(errors))

    # Graph 1: Prediction vs Ground Truth (X Component example)
    plt.figure(figsize=(12, 5))
    plt.plot(time_steps, y_true[:, 0], label="Ground Truth (X)", color="blue", alpha=0.6)
    plt.plot(time_steps, y_pred[:, 0], label="Predicted (X)", color="red", linestyle="--")
    plt.title("Vector Tracking: Ground Truth vs Predicted (X-Axis)")
    plt.xlabel("Time (steps)")
    plt.ylabel("Normalized Value")
    plt.legend()
    plt.savefig(f"{output_dir}/tracking_comparison.png")

    # Graph 2: Angular Error over time
    plt.figure(figsize=(12, 5))
    plt.plot(time_steps, errors, color="purple", label="Angular Error")
    plt.axhline(y=np.mean(errors), color='gray', linestyle='--', label=f'Mean: {np.mean(errors):.2f}°')
    plt.title("Angular Error During Test Scenario")
    plt.xlabel("Time (steps)")
    plt.ylabel("Error (Degrees °)")
    plt.legend()
    plt.savefig(f"{output_dir}/error_over_time.png")

    # Graph 3: Error Distribution Histogram
    plt.figure(figsize=(8, 6))
    plt.hist(errors, bins=50, color='green', edgecolor='black', alpha=0.7)
    plt.title("Angular Error Distribution")
    plt.xlabel("Error (Degrees °)")
    plt.ylabel("Frequency")
    plt.savefig(f"{output_dir}/error_histogram.png")

    print(f"✨ Test finished. Plots saved in: {output_dir}")
    plt.show()

if __name__ == "__main__":
    test()