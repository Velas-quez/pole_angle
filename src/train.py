import os
import hydra
import joblib
import mlflow
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

# Modular imports follow the Strategy and Factory patterns
from models.mlp import TactileMLP
from processing.hu_moments_strategy import HuMomentsStrategy
from utils.dataloader import create_loaders 

# Hydra Decorator: manages configuration files in the /conf folder
@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg: DictConfig):
    # Setup MLflow local tracking
    mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))
    mlflow.set_experiment(cfg.experiment_name)
    
    # Device management (GPU/CPU)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.use_gpu else "cpu")

    # --- RESUME LOGIC ---
    # Check if we are continuing an existing experiment or starting fresh
    resume_id = cfg.get("resume_run_id", None)
    
    # Start MLflow session. If run_id is provided, it enters "Update" mode.
    with mlflow.start_run(run_id=resume_id, run_name=cfg.model.name) as run:
        if resume_id:
            print(f"🔄 Resuming training for Run ID: {resume_id}")
            
            # 1. Load the original Scaler used in the previous session
            # This is critical to ensure data distribution remains consistent
            scaler_path = mlflow.get_artifact_uri("scaler.pkl").replace("file://", "")
            scaler = joblib.load(scaler_path)
            
            # 2. Load the previous model weights and architecture
            model_uri = f"runs:/{resume_id}/model"
            model = mlflow.pytorch.load_model(model_uri).to(device)
            
            # Extract input dimension from the first layer of the loaded network
            input_dim = model.net[0].in_features
        else:
            print(f"🚀 Starting NEW Training: {cfg.model.name}")
            scaler = None # A new scaler will be fitted during data loading
            model = None

        # --- PROCESSING STRATEGY ---
        # Apply the Hu Moments extraction logic to the raw Parquet data
        strategy = HuMomentsStrategy()
        df_train = strategy.transform(
            evtac_path=cfg.data.evtac, 
            markers_path=cfg.data.markers, 
            opti_path=cfg.data.opti
        )

        # 3. Data Preparation
        # create_loaders handles splitting and normalization. 
        # We pass the existing_scaler to maintain consistency if resuming.
        train_loader, val_loader, test_loader, scaler, input_dim_data = create_loaders(
            df_train, 
            batch_size=cfg.training.batch_size,
            existing_scaler=scaler 
        )

        # 4. Model Initialization
        if model is None:
            # First time training: initialize weights and log the new scaler
            model = TactileMLP(
                input_dim=input_dim_data, 
                hidden_layers=cfg.model.layers,
                dropout=cfg.model.dropout
            ).to(device)
            
            # Save scaler locally then log to MLflow as an artifact
            joblib.dump(scaler, "scaler.pkl")
            mlflow.log_artifact("scaler.pkl")

        # 5. Execution: The Training Loop
        # This function should handle loss calculation, backprop, and validation
        # run_training_loop is imported from your utils/trainer.py
        best_model = run_training_loop(model, train_loader, val_loader, cfg, device)

        # 6. Save Progress
        # Persist the best version of the model back to MLflow
        mlflow.pytorch.log_model(best_model, "model")
        
        print(f"✅ Process finished. Run ID: {run.info.run_id}")

if __name__ == "__main__":
    train()