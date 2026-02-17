import os
import hydra
import joblib
import mlflow
import torch
import importlib
from omegaconf import DictConfig

# Core utility imports
from utils.dataloader import create_loaders 
from utils.trainer import run_training_loop

def get_dynamic_class(folder, prefix, name_id, class_name):
    """
    Helper to dynamically import classes.
    folder: 'models' or 'processing'
    prefix: '' for models, 'processing_' for strategies
    name_id: the ID from your YAML (e.g., 'feat_mlp')
    class_name: 'Model' or 'Strategy'
    """
    module_path = f"{folder}.{prefix}{name_id}"
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"❌ Error: Could not find '{class_name}' in {module_path}.py. {e}")

@hydra.main(version_base=None, config_path="config", config_name="config")
def train(cfg: DictConfig):
    # Setup MLflow Tracking
    mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))
    mlflow.set_experiment(cfg.experiment_name)
    
    # Use training config for device management
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.training.use_gpu else "cpu")

    # Resume Logic
    resume_id = cfg.get("resume_run_id", None)
    model_name_id = cfg.models.name_id 
    
    with mlflow.start_run(run_id=resume_id, run_name=cfg.models.name) as run:
        if resume_id:
            print(f"🔄 Resuming Run: {resume_id}")
            scaler_path = mlflow.get_artifact_uri("scaler.pkl").replace("file://", "")
            scaler = joblib.load(scaler_path)
            model_uri = f"runs:/{resume_id}/model"
            model = mlflow.pytorch.load_model(model_uri).to(device)
        else:
            print(f"🚀 Starting NEW Training: {cfg.models.name}")
            scaler = None 
            model = None

        # --- DYNAMIC DATA PROCESSING ---
        # Look for processing/processing_{model_name_id}.py
        StrategyClass = get_dynamic_class("processing", "processing_", model_name_id, "Strategy")
        strategy = StrategyClass()
        
        df_processed = strategy.transform(
            evtac_path=cfg.data.evtac, 
            markers_path=cfg.data.markers, 
            opti_path=cfg.data.opti
        )

        # Automatically detect targets and prepare loaders
        train_loader, val_loader, _, scaler, input_dim, output_dim = create_loaders(
            df_processed, 
            batch_size=cfg.training.batch_size,
            existing_scaler=scaler 
        )

        # --- DYNAMIC MODEL INITIALIZATION ---
        if model is None:
            # Look for models/{model_name_id}.py
            ModelClass = get_dynamic_class("models", "", model_name_id, "Model")
            
            model = ModelClass(
                input_dim=input_dim, 
                output_dim=output_dim,
                cfg=cfg.models 
            ).to(device)
            
            # Save artifacts for this new run
            joblib.dump(scaler, "scaler.pkl")
            mlflow.log_artifact("scaler.pkl")

        # --- TRAINING ENGINE ---
        # All training logic is handled in the trainer utility
        best_model = run_training_loop(model, train_loader, val_loader, cfg, device)
        
        # Log the final best model to MLflow
        mlflow.pytorch.log_model(best_model, "model")
        print(f"✅ Process finished. Run ID: {run.info.run_id}")

if __name__ == "__main__":
    train()