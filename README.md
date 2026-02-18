# Predict Pole Angle with Tactile Data


## 📊 **Data Loading & Normalization Pipeline**

`utils/dataloader.py` handles the final transition from processed Parquet data to PyTorch-ready tensors. 

🔑 **Key Features**

1. **Automatic Column Identification**  
   The loader uses a keyword-based approach to separate inputs from outputs:

   - **Targets**: Any column containing the string `target` (e.g., `target_roll`, `target_pitch`) is identified as an output label ($y$).
   - **Features**: All other columns are treated as inputs ($X$)

2. **Sequential 80/20 Split**  
   To respect the temporal nature of robotic pole balancing, we do **not** use a random split:

   - **Training (80%)**: The first 80% of the recording. This data is shuffled internally by the DataLoader during training to improve stochastic gradient descent.
   - **Validation (20%)**: The final 20% of the data. This is kept as a continuous, non-shuffled block. This provides a much more honest assessment of how the model will perform in a real-time "future" sequence.
   - **Testing**: There is no test split in this loader. Testing is performed in an isolated run using dedicated "unseen" bag files recorded separately.

3. **Data Scaling (StandardScaler)**  
   Before entering the neural network, the tactile data is normalized using the `StandardScaler`.

   - **What it does**: It shifts the data distribution so the Mean is 0 and the Standard Deviation is 1.
   - **Why it's needed**: Tactile features (like Hu Moments) have vastly different numerical ranges. Scaling ensures that no single feature dominates the model's weight updates simply because of its magnitude, leading to faster and more stable convergence.

4. **Scaler Integrity & Resuming**  
   The scaler is treated as a "learned" component of the pipeline:

   - **Train-Only Fitting**: The scaler calculates its parameters ($\mu$ and $\sigma$) only from the Training set. This prevents "Data Leakage," where information from the future (validation set) might influence the training phase.
   - **Persistence**: The scaler is saved as `scaler.pkl` and logged as an MLflow artifact.
   - **Resume-Ready**: When resuming a training run, the code reloads the original scaler. This ensures the model always sees data on the same scale it was originally trained on, preventing performance collapse.

<!-- 🚀 **Usage**

To integrate the loader into your training pipeline:

```python
from utils.dataloader import create_loaders

# Just pass your dataframe!
train_loader, val_loader, _, scaler, input_dim = create_loaders(
    df=my_dataframe, 
    batch_size=cfg.training.batch_size,
    existing_scaler=None  # Pass a loaded scaler here if resuming a Run
) -->


## ⚙️ **Training Engine** (`trainer.py`)


- **Optimizer & Scheduling**  
  Adam optimizer + ReduceLROnPlateau (reduces LR by `scheduler_factor` after `scheduler_patience` epochs of no val loss improvement).

- **Best Model Checkpointing**  
  Tracks validation loss → saves deep copy of weights at lowest val loss → restores best weights at the end (prevents returning overfitted final model).

- **Experiment Tracking**  
  Logs `train_loss`, `val_loss`, `learning_rate` to MLflow every epoch.  
  Prints clean progress summary every 10 epochs.

- **Hardware Handling**  
  Automatically moves model and data to GPU (CUDA) or CPU based on environment.

## 🛠️ **Configuration (Hydra + YAML)**

Uses **Hydra** for clean, code-free hyperparameter and experiment management.

**Structure**

- `conf/config.yaml`  
  Main file – sets defaults and global settings

- `conf/models/`  
  One file per model architecture (e.g. `feat_mlp.yaml`, `multi_quads.yaml`)

**How It Works**

1. **Main config.yaml** (orchestrator)

```yaml
defaults:
  - _self_
  - models: feat_mlp           # ← selects model sub-config

experiment_name: "Tactile_Balance_v1"

training:
  batch_size: 128
  lr: 0.001
  epochs: 200
  scheduler_patience: 5
  scheduler_factor: 0.5
```

2. **Model sub-config** (example: `feat_mlp.yaml`)

```yaml
name: "Tactile MLP"
name_id: "feat_mlp"           # must match models/feat_mlp.py
layers: [128, 64, 32]
dropout: 0.1
```

**Quick Commands**

Run default settings  
```bash
python train.py
```

Switch model  
```bash
python train.py models=multi_quads
```

Override any value  
```bash
python train.py training.lr=0.0005 training.batch_size=64 model.dropout=0.3
```

Resume existing MLflow run  
```bash
python train.py resume_run_id="abc123..."
```

**Best Practices**

- Keep model-specific settings only in `conf/models/*.yaml`
- Keep training / paths / globals in `config.yaml`
- Make sure `name_id` matches the model filename


## 🏗️ **Experiment Orchestrator** (`train.py`)

Central script that coordinates the full training workflow.

**Core Responsibilities**

- Dynamically loads model and processing strategy using `name_id` from Hydra config  

- Manages local MLflow tracking (`file://./mlruns`)  
  Groups runs under `experiment_name` from YAML

- Supports resuming runs via `resume_run_id`:  
  • Restores best model weights  
  • Reloads exact same StandardScaler (prevents distribution shift)

- Auto-detects input/output dimensions from processed data:  
  • `input_dim` ← feature columns  
  • `output_dim` ← count of `target*` columns

**Execution Flow (simplified)**

1. Hydra config → merged settings  
2. Start MLflow run  
3. Load & process Parquet → clean DataFrame  
4. Create train/val loaders (80/20 sequential split + scaling)  
5. Build new model or load checkpoint  
6. Run training via `trainer.py`  
7. Save scaler + best weights as MLflow artifacts

**Quick tip**  
Run `mlflow ui` in terminal to visualize all runs and loss curves.

```bash
mlflow ui
```
