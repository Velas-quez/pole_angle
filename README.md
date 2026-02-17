# Predict Pole Angle with Tactile Data


📊 **Data Loading & Normalization Pipeline**

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


⚙️ **Training Engine** (`trainer.py`)


- **Optimizer & Scheduling**  
  Adam optimizer + ReduceLROnPlateau (reduces LR by `scheduler_factor` after `scheduler_patience` epochs of no val loss improvement).

- **Best Model Checkpointing**  
  Tracks validation loss → saves deep copy of weights at lowest val loss → restores best weights at the end (prevents returning overfitted final model).

- **Experiment Tracking**  
  Logs `train_loss`, `val_loss`, `learning_rate` to MLflow every epoch.  
  Prints clean progress summary every 10 epochs.

- **Hardware Handling**  
  Automatically moves model and data to GPU (CUDA) or CPU based on environment.

