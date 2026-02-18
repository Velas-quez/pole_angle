import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import copy


def run_training_loop(model, train_loader, val_loader, cfg, device):
    """Handles the optimization, scheduling, and weight updates."""

    # 1. Optimizer and Loss
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.lr)
    criterion = nn.MSELoss()

    # 2. Scheduler (using your YAML params)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.training.scheduler_factor,
        patience=cfg.training.scheduler_patience,
    )

    best_val_loss = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(cfg.training.epochs):
        # --- TRAINING PHASE ---
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        epoch_val_loss = val_loss / len(val_loader.dataset)

        # 3. Update Scheduler & Track Progress
        scheduler.step(epoch_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # Log to MLflow
        mlflow.log_metric("train_loss", epoch_train_loss, step=epoch)
        mlflow.log_metric("val_loss", epoch_val_loss, step=epoch)
        mlflow.log_metric("lr", current_lr, step=epoch)

        # Checkpoint: Save the best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch}/{cfg.training.epochs} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | LR: {current_lr}"
            )

    # Load best weights before returning
    model.load_state_dict(best_model_wts)
    return model
