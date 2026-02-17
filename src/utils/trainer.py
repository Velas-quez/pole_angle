import torch
import mlflow
from tqdm import tqdm


def run_training_loop(
    model, train_loader, val_loader, criterion, optimizer, scheduler, cfg, device
):
    """
    Motor de treino agnóstico.
    Recebe os componentes de otimização já instanciados pelo Hydra no train.py.
    """
    best_val_loss = float("inf")
    best_model_state = None

    for epoch in range(cfg.training.epochs):
        # --- FASE DE TREINO ---
        model.train()
        train_loss = 0.0
        for inputs, targets in tqdm(
            train_loader, desc=f"Epoch {epoch} [Train]", leave=False
        ):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            # Suporte flexível para CosineEmbeddingLoss ou MSE
            if isinstance(criterion, torch.nn.CosineEmbeddingLoss):
                loss = criterion(
                    outputs, targets, torch.ones(inputs.size(0)).to(device)
                )
            else:
                loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # --- FASE DE VALIDAÇÃO ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)

                if isinstance(criterion, torch.nn.CosineEmbeddingLoss):
                    loss = criterion(
                        outputs, targets, torch.ones(inputs.size(0)).to(device)
                    )
                else:
                    loss = criterion(outputs, targets)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        # --- SCHEDULER & LOGGING ---
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()

        mlflow.log_metrics(
            {
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "lr": optimizer.param_groups[0]["lr"],
            },
            step=epoch,
        )

        # --- CHECKPOINT LOGIC ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Em vez de salvar arquivo, retornamos o melhor estado
            best_model_state = model.state_dict()

    # Carrega os melhores pesos antes de retornar
    if best_model_state:
        model.load_state_dict(best_model_state)

    return model
