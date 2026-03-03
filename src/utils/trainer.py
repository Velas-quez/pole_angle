from __future__ import annotations
import copy
import signal
from types import FrameType
from dataclasses import dataclass
from typing import Dict, Any, Callable

import torch
import torch.nn as nn
import mlflow
import hydra
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from src.models.base_model import BaseTactileModel
from src.tactile_types import TactileInput
from src.utils.metrics import VariancePenalizedCosineLoss

# --- TIPAGEM DA CONFIGURAÇÃO (Refletindo o teu config.yaml) ---

CRITERION_WITH_LABELS = (nn.CosineEmbeddingLoss, VariancePenalizedCosineLoss)


@dataclass
class OptimizerConfig:
    _target_: str
    lr: float
    weight_decay: float


@dataclass
class SchedulerConfig:
    _target_: str
    mode: str
    factor: float
    patience: float
    verbose: bool


@dataclass
class TrainingConfig:
    batch_size: int
    epochs: int
    use_gpu: bool
    scale_evtac: bool
    optimizer: OptimizerConfig
    criterion: Dict[str, Any]  # _target_ da Loss
    scheduler: SchedulerConfig


class FullTrainConfig(DictConfig):
    """Extensão para ajudar o linter, embora o Hydra use DictConfig."""

    training: TrainingConfig
    experiment_name: str


stop_training = False


def run_training_loop(
    model: BaseTactileModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: FullTrainConfig,
    device: torch.device,
):
    # 1. Instanciar componentes via Hydra
    criterion = hydra.utils.instantiate(cfg.training.criterion)
    loss_args_fn = get_loss_args_fn(criterion)
    optimizer = hydra.utils.instantiate(
        cfg.training.optimizer, params=model.parameters()
    )
    # Função que converte o output (Euler ou Vetor) para Vetor Unitário 3D
    convert_fn = hydra.utils.instantiate(cfg.targets.convert_to_vector_fn)
    angle_metric_fn_cfg = cfg.targets.get("angle_metric_fn", None)
    angle_metric_fn = (
        hydra.utils.instantiate(angle_metric_fn_cfg)
        if angle_metric_fn_cfg is not None
        else None
    )

    scheduler = hydra.utils.instantiate(cfg.training.scheduler, optimizer=optimizer)

    best_val_loss = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())

    global stop_training

    signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl+C

    for epoch in range(cfg.training.epochs):
        if stop_training:
            break
        print(f"\nStarting epoch {epoch + 1}/{cfg.training.epochs}...", flush=True)
        # --- PHASE: TRAINING ---
        model.train()
        train_running_loss = 0.0
        train_running_angle = 0.0
        train_seen = 0

        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            inputs = TactileInput(**{k: v.to(device) for k, v in batch_x.items()})
            targets = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            # Cálculo da Loss de Otimização
            args = loss_args_fn(outputs, targets)
            loss = criterion(*args)

            loss.backward()
            optimizer.step()

            # Cálculo da Métrica Angular (sempre em graus)
            with torch.no_grad():
                if angle_metric_fn is not None:
                    angle_err = angle_metric_fn(outputs, targets)
                else:
                    out_vecs = convert_fn(outputs)
                    tgt_vecs = convert_fn(targets)
                    angle_err = angle_diff_deg(out_vecs, tgt_vecs)

                train_running_loss += loss.item() * targets.size(0)
                train_running_angle += angle_err.mean().item() * targets.size(0)
                train_seen += targets.size(0)

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                train_loss_so_far = train_running_loss / max(train_seen, 1)
                train_angle_so_far = train_running_angle / max(train_seen, 1)
                print(
                    (
                        f"\rEpoch {epoch + 1}/{cfg.training.epochs} | "
                        f"Train {batch_idx + 1}/{len(train_loader)} | "
                        f"tr_loss={train_loss_so_far:.4f} | tr_angle={train_angle_so_far:.2f}° | "
                        f"val_loss=... | val_angle=..."
                    ),
                    end="",
                    flush=True,
                )

        epoch_train_loss = train_running_loss / len(train_loader.dataset)
        epoch_train_angle = train_running_angle / len(train_loader.dataset)

        # --- PHASE: VALIDATION ---
        model.eval()
        val_running_loss = 0.0
        val_errors = []
        val_seen = 0

        with torch.no_grad():
            for batch_idx, (batch_x, batch_y) in enumerate(val_loader):
                inputs: TactileInput = {k: v.to(device) for k, v in batch_x.items()}
                targets = batch_y.to(device)

                outputs = model(inputs)

                # Loss bruta
                args = loss_args_fn(outputs, targets)
                loss = criterion(*args)

                # Ângulo real
                if angle_metric_fn is not None:
                    angle_err = angle_metric_fn(outputs, targets)
                else:
                    out_vecs = convert_fn(outputs)
                    tgt_vecs = convert_fn(targets)
                    angle_err = angle_diff_deg(out_vecs, tgt_vecs)

                val_running_loss += loss.item() * targets.size(0)
                val_errors.append(angle_err)
                val_seen += targets.size(0)

                if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(val_loader):
                    val_errors_so_far = torch.cat(val_errors)
                    val_loss_so_far = val_running_loss / max(val_seen, 1)
                    val_angle_so_far = val_errors_so_far.mean().item()
                    print(
                        (
                            f"\rEpoch {epoch + 1}/{cfg.training.epochs} | "
                            f"Val {batch_idx + 1}/{len(val_loader)} | "
                            f"tr_loss={epoch_train_loss:.4f} | tr_angle={epoch_train_angle:.2f}° | "
                            f"val_loss={val_loss_so_far:.4f} | val_angle={val_angle_so_far:.2f}°"
                        ),
                        end="",
                        flush=True,
                    )

        val_errors = torch.cat(val_errors)
        epoch_val_loss = val_running_loss / len(val_loader.dataset)
        epoch_val_angle_mean = val_errors.mean().item()
        epoch_val_angle_std = val_errors.std().item()
        print(
            (
                f"\rEpoch {epoch + 1}/{cfg.training.epochs} | Done | "
                f"tr_loss={epoch_train_loss:.4f} | tr_angle={epoch_train_angle:.2f}° | "
                f"val_loss={epoch_val_loss:.4f} | val_angle={epoch_val_angle_mean:.2f}°"
            ),
            flush=True,
        )

        # 3. Scheduler & Logging
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(epoch_val_loss)
        else:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        # MLflow Tracking: Loss (Otimização) e Angle (Física)
        mlflow.log_metric("train_loss", epoch_train_loss, step=epoch)
        mlflow.log_metric("train_angle_deg", epoch_train_angle, step=epoch)
        mlflow.log_metric("val_loss", epoch_val_loss, step=epoch)
        mlflow.log_metric("val_angle_deg", epoch_val_angle_mean, step=epoch)
        mlflow.log_metric("val_angle_deg_std", epoch_val_angle_std, step=epoch)
        mlflow.log_metric("lr", current_lr, step=epoch)

        # Checkpoint baseado na loss de validação
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 1 == 0:
            print(
                f"Epoch {epoch + 1} | Val Loss: {epoch_val_loss:.4f} | "
                f"Val Angle Err: {epoch_val_angle_mean:.2f}° (±{epoch_val_angle_std:.2f}°) | LR: {current_lr}"
            )

    model.load_state_dict(best_model_wts)
    return model


# --- FUNÇÕES AUXILIARES ---


def get_loss_args_fn(criterion: nn.Module) -> Callable:
    if isinstance(criterion, CRITERION_WITH_LABELS):
        return _loss_args_with_labels
    return _loss_args_without_labels


def _loss_args_with_labels(outputs, targets):
    labels = torch.ones(targets.size(0), device=outputs.device)
    return [outputs, targets, labels]


def _loss_args_without_labels(outputs, targets):
    return [outputs, targets]


def angle_diff_deg(vec1: torch.Tensor, vec2: torch.Tensor) -> torch.Tensor:
    # Garante normalização unitária para evitar erros no acos
    v1 = torch.nn.functional.normalize(vec1, p=2, dim=1)
    v2 = torch.nn.functional.normalize(vec2, p=2, dim=1)

    cos_sim = torch.sum(v1 * v2, dim=1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cos_sim))


def signal_handler(signum: int, frame: FrameType | None) -> None:
    global stop_training
    print("\nReceived stop signal. Saving model and exiting training early...")
    stop_training = True
