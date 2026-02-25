import numpy as np
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from typing import Dict, Optional, Tuple

from src.processing.strategies.base_strategy import BaseStrategy
from src.tactile_types import TactileData


def create_loaders(
    data_dict: TactileData,
    strategy: BaseStrategy,
    batch_size: int = 128,
    existing_scaler_evetac: Optional[StandardScaler] = None,
    existing_scaler_base: Optional[StandardScaler] = None,
    split_ratio: float = 0.8,
) -> Tuple[DataLoader, DataLoader, StandardScaler, StandardScaler]:
    """
    Cria DataLoaders com scalers independentes para EvETAC e Base.
    Retorna Dataloader de treino, Dataloader de validação, scaler para EvETAC e scaler para Base.
    """

    # 1. Split Sequencial
    num_samples = len(data_dict["target"])
    split_idx = int(num_samples * split_ratio)

    train_data = {k: v[:split_idx] for k, v in data_dict.items()}
    val_data = {k: v[split_idx:] for k, v in data_dict.items()}

    # 2. Scaling EvETAC
    scaler_evetac = existing_scaler_evetac or StandardScaler()
    if existing_scaler_evetac is None:
        train_data["evetac"] = scaler_evetac.fit_transform(train_data["evetac"])
    else:
        train_data["evetac"] = scaler_evetac.transform(train_data["evetac"])
    val_data["evetac"] = scaler_evetac.transform(val_data["evetac"])

    # 3. Scaling Base (Novidade)
    scaler_base = existing_scaler_base or StandardScaler()
    if existing_scaler_base is None:
        train_data["base"] = scaler_base.fit_transform(train_data["base"])
    else:
        train_data["base"] = scaler_base.transform(train_data["base"])
    val_data["base"] = scaler_base.transform(val_data["base"])

    # 4. Criar Datasets via Strategy
    train_ds = strategy.get_dataset(train_data)
    val_ds = strategy.get_dataset(val_data)

    # 5. Criar Loaders
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, pin_memory=True
    )

    print(f"✅ Loaders com Dual Scalers prontos.")
    return train_loader, val_loader, scaler_evetac, scaler_base
