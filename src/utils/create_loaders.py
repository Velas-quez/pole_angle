import numpy as np
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple, Union

from src.processing.strategies.base_strategy import BaseStrategy
from src.tactile_types import TactileData


class NoOpScaler:
    """
    Identity scaler to keep pipeline compatibility when scaling is disabled.
    """

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.n_features_in_ = int(np.prod(x.shape[1:])) if x.ndim > 1 else 1
        return x

    def transform(self, x: np.ndarray) -> np.ndarray:
        return x


ScalerLike = Union[StandardScaler, NoOpScaler]


def _scale_evtac_data(
    train_evetac: np.ndarray,
    val_evetac: np.ndarray,
    scaler: ScalerLike,
) -> Tuple[np.ndarray, np.ndarray]:
    train_shape = train_evetac.shape
    val_shape = val_evetac.shape

    train_2d = train_evetac.reshape(train_shape[0], -1)
    val_2d = val_evetac.reshape(val_shape[0], -1)

    train_scaled = scaler.fit_transform(train_2d).reshape(train_shape)
    val_scaled = scaler.transform(val_2d).reshape(val_shape)
    return train_scaled, val_scaled


def _scale_base_data(
    train_base: np.ndarray,
    val_base: np.ndarray,
    scaler: ScalerLike,
) -> Tuple[np.ndarray, np.ndarray]:
    if train_base.ndim == 2:
        train_2d = train_base
        val_2d = val_base
        restore_shape = None
    elif train_base.ndim == 3:
        # Sequence case: (N, T, F) -> (N*T, F)
        n_train, t_train, f_train = train_base.shape
        n_val, t_val, f_val = val_base.shape
        if f_train != f_val:
            raise ValueError(
                f"Incompatible base feature dims in train/val: {f_train} vs {f_val}"
            )
        train_2d = train_base.reshape(n_train * t_train, f_train)
        val_2d = val_base.reshape(n_val * t_val, f_val)
        restore_shape = (train_base.shape, val_base.shape)
    else:
        train_shape = train_base.shape
        val_shape = val_base.shape
        train_2d = train_base.reshape(train_shape[0], -1)
        val_2d = val_base.reshape(val_shape[0], -1)
        restore_shape = (train_shape, val_shape)

    train_scaled = scaler.fit_transform(train_2d)
    val_scaled = scaler.transform(val_2d)

    if restore_shape is not None:
        train_scaled = train_scaled.reshape(restore_shape[0])
        val_scaled = val_scaled.reshape(restore_shape[1])
    return train_scaled, val_scaled


def create_loaders(
    data_dict: TactileData,
    strategy: BaseStrategy,
    batch_size: int = 128,
    existing_scaler_evetac: Optional[ScalerLike] = None,
    existing_scaler_base: Optional[ScalerLike] = None,
    scale_evtac: bool = True,
    split_ratio: float = 0.8,
) -> Tuple[DataLoader, DataLoader, ScalerLike, ScalerLike]:
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
    if not scale_evtac:
        scaler_evetac = (
            existing_scaler_evetac
            if isinstance(existing_scaler_evetac, NoOpScaler)
            else NoOpScaler()
        )
        train_data["evetac"] = scaler_evetac.fit_transform(train_data["evetac"])
        val_data["evetac"] = scaler_evetac.transform(val_data["evetac"])
    else:
        scaler_evetac = existing_scaler_evetac or StandardScaler()
        if existing_scaler_evetac is None:
            train_data["evetac"], val_data["evetac"] = _scale_evtac_data(
                train_data["evetac"], val_data["evetac"], scaler_evetac
            )
        else:
            train_shape = train_data["evetac"].shape
            val_shape = val_data["evetac"].shape
            train_data["evetac"] = scaler_evetac.transform(
                train_data["evetac"].reshape(train_shape[0], -1)
            ).reshape(train_shape)
            val_data["evetac"] = scaler_evetac.transform(
                val_data["evetac"].reshape(val_shape[0], -1)
            ).reshape(val_shape)

    # 3. Scaling Base (Novidade)
    scaler_base = existing_scaler_base or StandardScaler()
    if existing_scaler_base is None:
        train_data["base"], val_data["base"] = _scale_base_data(
            train_data["base"], val_data["base"], scaler_base
        )
    else:
        if train_data["base"].ndim == 2:
            train_data["base"] = scaler_base.transform(train_data["base"])
            val_data["base"] = scaler_base.transform(val_data["base"])
        elif train_data["base"].ndim == 3:
            n_train, t_train, f_train = train_data["base"].shape
            n_val, t_val, f_val = val_data["base"].shape
            if f_train != f_val:
                raise ValueError(
                    f"Incompatible base feature dims in train/val: {f_train} vs {f_val}"
                )
            train_data["base"] = scaler_base.transform(
                train_data["base"].reshape(n_train * t_train, f_train)
            ).reshape(n_train, t_train, f_train)
            val_data["base"] = scaler_base.transform(
                val_data["base"].reshape(n_val * t_val, f_val)
            ).reshape(n_val, t_val, f_val)
        else:
            train_shape = train_data["base"].shape
            val_shape = val_data["base"].shape
            train_data["base"] = scaler_base.transform(
                train_data["base"].reshape(train_shape[0], -1)
            ).reshape(train_shape)
            val_data["base"] = scaler_base.transform(
                val_data["base"].reshape(val_shape[0], -1)
            ).reshape(val_shape)

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
