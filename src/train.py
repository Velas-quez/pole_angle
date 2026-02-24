import os
from pathlib import Path
from typing import MutableMapping

import hydra
from hydra.core.hydra_config import HydraConfig
import joblib
import mlflow.artifacts
import mlflow
import torch
from omegaconf import DictConfig

import sys

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))
# Utilitários do projeto
from src.processing.strategies.base_strategy import BaseStrategy
from src.utils.create_loaders import create_loaders
from src.utils.trainer import run_training_loop
from src.tactile_types import TactileData  # Importando o tipo para garantir consistência


@hydra.main(version_base=None, config_path="config", config_name="config")
def train(cfg: DictConfig):
    # 1. Setup MLflow
    mlflow.set_tracking_uri(Path("mlruns").resolve().as_uri())
    mlflow.set_experiment(cfg.experiment_name)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg.training.use_gpu else "cpu"
    )

    # 2. Instanciar Estratégia via Hydra
    choices = HydraConfig.get().runtime.choices
    print(cfg.strategies)
    strategy: BaseStrategy = hydra.utils.instantiate(cfg.strategies)

    # 3. Gerenciamento de Pastas
    processed_base_dir = Path(cfg.data.processed_dir.train)
    strategy_dir = processed_base_dir / strategy.get_dirname()

    # Processar se a pasta da estratégia não existir
    if not strategy_dir.exists():
        print(f"⚙️ Processando novos dados com a estratégia: {choices['strategies']}")
        strategy.transform(
            evtac_path=cfg.data.pre_processed.train.evtac,
            markers_path=cfg.data.pre_processed.train.markers,
            opti_path=cfg.data.pre_processed.train.opti,
            output_base_dir=str(processed_base_dir),
        )
    else:
        print(f"📦 Cache encontrado em: {strategy_dir.name}")

    # 4. Carregar Dados (Delegado à Strategy)
    # A strategy retorna um objeto TactileData (TypedDict)
    target_cols = list(cfg.targets.columns)
    data_dict: TactileData = strategy.load_processed_data(
        base_path=str(processed_base_dir), target_cols=target_cols
    )

    # 5. Lógica de Resume vs New Run (Dual Scalers)
    resume_id = cfg.get("resume_run_id", None)
    scaler_evetac_path = strategy_dir / "scaler_evetac.pkl"
    scaler_base_path = strategy_dir / "scaler_base.pkl"

    scaler_evetac = None
    scaler_base = None
    model = None

    with mlflow.start_run(
        run_id=resume_id, run_name=f"{choices['strategies']}_{choices['models']}"
    ) as run:
        mlflow.log_param("strategy", choices["strategies"])
        mlflow.log_param("model", choices["models"])
        mlflow.log_param("target", choices["targets"])

        if resume_id:
            print(f"🔄 Resuming Run: {resume_id}")
            # Recuperar scalers do MLflow
            uri_evetac = mlflow.artifacts.download_artifacts(
                run_id=resume_id, artifact_path="scaler_evetac.pkl"
            )
            uri_base = mlflow.artifacts.download_artifacts(
                run_id=resume_id, artifact_path="scaler_base.pkl"
            )
            scaler_evetac = joblib.load(uri_evetac)
            scaler_base = joblib.load(uri_base)

            # Recuperar modelo
            model_uri = f"runs:/{resume_id}/model"
            model = mlflow.pytorch.load_model(model_uri).to(device)

        elif scaler_evetac_path.exists() and scaler_base_path.exists():
            print("✅ Carregando Scalers pré-existentes do disco.")
            scaler_evetac = joblib.load(scaler_evetac_path)
            scaler_base = joblib.load(scaler_base_path)

        # 6. Preparar Loaders com Dual Scalers
        train_loader, val_loader, scaler_evetac, scaler_base = create_loaders(
            data_dict=data_dict,
            strategy=strategy,
            batch_size=cfg.training.batch_size,
            existing_scaler_evetac=scaler_evetac,
            existing_scaler_base=scaler_base,
        )

        # 7. Definição das Dimensões para o Modelo
        # Pegamos os shapes diretamente do TactileData processado

        input_shape_dict = {
            "evetac": data_dict["evetac"].shape[1:],  # Pega tudo após o Batch
            "base": data_dict["base"].shape[1:],
        }
        output_shape = data_dict["target"].shape[1:]
        # 8. Instanciar Modelo
        if model is None:
            print(f"🚀 Inicializando modelo: {cfg.models._target_}")
            # Passamos as dimensões separadas para que o modelo saiba lidar com o TactileInput
            model = hydra.utils.instantiate(
                cfg.models,
                input_shape_dict=input_shape_dict,
                output_shape=output_shape,
            ).to(device)

            # Salvar Scalers localmente e no MLflow
            joblib.dump(scaler_evetac, scaler_evetac_path)
            joblib.dump(scaler_base, scaler_base_path)
            mlflow.log_artifact(str(scaler_evetac_path))
            mlflow.log_artifact(str(scaler_base_path))

        # Logar parâmetros globais
        mlflow.log_params(cfg.strategies)
        mlflow.log_params(cfg.targets)
        mlflow.log_params(cfg.training)

        # 9. Training Loop
        best_model = run_training_loop(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=cfg,
            device=device,
        )

        # 10. Finalizar
        mlflow.pytorch.log_model(best_model, "model")
        print(f"✨ Run {run.info.run_id} finalizada com sucesso.")


def flatten_dict(d: MutableMapping, parent_key: str = "", sep: str = ".") -> dict:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"  # Para debug mais detalhado
    train()
