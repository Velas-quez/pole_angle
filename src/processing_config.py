import hydra
from pathlib import Path
from omegaconf import DictConfig
import os


@hydra.main(version_base=None, config_path="config", config_name="processing_config")
def main(cfg: DictConfig):
    print("🚀 Iniciando Runner de Processamento Offline")
    print("-" * 30)

    # 1. Instanciar a Strategy definida no YAML (via defaults)
    strategy = hydra.utils.instantiate(cfg.strategies)

    # 2. Resolver caminhos de entrada
    input_base = Path(cfg.input.dir)
    evtac_path = input_base / cfg.input.evtac
    markers_path = input_base / cfg.input.markers
    opti_path = input_base / cfg.input.opti

    # 3. Validar se arquivos brutos existem
    for p in [evtac_path, markers_path, opti_path]:
        if not p.exists():
            print(f"❌ Erro: Arquivo não encontrado: {p}")
            return

    # 4. Definir saída
    output_base_dir = Path(cfg.output.base_dir)
    strategy_dir = output_base_dir / strategy.get_dirname()

    if strategy_dir.exists() and not cfg.get("force_reprocess", False):
        print(f"⏩ Cache já existe em: {strategy_dir.name}. Pulando...")
        return

    print(f"⚙️ Executando Strategy: {strategy.name}")
    print(f"📂 Destino: {strategy_dir}")

    # 5. Executar a Transformação
    strategy.transform(
        evtac_path=str(evtac_path),
        markers_path=str(markers_path),
        opti_path=str(opti_path),
        output_base_dir=str(output_base_dir),
    )

    print("-" * 30)
    print("✅ Processamento concluído com sucesso!")


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"  # Para debug mais detalhado
    main()
