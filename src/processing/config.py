from __future__ import annotations

from pathlib import Path
from typing import Dict

import yaml
import importlib.util
import sys

from processing.base_processing import BaseStrategy

CONFIG_PATH = Path(__file__).with_name("config.yaml")
STRATEGIES_DIR = Path(__file__).parent / "strategies"


def load_model_map(config_path: Path | None = None) -> Dict[str, str]:
    """Load and validate the model-to-processor mapping from config.yaml."""
    path = config_path or CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    models = data.get("models", {})
    if not isinstance(models, dict) or not models:
        raise ValueError("Config must define a non-empty 'models' mapping")

    return {str(k): str(v) for k, v in models.items()}


def load_algorithm_module(name: str, strategies_dir: Path | None = None):
    """Dynamically load a processing algorithm module by name."""
    base_dir = strategies_dir or STRATEGIES_DIR
    path = base_dir / f"{name}.py"
    if not path.exists():
        raise FileNotFoundError(f"Algorithm not found: {path}")

    module_name = f"processing_algorithm_{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load algorithm module from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def get_model_strategy(model_name: str, **strategy_kwargs) -> BaseStrategy:
    """Return the Strategy instance for the given model name."""
    model_map = load_model_map()
    if model_name not in model_map:
        raise KeyError(f"Processing config has no processor for model '{model_name}'")

    processor_name = model_map[model_name]
    module = load_algorithm_module(processor_name)
    if not hasattr(module, "Strategy"):
        raise AttributeError(
            f"Processor '{processor_name}' does not expose a Strategy class"
        )
    instance = module.Strategy(**strategy_kwargs) if strategy_kwargs else module.Strategy()
    if not isinstance(instance, BaseStrategy):
        raise TypeError(
            f"Processor '{processor_name}' Strategy must inherit from BaseStrategy"
        )
    return instance
