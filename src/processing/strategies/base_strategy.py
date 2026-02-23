from abc import ABC, abstractmethod
from typing import Dict, Any, List
from torch.utils.data import Dataset
from src.types import TactileData


class BaseStrategy(ABC):
    name: str = "BaseStrategy"

    @abstractmethod
    def get_dirname(self) -> str:
        """Retorna o nome da pasta onde os dados serão guardados."""
        pass

    @abstractmethod
    def transform(
        self, evtac_path: str, markers_path: str, opti_path: str, output_base_dir: str
    ) -> None:
        """Executa o processamento e guarda os ficheiros na pasta da estratégia."""
        pass

    @abstractmethod
    def load_processed_data(
        self, base_path: str, target_cols: List[str]
    ) -> TactileData:
        """Lê os dados da pasta e retorna um dicionário de arrays numpy."""
        pass

    @abstractmethod
    def get_dataset(self, data_dict: Dict[str, Any]) -> Dataset:
        """Retorna a instância do Dataset correta para esta estratégia."""
        pass
