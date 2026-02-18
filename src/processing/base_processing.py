from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseStrategy(ABC):
    @abstractmethod
    def transform(
        self,
        evtac_path: str,
        markers_path: str,
        opti_path: str,
        output_path: str | None = None,
    ) -> pd.DataFrame:
        """Load raw inputs, optionally persist parquet, and return a processed DataFrame."""
        raise NotImplementedError
