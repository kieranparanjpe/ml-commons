from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch

from ml_commons.stats import NormalisationStats


class SaveableNetwork(ABC):

    @abstractmethod
    def save(self, path, norm_stats=None) -> None:
        pass

    @classmethod
    @abstractmethod
    def load(cls, path, map_location="cpu", **kwargs):
        pass

    @staticmethod
    def load_norm_stats(path, map_location="cpu") -> Optional[dict[str, NormalisationStats]]:
        """Read the norm_stats dict out of any checkpoint saved via SaveableNetwork.save(), without reconstructing the model."""
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
        return checkpoint.get("norm_stats")
