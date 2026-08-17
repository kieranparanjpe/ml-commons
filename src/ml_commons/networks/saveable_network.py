from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from ml_commons.stats import NormalisationStats


class SaveableNetwork(ABC):
    """Norm stats live as named attributes on the model (e.g. obs_norm_stats), set in __init__,
    persisted automatically by save()/load() alongside the rest of the checkpoint -- same pattern
    already used for `config`. No norm_stats param on save(), no tuple return from load()."""

    @abstractmethod
    def save(self, path) -> None:
        pass

    @classmethod
    @abstractmethod
    def load(cls, path, map_location="cpu", **kwargs):
        pass

    @staticmethod
    def load_norm_stats(path, map_location="cpu") -> dict[str, NormalisationStats]:
        """Read every NormalisationStats-valued entry out of a checkpoint saved via SaveableNetwork.save(),
        keyed by whatever attribute name the model saved it under, without reconstructing the model."""
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
        return {k: v for k, v in checkpoint.items() if isinstance(v, NormalisationStats)}
