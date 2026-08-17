from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class NormalisationStats:
    mean: np.ndarray
    var: np.ndarray

    def std(self) -> np.ndarray:
        return np.sqrt(self.var)

    def as_tensors(self, dtype: torch.dtype = torch.float32, device=None) -> tuple[torch.Tensor, torch.Tensor]:
        mean = torch.as_tensor(self.mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.std(), dtype=dtype, device=device)
        return mean, std


torch.serialization.add_safe_globals([
    NormalisationStats,
    np._core.multiarray._reconstruct, np.ndarray, np.dtype,
    np.dtypes.Float32DType, np.dtypes.Float64DType,
])
