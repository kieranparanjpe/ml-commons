from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class NormalisationStats:
    """Default is the identity transform (mean=0, var=1) -- scalars broadcast against any
    obs/action shape, so "no stats known" and "no-op normalization" are the same value."""
    mean: np.ndarray = field(default_factory=lambda: np.array(0.0))
    var: np.ndarray = field(default_factory=lambda: np.array(1.0))

    def std(self) -> np.ndarray:
        return np.sqrt(self.var)

    def mean_t(self, dtype: torch.dtype = torch.float32, device=None) -> torch.Tensor:
        return torch.as_tensor(self.mean, dtype=dtype, device=device)

    def std_t(self, dtype: torch.dtype = torch.float32, device=None) -> torch.Tensor:
        return torch.as_tensor(self.std(), dtype=dtype, device=device)

    def as_tensors(self, dtype: torch.dtype = torch.float32, device=None) -> tuple[torch.Tensor, torch.Tensor]:
        return self.mean_t(dtype, device), self.std_t(dtype, device)


torch.serialization.add_safe_globals([
    NormalisationStats,
    np._core.multiarray._reconstruct, np.ndarray, np.dtype,
    np.dtypes.Float32DType, np.dtypes.Float64DType,
])
