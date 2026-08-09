from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch

from ml_commons.config.run_info import RunInfo
from ml_commons.log import Logger, WandBLogger, NullLogger


class BaseTrainer(ABC):

    _logger: Logger

    def __init__(self, run_info: RunInfo, run_config: Any,
                 entity: str, project: str, log_elements: dict,
                 logging: bool = True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._run_info = run_info
        self._run_config = run_config

        print(f"\nRun: {self._run_info.run_name()}")
        print(f"Config: {self._run_config!r}")

        self._logger = self._create_logger(
            run_info, entity, project,
            hyperparameters=vars(run_config),
            elements=log_elements,
            logging=logging,
        )

    def _create_logger(self, run_info: RunInfo, entity: str, project: str,
                       hyperparameters: dict, elements: dict,
                       logging: bool) -> Logger:
        """Create the logger for this run. Override in subclasses to use a different logger type."""
        if logging:
            return WandBLogger(run_info, entity, project,
                               hyperparameters=hyperparameters,
                               elements=elements)
        return NullLogger()

    @abstractmethod
    def run(self):
        """Execute the main training loop."""
        pass
