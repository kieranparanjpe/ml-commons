from __future__ import annotations

import torch
from typing import Any, Callable

from ml_commons.config.run_info import RunInfo
from ml_commons.execution.base_trainer import BaseTrainer


def run_one(run_config: Any, index: int | None, *,
            trainer_factory: Callable[[Any, int | None], BaseTrainer]) -> bool:
    """Standard single-run entry point for process-pool training.

    Positional args (run_config, index) vary per call; keyword args are fixed
    and designed to be bound via functools.partial.

    Args:
        run_config: Project-specific config for this run.
        index: Grid index if part of a grid search, else None.
        trainer_factory: Callable(run_config, index) -> BaseTrainer.
    """
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    trainer_factory(run_config, index).run()
    return True
