from __future__ import annotations

import torch
from typing import Any, Callable

from ml_commons.config.run_info import RunInfo
from ml_commons.execution.base_trainer import BaseTrainer


def run_one(run_config: Any, index: int | None, *,
            run_info_factory: Callable[[int | None], RunInfo],
            trainer_factory: Callable[[RunInfo, Any], BaseTrainer]) -> bool:
    """Standard single-run entry point for process-pool training.

    Positional args (run_config, index) vary per call; keyword args are fixed
    and designed to be bound via functools.partial.

    Args:
        run_config: Project-specific config for this run.
        index: Grid index if part of a grid search, else None.
        run_info_factory: Callable(index) -> RunInfo. Creates run metadata.
        trainer_factory: Callable(run_info, run_config) -> BaseTrainer.
    """
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    run_info = run_info_factory(index)
    trainer_factory(run_info, run_config).run()
    return True
