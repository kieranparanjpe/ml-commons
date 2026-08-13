from datetime import datetime
from unittest.mock import MagicMock

import pytest
import wandb

from ml_commons.config.run_info import RunInfo


@pytest.fixture
def sample_run_info() -> RunInfo:
    return RunInfo(
        task_id="CartPole-v1",
        algorithm_id="ppo",
        grid_index=None,
        time=datetime(2026, 1, 1, 12, 0, 0),
    )


@pytest.fixture
def mock_wandb(monkeypatch):
    mock_run = MagicMock()
    mock_init = MagicMock(return_value=mock_run)
    monkeypatch.setattr(wandb, "init", mock_init)
    return mock_init, mock_run
