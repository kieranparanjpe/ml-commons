from dataclasses import dataclass

import torch

from ml_commons.execution.base_trainer import BaseTrainer
from ml_commons.log.logger import NullLogger, WandBLogger


class _DummyTrainer(BaseTrainer):
    def run(self):
        pass


@dataclass
class _RunConfig:
    lr: float = 0.1


def test_create_logger_returns_null_logger_when_logging_disabled(sample_run_info):
    trainer = _DummyTrainer(sample_run_info, _RunConfig(), entity="me", project="proj",
                            log_elements={}, logging=False)
    assert isinstance(trainer._logger, NullLogger)


def test_create_logger_returns_wandb_logger_when_logging_enabled(mock_wandb, sample_run_info):
    trainer = _DummyTrainer(sample_run_info, _RunConfig(), entity="me", project="proj",
                            log_elements={"loss": 0.0}, logging=True)
    assert isinstance(trainer._logger, WandBLogger)


def test_device_selects_cpu_when_cuda_unavailable(monkeypatch, sample_run_info):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    trainer = _DummyTrainer(sample_run_info, _RunConfig(), entity="me", project="proj",
                            log_elements={}, logging=False)
    assert trainer.device == torch.device("cpu")


def test_device_selects_cuda_when_available(monkeypatch, sample_run_info):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    trainer = _DummyTrainer(sample_run_info, _RunConfig(), entity="me", project="proj",
                            log_elements={}, logging=False)
    assert trainer.device == torch.device("cuda")
