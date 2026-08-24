from __future__ import annotations

import os
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from ml_commons.config.run_info import RunInfo

import wandb


class Logger(ABC):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def finish(self):
        pass

    @abstractmethod
    def add_elements(self, elements: Dict[str, Any]):
        pass

    @abstractmethod
    def set_element_step_metric(self, elements: Dict[str, str]):
        pass

    @abstractmethod
    def reset(self, *fields: str):
        pass

    @abstractmethod
    def set_log_data(self, kvps: Dict[str, Any]):
        pass

    @abstractmethod
    def sum_log_data(self, kvps: Dict[str, Any]):
        pass

    @abstractmethod
    def log_data(self, *fields):
        pass

    @abstractmethod
    def set_prefix(self, prefix, *fields):
        pass

    @abstractmethod
    def add_tags(self, *tags):
        pass


class WandBLogger(Logger):
    def __init__(self, run_info: RunInfo, entity: str, project: str,
                 hyperparameters: Dict[str, Any], elements: Dict[str, Any], default_x_axis="global_step"):
        super().__init__()
        self._run = self._wandb_run = wandb.init(
            entity=entity,
            project=project,
            name=run_info.run_name(),
            tags=run_info.tags(),
            job_type="train",
            config=hyperparameters,
            group=run_info.group()
        )

        self._elements_start = elements
        self._elements = deepcopy(self._elements_start)
        self._elements_prefix = {}

        self._run.define_metric("*", step_metric=default_x_axis)

    def finish(self):
        self._run.finish()

    def add_elements(self, elements: Dict[str, Any]):
        self._elements.update(deepcopy(elements))
        self._elements_start.update(elements)

    def set_element_step_metric(self, elements: Dict[str, str]):
        for field in elements.keys():
            self._run.define_metric(field, step_metric=elements[field])

    def reset(self, *fields: str):
        if fields is None or len(fields) == 0:
            self._elements = deepcopy(self._elements_start)
        else:
            for field in fields:
                self._elements[field] = self._elements_start[field]

    def set_log_data(self, kvps: Dict[str, Any]):
        self._elements.update(kvps)

    def sum_log_data(self, kvps: Dict[str, Any]):
        for k, v in kvps.items():
            self._elements[k] += v

    def log_data(self, *fields):
        prefixed_elements = {f"{self._elements_prefix.get(k, "")}{k}": v for k, v in self._elements.items()}

        if fields is None or len(fields) == 0:
            self._run.log(data=prefixed_elements)
        else:
            data = {k: v for k, v in prefixed_elements.items() if k in fields}
            self._run.log(data=data)

    def set_prefix(self, elements : Dict[str, str]):
        for k, v in elements.items():
            self._elements_prefix[k] = v

    def add_tags(self, *tags):
        self._run.tags = (*self._run.tags, *tags)


class NullLogger(Logger):

    def finish(self):
        pass

    def add_elements(self, elements: Dict[str, Any]):
        pass

    def reset(self, *fields: str):
        pass

    def set_log_data(self, kvps: Dict[str, Any]):
        pass

    def sum_log_data(self, kvps: Dict[str, Any]):
        pass

    def log_data(self, *fields):
        pass

    def set_element_step_metric(self, elements: Dict[str, str]):
        pass

    def set_prefix(self, prefix, *fields):
        pass

    def add_tags(self, *tags):
        pass
