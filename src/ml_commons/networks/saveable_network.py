from __future__ import annotations

from abc import ABC, abstractmethod


class SaveableNetwork(ABC):

    @abstractmethod
    def save(self, path, norm_stats=None) -> None:
        pass

    @classmethod
    @abstractmethod
    def load(cls, path, map_location="cpu", **kwargs):
        pass
