from dataclasses import dataclass, field
from pathlib import Path

from ml_commons.config.loader import ConfigLoader

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@dataclass
class AlgoSection:
    n_timesteps: int = 1_000_000
    lr: float = 3e-4


@dataclass
class PolicySection:
    hidden_sizes: list[int] = field(default_factory=lambda: [64, 64])
    activation: str = "relu"


SECTIONS = {"algorithm": AlgoSection, "policy": PolicySection}


def test_load_single():
    result = ConfigLoader.load_single(str(FIXTURES_DIR / "single_run.json"), SECTIONS)

    assert result["algorithm"] == AlgoSection(n_timesteps=50000, lr=0.001)
    assert result["policy"] == PolicySection(hidden_sizes=[32, 32], activation="tanh")


def test_load_grid_cartesian_product():
    configs = ConfigLoader.load_grid(str(FIXTURES_DIR / "grid_run.json"), SECTIONS)

    assert len(configs) == 4  # 2 lr values x 2 activation values

    seen = {(c["algorithm"].lr, c["policy"].activation) for c in configs}
    assert seen == {
        (0.001, "relu"), (0.001, "tanh"),
        (0.0003, "relu"), (0.0003, "tanh"),
    }

    for c in configs:
        assert c["algorithm"].n_timesteps == 50000
        assert c["policy"].hidden_sizes == [32, 32]


def test_load_grid_only_requested_sections_are_parsed():
    configs = ConfigLoader.load_grid(str(FIXTURES_DIR / "grid_run.json"), {"algorithm": AlgoSection})

    assert len(configs) == 2
    assert {c["algorithm"].lr for c in configs} == {0.001, 0.0003}
