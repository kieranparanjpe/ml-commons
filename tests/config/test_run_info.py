from datetime import datetime

import pytest

from ml_commons.config.run_info import RunInfo


def _run_info(grid_index):
    return RunInfo(
        task_id="CartPole-v1",
        algorithm_id="ppo",
        grid_index=grid_index,
        time=datetime(2026, 1, 1, 12, 0, 0),
    )


def test_tags():
    assert _run_info(None).tags() == ["ppo", "CartPole-v1"]


@pytest.mark.parametrize("grid_index,expected_group", [
    (None, None),
    (3, "CartPole-v1@2026-01-01-12-00-00"),
])
def test_group(grid_index, expected_group):
    assert _run_info(grid_index).group() == expected_group


@pytest.mark.parametrize("grid_index,expected_name", [
    (None, "CartPole-v1@2026-01-01-12-00-00"),
    (3, "CartPole-v1@2026-01-01-12-00-00_RUN-3"),
])
def test_run_name(grid_index, expected_name):
    assert _run_info(grid_index).run_name() == expected_name


@pytest.mark.parametrize("grid_index,expected_path", [
    (None, "saved_policies/CartPole-v1/CartPole-v1@2026-01-01-12-00-00"),
    (3, "saved_policies/CartPole-v1/CartPole-v1@2026-01-01-12-00-00/"
        "CartPole-v1@2026-01-01-12-00-00_RUN-3"),
])
def test_local_folder_path(grid_index, expected_path):
    assert _run_info(grid_index).local_folder_path("saved_policies") == expected_path
