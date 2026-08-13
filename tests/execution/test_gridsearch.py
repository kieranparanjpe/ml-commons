import json

from ml_commons.execution.gridsearch import gridsearch

from tests.execution._workers import record_index


def test_gridsearch_runs_every_config_exactly_once(tmp_path):
    configs = [{"out_dir": str(tmp_path), "value": v * 10} for v in range(6)]

    gridsearch(record_index, configs, max_parallel=3)

    written = sorted(tmp_path.glob("*.json"))
    assert len(written) == 6

    seen_indices = set()
    for path in written:
        data = json.loads(path.read_text())
        seen_indices.add(data["index"])
        assert data["value"] == configs[data["index"]]["value"]

    assert seen_indices == set(range(6))


def test_gridsearch_defaults_max_parallel(tmp_path):
    configs = [{"out_dir": str(tmp_path), "value": v} for v in range(3)]

    gridsearch(record_index, configs)

    assert len(list(tmp_path.glob("*.json"))) == 3
