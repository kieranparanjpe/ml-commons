import torch

from ml_commons.execution.run_one import run_one


class _StubTrainer:
    def __init__(self):
        self.run_called = 0

    def run(self):
        self.run_called += 1


# torch.set_num_interop_threads can only be called once per process (raises on a
# second call), so run_one is only ever invoked once across this whole test suite.
def test_run_one_executes_trainer_and_sets_single_threaded_torch():
    stub = _StubTrainer()

    result = run_one({"x": 1}, 2, trainer_factory=lambda config, index: stub)

    assert result is True
    assert stub.run_called == 1
    assert torch.get_num_threads() == 1
    assert torch.get_num_interop_threads() == 1
