import pytest
import torch

from ml_commons.networks import ACTIVATION_MAP, NetworkConfig


@pytest.mark.parametrize("activation,expected_type", [
    ("relu", torch.nn.ReLU),
    ("tanh", torch.nn.Tanh),
    ("leaky_relu", torch.nn.LeakyReLU),
])
def test_build_trunk_uses_correct_activation(activation, expected_type):
    trunk, out_size = NetworkConfig(hidden_sizes=[16], activation=activation).build_trunk(8)

    assert isinstance(trunk[0], torch.nn.Linear)
    assert isinstance(trunk[1], expected_type)
    assert out_size == 16


def test_build_trunk_layer_shapes():
    trunk, out_size = NetworkConfig(hidden_sizes=[32, 16]).build_trunk(8)

    linears = [layer for layer in trunk if isinstance(layer, torch.nn.Linear)]
    assert [l.in_features for l in linears] == [8, 32]
    assert [l.out_features for l in linears] == [32, 16]
    assert out_size == 16


def test_build_trunk_forward_pass_shape():
    trunk, out_size = NetworkConfig(hidden_sizes=[32, 16]).build_trunk(8)

    output = trunk(torch.randn(4, 8))
    assert output.shape == (4, out_size)


def test_build_trunk_empty_hidden_sizes_is_identity():
    trunk, out_size = NetworkConfig(hidden_sizes=[]).build_trunk(8)

    assert len(list(trunk.children())) == 0
    assert out_size == 8


def test_build_trunk_invalid_activation_raises():
    with pytest.raises(KeyError):
        NetworkConfig(activation="invalid").build_trunk(8)


def test_activation_map_matches_literal_options():
    assert set(ACTIVATION_MAP.keys()) == {"relu", "tanh", "leaky_relu"}
