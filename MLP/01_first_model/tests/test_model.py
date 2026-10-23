import pytest
import torch

from src.model import MLP


def test_output_shape():
    model = MLP(2, [16, 16], 2)
    x = torch.randn(5, 2)
    assert model(x).shape == (5, 2)


@pytest.mark.parametrize("hidden, expected", [([], 6), ([16], 82), ([16, 16], 354)])
def test_parameter_count(hidden, expected):
    model = MLP(2, hidden, 2)
    assert sum(p.numel() for p in model.parameters()) == expected


def test_gradients_reach_every_parameter():
    model = MLP(2, [16, 16], 2)
    loss = model(torch.randn(8, 2)).sum()
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no gradient for {name}"
