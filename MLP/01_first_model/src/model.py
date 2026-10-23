import torch
from torch import nn


class MLP(nn.Module):
    """Multi-layer perceptron: (Linear -> ReLU) blocks followed by a final Linear layer."""

    def __init__(self, in_features: int, hidden_sizes: list[int], out_features: int):
        super().__init__()
        layers: list[nn.Module] = []  # Create a list to hold the layers of the MLP
        prev = in_features
        for size in hidden_sizes:
            layers.append(nn.Linear(prev, size))  # Add a Linear layer to the list
            layers.append(nn.ReLU())  # Add a ReLU activation function to the list
            prev = size
        layers.append(nn.Linear(prev, out_features))  # Add the final Linear layer to the list
        self.model = nn.Sequential(*layers)  # Create a Sequential model from the list of layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)  # Pass the input through the model and return the output
