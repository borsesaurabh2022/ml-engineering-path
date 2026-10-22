import matplotlib.pyplot as plt
import torch

from src.data import make_moons, train_val_test_split
from src.model import MLP
from src.visualize import plot_decision_boundary

if __name__ == "__main__":
    generator = torch.Generator().manual_seed(0)
    X, y = make_moons(1000, noise=0.2, generator=generator)
    _, (X_val, y_val), _ = train_val_test_split(
        X, y, val_frac=0.15, test_frac=0.15, generator=generator)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, hidden in zip(axes, [[], [16, 16]]):
        name = "mlp_" + ("-".join(map(str, hidden)) or "linear")
        model = MLP(2, hidden, 2)
        model.load_state_dict(torch.load(f"checkpoints/{name}.pt", weights_only=True))
        plot_decision_boundary(model, X_val, y_val, ax, f"{name}: validation split")
    fig.savefig("figures/decision_boundary_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
