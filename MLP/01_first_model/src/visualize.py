from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn

from src.model import MLP


def draw_mlp(model: nn.Module, ax: plt.Axes) -> None:
    """Draw neurons as circles and weights as lines (colour = sign, width = magnitude)."""
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    sizes = [linears[0].in_features] + [layer.out_features for layer in linears]

    # (x, y) of every neuron: x = layer index, y centred around 0
    positions = [[(x, y - (n - 1) / 2) for y in range(n)] for x, n in enumerate(sizes)]

    max_abs = max(layer.weight.abs().max().item() for layer in linears)
    for k, layer in enumerate(linears):
        weights = layer.weight.detach()  # shape (out, in)
        for i, (x1, y1) in enumerate(positions[k + 1]):
            for j, (x0, y0) in enumerate(positions[k]):
                w = weights[i, j].item()
                ax.plot([x0, x1], [y0, y1],
                        color="tab:blue" if w > 0 else "tab:red",
                        linewidth=2.5 * abs(w) / max_abs, alpha=0.6, zorder=1)

    for layer_pos in positions:
        xs, ys = zip(*layer_pos, strict=True)
        ax.scatter(xs, ys, s=150, color="white", edgecolors="black", zorder=2)

    names = ["input"] + [f"hidden {i}" for i in range(1, len(sizes) - 1)] + ["output"]
    ax.set_xticks(range(len(sizes)), [f"{name}\n({n})" for name,
                                      n in zip(names, sizes, strict=True)])
    ax.set_yticks([])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    n_params = sum(p.numel() for p in model.parameters())
    ax.set_title(f"MLP {sizes}: {n_params} parameters (blue = positive, red = negative weight)")


@torch.no_grad()
def plot_decision_boundary(model: nn.Module, X: torch.Tensor, y: torch.Tensor,
                           ax: plt.Axes, title: str) -> None:
    """Colour the input plane by predicted P(class 1); black line = boundary at 0.5."""
    model.eval()
    x0_min, x0_max = X[:, 0].min().item() - 0.5, X[:, 0].max().item() + 0.5
    x1_min, x1_max = X[:, 1].min().item() - 0.5, X[:, 1].max().item() + 0.5
    g0, g1 = torch.meshgrid(torch.linspace(x0_min, x0_max, 300),
                            torch.linspace(x1_min, x1_max, 300), indexing="xy")
    grid = torch.stack([g0.ravel(), g1.ravel()], dim=1)  # (90000, 2)
    p1 = torch.softmax(model(grid), dim=1)[:, 1].reshape(g0.shape)

    ax.contourf(g0, g1, p1, levels=20, cmap="coolwarm", alpha=0.6)
    ax.contour(g0, g1, p1, levels=[0.5], colors="black", linewidths=1.5)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="black", s=15)
    ax.set_title(title)



if __name__ == "__main__":
    torch.manual_seed(0)
    model = MLP(2, [16, 16], 2)
    fig, ax = plt.subplots(figsize=(9, 6))
    draw_mlp(model, ax)
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/architecture.png", dpi=150, bbox_inches="tight")
    plt.show()
