import math

import matplotlib.pyplot as plt
import torch

Split = tuple[torch.Tensor, torch.Tensor]


def make_moons(n_samples: int, noise: float, generator: torch.Generator) -> Split:
    """Two interleaved half-circles in 2-D. Returns features (N, 2) float32, labels (N,) long."""
    n_upper = n_samples // 2
    n_lower = n_samples - n_upper
    theta_upper = torch.rand(n_upper, generator=generator) * math.pi
    theta_lower = torch.rand(n_lower, generator=generator) * math.pi

    upper = torch.stack([torch.cos(theta_upper), torch.sin(theta_upper)], dim=1)
    lower = torch.stack([1 - torch.cos(theta_lower), 0.5 - torch.sin(theta_lower)], dim=1)

    X = torch.cat([upper, lower])
    X = X + noise * torch.randn(X.shape, generator=generator)
    y = torch.cat([torch.zeros(n_upper, dtype=torch.long),
                   torch.ones(n_lower, dtype=torch.long)])
    return X, y


def train_val_test_split(
        X: torch.Tensor, y: torch.Tensor, val_frac: float, test_frac: float,
        generator: torch.Generator
) -> tuple[Split, Split, Split]:
    """Shuffle once, then cut into train / validation / test."""
    perm = torch.randperm(len(X), generator=generator)
    n_test = int(len(X) * test_frac)
    n_val = int(len(X) * val_frac)
    test_idx = perm[:n_test]
    val_idx = perm[n_test:n_test + n_val]
    train_idx = perm[n_test + n_val:]
    return (X[train_idx], y[train_idx]), (X[val_idx], y[val_idx]), (X[test_idx], y[test_idx])


if __name__ == "__main__":
    generator = torch.Generator().manual_seed(0)
    X, y = make_moons(1000, noise=0.2, generator=generator)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = train_val_test_split(
        X, y, val_frac=0.15, test_frac=0.15, generator=generator)

    print("X:", X.shape, X.dtype, "| y:", y.shape, y.dtype)
    print("train / val / test:", len(X_train), len(X_val), len(X_test))
    print("class-1 fraction in train:", y_train.float().mean().item())

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap="coolwarm", s=10)
    ax.set_title("Two moons: training split")
    plt.show()
