import torch

from src.data import make_moons, train_val_test_split


def make_splits(seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    X, y = make_moons(1000, noise=0.2, generator=generator)
    return X, y, train_val_test_split(X, y, val_frac=0.15, test_frac=0.15, generator=generator)


def as_row_set(X: torch.Tensor) -> set[tuple[float, ...]]:
    return {tuple(row) for row in X.tolist()}


def test_shapes_and_dtypes():
    X, y, _ = make_splits()
    assert X.shape == (1000, 2) and X.dtype == torch.float32
    assert y.shape == (1000,) and y.dtype == torch.long
    assert set(y.tolist()) == {0, 1}


def test_split_sizes():
    _, _, ((X_train, _), (X_val, _), (X_test, _)) = make_splits()
    assert (len(X_train), len(X_val), len(X_test)) == (700, 150, 150)


def test_splits_do_not_overlap():
    _, _, ((X_train, _), (X_val, _), (X_test, _)) = make_splits()
    train, val, test = as_row_set(X_train), as_row_set(X_val), as_row_set(X_test)
    assert not train & val and not train & test and not val & test


def test_same_seed_gives_same_data():
    X1, y1, _ = make_splits(seed=0)
    X2, y2, _ = make_splits(seed=0)
    assert torch.equal(X1, X2) and torch.equal(y1, y2)
