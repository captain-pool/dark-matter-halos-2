from jax import numpy as jnp
from jax.typing import ArrayLike
import numpy as np


def train_test_knn(
    distance_matrix: ArrayLike,
    targets: ArrayLike,
    train_indices: ArrayLike,
    test_indices: ArrayLike,
    k: int = 4,
    weighting: str = "uniform",
) -> float:
    tol = 1e-8
    # assert (
    #     len(set(train_indices).union(set(test_indices)))
    #     == max(jnp.concatenate([train_indices, test_indices])) + 1
    # )
    # assert len(set(train_indices).intersection(set(test_indices))) == 0
    test_to_train_distance_matrix = distance_matrix[test_indices, :][:, train_indices]
    # Take nearest k points that are not the point itself:
    nearest_k_pts = test_to_train_distance_matrix.argsort(axis=1)[:, :k]
    values = targets[train_indices][nearest_k_pts]
    match weighting:
        case "inverse_sqrt":
            distances = jnp.sort(test_to_train_distance_matrix, axis=1)[:, :k]
            weights = 1 / (jnp.sqrt(distances) + tol)
            weights = weights / weights.sum(axis=-1, keepdims=True)
        case "inverse_square":
            distances = jnp.sort(test_to_train_distance_matrix, axis=1)[:, :k]
            weights = 1 / (jnp.square(distances) + tol)
            weights = weights / weights.sum(axis=-1, keepdims=True)
        case "inverse":
            distances = jnp.sort(test_to_train_distance_matrix, axis=-1)[:, :k]
            weights = 1 / (distances + tol)
            weights = weights / weights.sum(axis=-1, keepdims=True)
        case "uniform":
            weights = jnp.ones(k) / k

        case _:
            raise ValueError(f"Weight method {weighting} unknown")

    residuals = (values * weights).sum(axis=-1) - targets[test_indices]

    # returning RMSE:
    return jnp.sqrt(jnp.mean(jnp.square(residuals))).item()


def test_train_test_knn(k: int, weighting: str = "uniform"):
    """Does a very specific test of the function `test_train_test_knn`. If
    `k=2` and `weighting="uniform"`, then the knn RMSE should be zero."""
    distance_matrix = jnp.array(
        [
            [0.0, 1.0, 1.0, 0.0, 2.0],
            [1.0, 0.0, 2.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 2.0, 1.0],
            [0.0, 1.0, 2.0, 0.0, 4.0],
            [2.0, 0.0, 1.0, 4.0, 0.0],
        ]
    )
    targets = jnp.array([0.0, 3.0, 5.0, 1.5, 4.0])
    train_indices = np.array([0, 1, 2])
    test_indices = np.array([3, 4])
    rmse = train_test_knn(
        distance_matrix, targets, train_indices, test_indices, k, weighting
    )

    print(f"RMSE: {rmse}")
    if k == 2 and weighting == "uniform":
        assert jnp.allclose(rmse, 0)
