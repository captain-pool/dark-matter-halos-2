from matplotlib import pyplot as plt
from jax import numpy as jnp
from jax.typing import ArrayLike
from typing import Optional
from itertools import product
import pandas as pd


def oat_validate_knn(
    train_distance_matrix: ArrayLike,
    train_targets: ArrayLike,
    k: int = 4,
    weighting: str = "uniform",
) -> float:
    tol = 1e-8
    # Take nearest k points that are not the point itself:
    nearest_k_pts = train_distance_matrix.argsort(axis=-1)[:, 1 : k + 1]
    values = train_targets[nearest_k_pts]
    match weighting:
        case "inverse_sqrt":
            distances = jnp.sort(train_distance_matrix, axis=-1)[:, 1 : k + 1]
            weights = 1 / (jnp.sqrt(distances) + tol)
            weights = weights / weights.sum(axis=-1, keepdims=True)
        case "inverse_square":
            distances = jnp.sort(train_distance_matrix, axis=-1)[:, 1 : k + 1]
            weights = 1 / (jnp.square(distances) + tol)
            weights = weights / weights.sum(axis=-1, keepdims=True)
        case "inverse":
            distances = jnp.sort(train_distance_matrix, axis=-1)[:, 1 : k + 1]
            weights = 1 / (distances + tol)
            weights = weights / weights.sum(axis=-1, keepdims=True)
        case "uniform":
            weights = jnp.ones(k) / k

        case _:
            raise ValueError(f"Weight method {weighting} unknown")

    residuals = (values * weights).sum(axis=-1) - train_targets

    # returning RMSE:
    return jnp.sqrt(jnp.mean(jnp.square(residuals))).item()


def oat_cross_validate(
    train_distance_matrix: ArrayLike,
    train_targets: ArrayLike,
    params: Optional[dict[str, any]] = None,
):
    if params is None:
        k = list(range(1, 30))
        weighting = ["uniform", "inverse", "inverse_square", "inverse_sqrt"]
        params = {"k": k, "weighting": weighting}
    keys = params.keys()
    params_list = list(product(*params.values()))
    losses = [
        oat_validate_knn(train_distance_matrix, train_targets, **dict(zip(keys, param)))
        for param in params_list
    ]

    output = pd.DataFrame(params_list, columns=keys).assign(loss=losses)
    return output


def plot_cv_output(cv_output: pd.DataFrame, ax: Optional[plt.Axes] = None):
    if ax is None:
        _fig, ax = plt.subplots()

    try:
        assert "k" in cv_output.columns
    except AssertionError:
        raise AssertionError(
            "Cross validation not done over various k (required for plotting)"
        )

    try:
        assert "loss" in cv_output.columns
    except AssertionError:
        raise AssertionError("Losses missing from dataframe")

    other_params = list(set(cv_output.columns).difference(set(["k", "loss"])))
    for param_choice, df in cv_output.groupby(other_params):
        ax.plot(
            df["k"], df["loss"], label=str(dict(zip(other_params, param_choice)))[1:-1]
        )

    ax.legend()


def plot_oat_cross_validate(
    train_distance_matrix: ArrayLike,
    train_targets: ArrayLike,
    params: Optional[dict[str, any]] = None,
):
    _fig, ax = plt.subplots(figsize=(10, 6))
    output = oat_cross_validate(train_distance_matrix, train_targets, params)
    plot_cv_output(output, ax)
    ax.set(
        xlabel="k", ylabel="Root Mean Squared Error", title="O.A.T. Cross Validation"
    )
