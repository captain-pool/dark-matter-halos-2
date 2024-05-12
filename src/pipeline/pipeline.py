from dataclasses import dataclass
import datetime
import os
from pathlib import Path
from typing import Literal
import jax.numpy as jnp
import jax
from tqdm.notebook import tqdm
from src.gw.lower_bounds import second
from src.summarize.oat import oat_validate_knn
import numpy as np
from itertools import product
import pickle
from src.pipeline.rescale import *


def slb_func(p: float, q: float, scan: bool = False):
    def compute_slb(
        cloud_a: jnp.ndarray,
        cloud_b: jnp.ndarray,
        weights_a: jnp.ndarray,
        weights_b: jnp.ndarray,
    ):
        cost_matrix_a = second.cost_matrix(cloud_a)
        cost_matrix_b = second.cost_matrix(cloud_b)

        return second.second_lower_bound(
            cost_matrix_a, cost_matrix_b, weights_a, weights_b, p, q
        )

    get_slb_dists_with = jax.vmap(compute_slb, [0, None, 0, None])
    # In some tests, not scanning was 6x faster than scanning...
    if not scan:
        get_slb_dists = jax.vmap(get_slb_dists_with, [None, 0, None, 0])
        return get_slb_dists

    get_slb_dists = lambda clouds_a, clouds_b, weights_a, weights_b: jax.lax.scan(
        lambda carry, cloud_b_weight_b: (
            carry,
            get_slb_dists_with(
                clouds_a, cloud_b_weight_b[0], weights_a, cloud_b_weight_b[1]
            ),
        ),
        init=0,
        xs=(clouds_b, weights_b),
    )
    return get_slb_dists


def batched_slb_func(p: float, q: float, batch_size: int, pbar: bool = True):
    """Returns the function that produces the (p, q)-Second Lower Bound of the
    Gromov-Wasserstein distance between arrays of points clouds and their
    weights.

    The function allows for splitting the computations along one (of two) axes
    into smaller batches if you are running into memory issues.

    An alternative is to scan the output of `get_slb_func_with` along the final
    axis, but I found that this performs significantly more slowly than just a
    the for loop over the vmap. I think this is because the scan doesn't get to
    fully use all the memory on the gpu, while this lets you dynamically scale
    up the size of a batch to use all the memory on the gpu.
    """
    get_slb_dists = slb_func(p, q, scan=False)

    def batched_get_slb_dists(
        clouds_a: jnp.ndarray,
        clouds_b: jnp.ndarray,
        weights_a: jnp.ndarray,
        weights_b: jnp.ndarray,
    ):
        n_batches = (clouds_b.shape[0] // batch_size) + 1
        _pbar = tqdm(range(n_batches)) if pbar else range(n_batches)
        return jnp.concatenate(
            [
                get_slb_dists(
                    clouds_a,
                    clouds_b[batch_ix * batch_size : (batch_ix + 1) * batch_size],
                    weights_a,
                    weights_b[batch_ix * batch_size : (batch_ix + 1) * batch_size],
                )
                for batch_ix in _pbar
            ],
            axis=0,
        )

    return batched_get_slb_dists


def test_batched_slb_func(
    p: float,
    q: float,
    batch_size: int,
    points_subsampled: jnp.ndarray,
    weights_subsampled: jnp.ndarray,
) -> None:
    slb_dists = batched_slb_func(p, q, batch_size=batch_size)(
        points_subsampled,
        points_subsampled,
        weights_subsampled,
        weights_subsampled,
    )

    assert jnp.allclose(
        slb_dists,
        slb_func(p, q)(
            points_subsampled,
            points_subsampled,
            weights_subsampled,
            weights_subsampled,
        ),
    )
    print("Test Passed! :)")


def combine_dists(
    alphas: jnp.ndarray,
    distance_matrices: list[jnp.ndarray],
    normalize_by_std: bool = True,
):
    """Combines distance matrices so that the new distance is:
    `d^2 = alphas_1 * d_1^2 + ... + alphas_n * d_n^2`
    """
    try:
        assert len(alphas) == len(distance_matrices)
    except AssertionError:
        raise AssertionError(
            f"Number of parameters ({len(alphas)}) not equal to number of distance matrices ({len(distance_matrices)})"
        )

    stacked_distance_matrices = jnp.stack(distance_matrices, axis=0)

    if normalize_by_std:
        stds = jnp.std(
            stacked_distance_matrices.reshape(stacked_distance_matrices.shape[0], -1),
            axis=1,
        )
        prec = jnp.nan_to_num(1 / stds, posinf=0.0)
        stacked_distance_matrices *= prec[:, None, None]

    stacked_distance_matrices *= alphas[:, None, None]

    return jnp.sqrt(jnp.sum(jnp.square(stacked_distance_matrices), 0))


def test_combine_dists(
    distance_matrices: list[jnp.ndarray],
):
    """Tests `combine_dists` by comparing the combined distances with `alpha`
    being the Kronecker δ to the corresponding distance matrix"""
    for i, distance_matrix in enumerate(distance_matrices):
        alphas = jnp.full(len(distance_matrices), 0.0)
        alphas = alphas.at[i].set(1.0)

        # Checking the normalization case:
        combined_distance_matrix = combine_dists(
            alphas, distance_matrices, normalize_by_std=True
        )
        assert np.allclose(
            combined_distance_matrix, distance_matrix / distance_matrix.flatten().std()
        )

        # Checking the no-normalization case:
        combined_distance_matrix = combine_dists(
            alphas, distance_matrices, normalize_by_std=False
        )
        assert np.allclose(combined_distance_matrix, distance_matrix)

    print("All tests passed :)")


# Should probably name this class better:
@dataclass
class ProblemContext:
    """
    The setting for an instance of our problem. Stuff is easier if I just pass
    one of these objects around. Contains the data:

    Parameters
    ----------
    points: `jnp.ndarray`
        The pointclouds for each halo. Has shape `[n_halos, n_points_per_halo, 3]`

    weights: `jnp.ndarray`
        The weights corresponding to the points in each halo. Has shape
        `[n_halos, n_points_per_halo]`

    velocities: `jnp.ndarray`
        The velocities corresponding to the points in each halo. Has shape
        `[n_halos, n_points_per_halo, 3]`

    masses: `jnp.ndarray`
        The overall masses (M_200) of each halo. Has shape `[n_halos,]`

    concentrations: `jnp.ndarray`
        The concentrations (Subhalo C_200) of each halo. Has shape `[n_halos,]`

    labels: `jnp.ndarray`
        The target parameters to learn. One label per halo, so has shape
        `[n_halos,]`. Examples of possible labels include Stellar Mass or Stellar
        Metallicity.
    """

    points: jnp.ndarray
    weights: jnp.ndarray
    velocities: jnp.ndarray
    masses: jnp.ndarray
    concentrations: jnp.ndarray
    labels: jnp.ndarray


class Hyperparametrization:
    """
    The set of hyperparameters of which to explore the product space.

    """

    def __init__(
        self,
        rescale_strategy: list[Literal["unitless", "dispersion", "none"]],
        p: float | np.ndarray,
        q: float | np.ndarray,
        tau: float | np.ndarray,
        alpha_C: float | np.ndarray,
        alpha_M: float | np.ndarray,
        alpha_SLB: float | np.ndarray,
        n_neighbors: int | np.ndarray[int],
    ):
        self.rescale_strategy = rescale_strategy
        self.p: np.ndarray = np.array(p).flatten()
        self.q: np.ndarray = np.array(q).flatten()
        self.tau: np.ndarray = np.array(tau).flatten()
        self.alpha_C: np.ndarray = np.array(alpha_C).flatten()
        self.alpha_M: np.ndarray = np.array(alpha_M).flatten()
        self.alpha_SLB: np.ndarray = np.array(alpha_SLB).flatten()
        self.n_neighbors: np.ndarray[int] = np.array(n_neighbors, dtype=int).flatten()

    @property
    def params(self):
        return [
            self.rescale_strategy,
            self.p,
            self.q,
            self.tau,
            self.alpha_C,
            self.alpha_M,
            self.alpha_SLB,
            self.n_neighbors,
        ]

    @property
    def params_lengths(self):
        return [len(p) for p in self.params]

    @property
    def first_layer_params(self):
        return product(
            self.rescale_strategy,
            self.p,
            self.q,
            self.tau,
        )

    @property
    def second_layer_params(self):
        return product(
            self.alpha_C,
            self.alpha_M,
            self.alpha_SLB,
        )

    @property
    def third_layer_params(self):
        return self.n_neighbors


def get_losses(
    problem_context: ProblemContext,
    hyperparametrization: Hyperparametrization,
    pbar=True,
):
    losses = []

    _pbar = (
        tqdm(list(hyperparametrization.first_layer_params))
        if pbar
        else hyperparametrization.first_layer_params
    )
    for rescale_strategy, p, q, tau in _pbar:
        if rescale_strategy == "unitless":
            positions_rescaled, velocities_rescaled = make_dimensionless(
                problem_context.points,
                problem_context.velocities,
                problem_context.masses,
            )
        if rescale_strategy == "dispersion":
            positions_rescaled = rescale_by_dispersion(problem_context.points)
            velocities_rescaled = rescale_by_dispersion(problem_context.velocities)

        if rescale_strategy == "none":
            positions_rescaled = problem_context.points
            velocities_rescaled = problem_context.velocities
        phase_points = jnp.concatenate(
            [
                positions_rescaled,
                tau * velocities_rescaled,
            ],
            axis=-1,
        )

        slb_dists = batched_slb_func(
            p, q, batch_size=5000 // problem_context.masses.shape[0], pbar=False
        )(phase_points, phase_points, problem_context.weights, problem_context.weights)

        concentration_dists = jnp.abs(
            problem_context.concentrations[:, None]
            - problem_context.concentrations[None, :]
        )
        mass_dists = jnp.abs(
            problem_context.masses[:, None] - problem_context.masses[None, :]
        )

        for (
            alpha_C,
            alpha_M,
            alpha_SLB,
        ) in hyperparametrization.second_layer_params:
            blended_dists = combine_dists(
                jnp.array([alpha_C, alpha_M, alpha_SLB]),
                [concentration_dists, mass_dists, slb_dists],
                normalize_by_std=True,
            )

            for n_neighbors in hyperparametrization.third_layer_params:
                losses.append(
                    {
                        "loss": oat_validate_knn(
                            blended_dists,
                            problem_context.labels,
                            k=n_neighbors,
                            weighting="uniform",
                        ),
                        "parameters": {
                            "rescale_strategy": rescale_strategy,
                            "p": p,
                            "q": q,
                            "tau": tau,
                            "alpha_C": alpha_C,
                            "alpha_M": alpha_M,
                            "alpha_SLB": alpha_SLB,
                            "n_neighbors": n_neighbors,
                        },
                    }
                )

    loss_vals = [loss["loss"] for loss in losses]
    loss_array = (
        np.array(loss_vals).reshape(*hyperparametrization.params_lengths).squeeze()
    )
    return losses, loss_array


def save_results(
    results_path: Path,
    losses: list,
    mass_range: tuple[float],
    label_name: str,
    loss_array: np.ndarray,
):
    results_prev = os.listdir(results_path)
    now = str(datetime.datetime.now().date())
    prev_results_today = sum([f.startswith(now) for f in results_prev])

    pickle.dump(
        {"losses": losses, "mass_range": mass_range, "loss_array": loss_array},
        open(
            results_path
            / f"{now}_losses_run{prev_results_today}_label_{label_name}.pkl",
            "wb",
        ),
    )
