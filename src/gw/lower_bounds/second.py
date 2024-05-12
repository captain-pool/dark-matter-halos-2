from jax import numpy as jnp, jit
import jax
import numpy as np
from typing import Callable

cost_matrix: Callable[[jnp.ndarray], jnp.ndarray] = lambda pts: jnp.linalg.norm(
    pts[:, None] - pts[None, :], axis=-1
)


def fix_torus(points: jnp.ndarray):
    coords_to_fix = jnp.any(points < 1e3, axis=0)
    for coord_to_fix in np.arange(3)[coords_to_fix]:

        shift = 75_000
        split = shift / 2

        to_shift = points[:, coord_to_fix] > split

        points = points.at[to_shift, coord_to_fix].set(
            points[to_shift, coord_to_fix] - shift
        )
    return points


def bfill(arr: jnp.ndarray):
    """
    Backfills nan's in a monotonically increasing array.
    """
    arr = jax.lax.scan(
        f=lambda carry, x: (jnp.minimum(carry, x), jnp.minimum(carry, x)),
        init=jnp.inf,
        xs=jnp.nan_to_num(arr, nan=jnp.inf),
        reverse=True,
    )[-1]

    # The `arr` computed above has 1e38 in place of jnp.inf,
    # This where condition turns it back into a jnp.nan
    # I think there should be a better solution (in fixing the scan, but I don't see it)
    return jnp.where(arr >= 1e35, jnp.nan, arr)


def pq_wasserstein(
    space_a: jnp.ndarray,
    space_b: jnp.ndarray,
    weights_a: jnp.ndarray,
    weights_b: jnp.ndarray,
    p: float,
    q: float,
):
    """
    Computes the `(p, q)`-One Dimensional Wasserstein distance on the positive Real line.
    Assumes that the points in the input spaces are *already sorted*.
    """
    cdf_a = jnp.cumsum(weights_a)
    cdf_b = jnp.cumsum(weights_b)

    all_cdfs = jnp.concatenate([cdf_a, cdf_b])
    all_cdfs_sorter = jnp.argsort(all_cdfs)
    all_cdfs_sorted = jnp.concatenate([jnp.zeros([1]), all_cdfs[all_cdfs_sorter]])
    diffs = jnp.diff(all_cdfs_sorted)[:-1]

    cdf_a_inv = bfill(
        jnp.concatenate([space_a, jnp.full_like(space_b, jnp.nan)])[all_cdfs_sorter]
    )[:-1]
    cdf_b_inv = bfill(
        jnp.concatenate([jnp.full_like(space_a, jnp.nan), space_b])[all_cdfs_sorter]
    )[:-1]

    def lambda_q(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        return jnp.abs((a**q) - (b**q)) ** (1 / q)

    integrand = lambda_q(cdf_a_inv, cdf_b_inv) ** p

    integral = integrand * diffs

    return jnp.sum(integral) ** (1 / p)


def quick_reduce(points: jnp.ndarray, weights: jnp.ndarray, quantiles: jnp.ndarray):
    eps = 1e-6
    points_sorter = jnp.argsort(points)

    weights_sorted = weights[points_sorter]
    cdf = jnp.concatenate([jnp.zeros([1]), jnp.cumsum(weights_sorted)])

    # Subtract an epsilon from quantiles to ensure that they land on the left of the
    # intended boundary:
    indices = jnp.searchsorted(cdf, quantiles - eps, side="left")

    return points[points_sorter][indices]


def quick_reduce_cost_matrix(
    cost_matrix: jnp.ndarray, weights: jnp.ndarray, quantiles: jnp.ndarray
):
    # weights_matrix = weights[:, None] * weights[None, :]
    return jax.vmap(quick_reduce, (0, None, None))(
        cost_matrix,
        weights,
        quantiles,
    )

@jax.jit
def global_dod_approx(
    cost_matrix: jnp.ndarray, weights: jnp.ndarray, quantiles: jnp.ndarray
):
    costs = quick_reduce_cost_matrix(
        cost_matrix,
        weights,
        quantiles,
    ).flatten()
    costs_sorter = costs.argsort()
    cost_weights = (
        weights[:, None] * jnp.ones_like(quantiles)[None, :] / quantiles.shape[0]
    ).flatten()

    costs_sorted = costs[costs_sorter]
    cost_weights_sorted = cost_weights[costs_sorter]

    return costs_sorted, cost_weights_sorted


def second_lower_bound_approx(
    cost_matrix_a: jnp.ndarray,
    cost_matrix_b: jnp.ndarray,
    weights_a: jnp.ndarray,
    weights_b: jnp.ndarray,
    p: float,
    q: float,
    quantiles: jnp.ndarray = jnp.linspace(0, 1, 100, endpoint=False),
):
    """
    Computes the `(p, q)`-Second Lower Bound from the paper.
    We omit the leading factor of 1/2 for now.
    """

    costs_a_sorted, cost_weights_a_sorted = global_dod_approx(
        cost_matrix_a, weights_a, quantiles
    )
    costs_b_sorted, cost_weights_b_sorted = global_dod_approx(
        cost_matrix_b, weights_b, quantiles
    )

    dist = pq_wasserstein(
        costs_a_sorted,
        costs_b_sorted,
        cost_weights_a_sorted,
        cost_weights_b_sorted,
        p=p,
        q=q,
    )
    return dist


def global_dod(cost_matrix: jnp.ndarray, weights: jnp.ndarray):
    costs = cost_matrix.flatten()
    costs_sorter = costs.argsort()
    cost_weights = (weights[:, None] * weights[None, :]).flatten()

    costs_sorted = costs[costs_sorter]
    cost_weights_sorted = cost_weights[costs_sorter]

    return costs_sorted, cost_weights_sorted


def second_lower_bound(
    cost_matrix_a: jnp.ndarray,
    cost_matrix_b: jnp.ndarray,
    weights_a: jnp.ndarray,
    weights_b: jnp.ndarray,
    p: float,
    q: float,
):
    """
    Computes the `(p, q)`-Second Lower Bound from the paper.
    We omit the leading factor of 1/2 for now.
    """

    costs_a_sorted, cost_weights_a_sorted = global_dod(cost_matrix_a, weights_a)
    costs_b_sorted, cost_weights_b_sorted = global_dod(cost_matrix_b, weights_b)

    dist = pq_wasserstein(
        costs_a_sorted,
        costs_b_sorted,
        cost_weights_a_sorted,
        cost_weights_b_sorted,
        p=p,
        q=q,
    )
    return dist
