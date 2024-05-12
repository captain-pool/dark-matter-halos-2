from ott.solvers.quadratic import gromov_wasserstein
from ott.problems.quadratic import quadratic_problem
from ott.solvers.quadratic import lower_bound
from ott.geometry import pointcloud
from typing import List
from jax.typing import ArrayLike
from pathlib import Path
import pickle

from sklearn.cluster import KMeans
import numpy as np
from tqdm.notebook import tqdm


def kmeans_subsample(points, subsample, random_state=0, include_labels=False):
    kmeans = KMeans(n_clusters=subsample, random_state=random_state, n_init="auto").fit(
        points
    )
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    weights = counts / counts.sum()
    if include_labels:
        return weights, pointcloud.PointCloud(kmeans.cluster_centers_), kmeans.labels_
    return weights, pointcloud.PointCloud(kmeans.cluster_centers_)


def uniform_subsample(points, subsample, random_state=0):
    rng = np.random.default_rng(random_state)
    idx = rng.choice(points.shape[0], subsample, replace=False)
    weights = np.ones(subsample) / subsample
    return weights, pointcloud.PointCloud(points[idx])


def subsample_cloud(points, subsample, random_state=0, subsample_method="kmeans"):

    points_rec = points - points.mean(axis=0, keepdims=True)
    points_norm = points_rec / 30

    match subsample_method:
        case "kmeans":
            weights, cloud = kmeans_subsample(
                points_norm, subsample, random_state=random_state
            )
        case "uniform":
            weights, cloud = uniform_subsample(
                points_norm, subsample, random_state=random_state
            )
        case _:
            raise ValueError(f"Unknown subsample method {subsample_method}")

    return weights, cloud


def trial(
    points_a,
    points_b,
    subsample,
    random_state=0,
    method="gw",
    subsample_method="kmeans",
):
    weights_a, cloud_a = subsample_cloud(
        points_a,
        subsample,
        random_state=random_state,
        subsample_method=subsample_method,
    )
    weights_b, cloud_b = subsample_cloud(
        points_b,
        subsample,
        random_state=random_state,
        subsample_method=subsample_method,
    )

    prob = quadratic_problem.QuadraticProblem(
        cloud_a, cloud_b, a=weights_a, b=weights_b
    )
    match method:
        case "gw":
            solver = gromov_wasserstein.GromovWasserstein(epsilon=0.002)
            soln = solver(prob)
        case "lb":
            solver = lower_bound.LowerBoundSolver(epsilon=0.002)
            soln = solver(prob)
        case _:
            raise ValueError(f"Unknown method {method}")
    return soln


def do_experiment(
    points_a, points_b, num_trials, subsample, method="gw", subsample_method="kmeans"
):
    solns = []
    for t in range(num_trials):
        solns.append(
            trial(
                points_a,
                points_b,
                subsample,
                random_state=t,
                method=method,
                subsample_method=subsample_method,
            )
        )
    return solns


def do_experiments(
    points_array: List[ArrayLike],
    mat_size: int,
    num_trials: int,
    subsample_size: int,
    pbar=False,
    subsample_method="kmeans",
):
    cost_matrix = np.zeros((mat_size, mat_size, num_trials))

    outer_progress_bar = tqdm(range(mat_size)) if pbar else range(mat_size)
    for i in outer_progress_bar:
        inner_progress_bar = tqdm(range(i, mat_size)) if pbar else range(i, mat_size)
        for j in inner_progress_bar:
            solns = do_experiment(
                points_array[i],
                points_array[j],
                num_trials,
                subsample=subsample_size,
                method="gw",
                subsample_method=subsample_method,
            )
            costs = [soln.primal_cost for soln in solns]
            cost_matrix[i, j] = costs
            cost_matrix[j, i] = costs

    return cost_matrix
