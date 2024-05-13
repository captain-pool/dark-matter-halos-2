from dataclasses import dataclass
import os
from pathlib import Path
from matplotlib import patches, pyplot as plt
from typing import Optional
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

square_or_cube_root = "square"
power = {"square": 2, "cube": 3}[square_or_cube_root]


@dataclass
class HaloInfo:
    all_points: np.ndarray
    all_velocities: np.ndarray
    subsampled_points: np.ndarray
    subsampled_velocities: np.ndarray
    subsampled_weights: np.ndarray


def recenter(V: np.ndarray) -> np.ndarray:
    return V - np.mean(V, axis=0)


def normalize(v: np.ndarray, axis: int = -1) -> np.ndarray:
    return v / np.linalg.norm(v, axis=axis)


def r(point_cloud: np.ndarray) -> np.ndarray:
    center = point_cloud.mean(axis=0)
    r = point_cloud - center
    return r


def angular_momenta(point_cloud: np.ndarray, velocities: np.ndarray) -> np.ndarray:
    center = point_cloud.mean(axis=0)
    r = point_cloud - center
    return np.cross(r, velocities)


def angular_momentum(point_cloud: np.ndarray, velocities: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
    if weights is None:
        return angular_momenta(point_cloud, velocities).mean(0)
    return (angular_momenta(point_cloud, velocities) * weights[:, None]).sum(0)


def get_2d_slice(
    point_cloud: np.ndarray, x_direction: np.ndarray, y_direction: np.ndarray
) -> np.ndarray:
    conversion_matrix = np.stack([x_direction, y_direction], axis=-1)
    return point_cloud @ conversion_matrix


def velocity_triangle(
    velocity: np.ndarray, point: np.ndarray, weight: float, color: str = "C0"
):
    tip = velocity

    normal_vector = normalize(np.flip(velocity)) * np.array([1.0, -1.0])
    base_1 = weight * normal_vector / 2
    base_2 = -(weight * normal_vector) / 2

    return patches.Polygon(
        point[None, :] + (np.stack([tip, base_1, base_2], axis=0)), color=color
    )


def comet_tail(
    velocity: np.ndarray, point: np.ndarray, weight: float, color: str = "C0"
):
    tip = -velocity

    normal_vector = normalize(np.flip(velocity)) * np.array([1.0, -1.0])
    base_1 = weight * normal_vector / 2
    base_2 = -(weight * normal_vector) / 2

    return patches.Polygon(
        point[None, :] + (np.stack([tip, base_1, base_2], axis=0)), color=color
    )


def plot_at_x_y(
    halo_info: HaloInfo,
    x: np.ndarray,
    y: np.ndarray,
    ax: plt.Axes,
    z: Optional[np.ndarray] = None,
    **kwargs,
):
    rel_size: float = kwargs["rel_size"] if "rel_size" in kwargs.keys() else 1.0
    scale_with_z_order: bool = (
        kwargs["scale_with_z_order"] if "scale_with_z_order" in kwargs.keys() else False
    )
    tail_color: tuple[float] = kwargs["tail_color"] if "tail_color" in kwargs.keys() else (0.3, 0.3, 0.3, 0.9)

    points_sliced = get_2d_slice(halo_info.all_points, x, y)
    points_subsampled_sliced = get_2d_slice(halo_info.subsampled_points, x, y)
    velocities_subsampled_sliced = get_2d_slice(halo_info.subsampled_velocities, x, y)
    radii = 2 * np.power(halo_info.subsampled_weights * rel_size, 1 / power)

    all_points = np.concatenate([halo_info.all_points, halo_info.subsampled_points])
    all_points_sliced = get_2d_slice(all_points, x, y)

    all_sizes = np.concatenate(
        [
            0.15 * np.ones(points_sliced.shape[0]),
            400 * np.square(radii) * np.pi,
        ]
    )
    colors = np.concatenate(
        [
            np.ones(points_sliced.shape[0])[:, None]
            * np.array([0.6, 0.6, 0.6, 0.9])[None, :],
            np.ones(points_subsampled_sliced.shape[0])[:, None]
            * np.array([0.0, 0.0, 0.0, 0.9])[None, :],
        ],
        axis=0,
    )

    z_order = np.ones(all_points.shape[0]) if z is None else (all_points @ z)
    order = np.argsort(z_order)

    if scale_with_z_order:
        z_scale = 1 / np.abs(z_order + 3)
        all_sizes *= z_scale

    subsampled_indices = np.argwhere(
        np.concatenate(
            [
                np.zeros(halo_info.all_points.shape[0]),
                np.ones(halo_info.subsampled_points.shape[0]),
            ]
        )[order]
    )[:, 0]
    fence_posts = np.concatenate(
        [np.array([0]), subsampled_indices, np.array([all_points.shape[0]])]
    )

    subsampled_z_order = halo_info.subsampled_points @ z
    subsampled_order = np.argsort(subsampled_z_order)

    for ix in range(len(fence_posts) - 1):
        ax.scatter(
            all_points_sliced[order, 0][fence_posts[ix] : fence_posts[ix + 1]],
            all_points_sliced[order, 1][fence_posts[ix] : fence_posts[ix + 1]],
            s=all_sizes[order][fence_posts[ix] : fence_posts[ix + 1]],
            c=colors[order][fence_posts[ix] : fence_posts[ix + 1]],
        )

        tail = comet_tail(
            velocities_subsampled_sliced[subsampled_order[ix - 1]] / 2,
            points_subsampled_sliced[subsampled_order[ix - 1]],
            radii[subsampled_order[ix - 1]] / 20,
            color=tail_color,
        )
        ax.add_patch(tail)

    ax.axis("off")
    ax.set(aspect="equal")


def plot_from_above_perturbed(
    theta: float,
    halo_info: HaloInfo,
    ax: plt.Axes,
    **kwargs,
):
    center = np.mean(halo_info.all_points, axis=0, keepdims=True)
    points = halo_info.all_points - center
    subsampled_points = halo_info.subsampled_points - center
    z = normalize(angular_momentum(points, halo_info.all_velocities))
    points_in_plane = points - (points @ z)[:, None] * z[None, :]

    pca = PCA(n_components=3).fit(points_in_plane)
    x = pca.components_[0]
    y = pca.components_[1]

    max_radius = np.max(np.linalg.norm(points, axis=-1))
    ax.set(xlim=(-1, 1), ylim=(-1, 1))

    z_hat = z * np.cos(theta) + y * np.sin(theta)
    y_hat = -z * np.sin(theta) + y * np.cos(theta)
    plot_at_x_y(
        HaloInfo(
            all_points=points / max_radius,
            all_velocities=halo_info.all_velocities / max_radius,
            subsampled_points=subsampled_points / max_radius,
            subsampled_velocities=halo_info.subsampled_velocities / max_radius,
            subsampled_weights=halo_info.subsampled_weights,
        ),
        y_hat,
        x,
        ax,
        z=z_hat,
        **kwargs,
    )


def plot_save_stereogram(
    angle: float,
    theta: float,
    halo_info: HaloInfo,
    fname: str,
):
    fig, axs = plt.subplots(1, 2, figsize=(28, 16))
    plot_from_above_perturbed(
        angle,
        halo_info,
        axs[0],
    )
    plot_from_above_perturbed(
        angle + theta,
        halo_info,
        axs[1],
    )
    plt.tight_layout()
    fig.subplots_adjust(wspace=-0.3, hspace=0)
    plt.savefig(fname=fname)
    # return axs


def make_gif_frames(halo_info: HaloInfo, folder_path: Path, n_frames: int = 100):
    rotation_per_frame = 2 * np.pi / n_frames
    os.mkdir(folder_path)
    for i in tqdm(range(n_frames)):
        plot_save_stereogram(
            i * rotation_per_frame,
            0.1,
            halo_info,
            fname=folder_path / f"stereo_{str(i).zfill(3)}.png",
        )
        plt.close()
