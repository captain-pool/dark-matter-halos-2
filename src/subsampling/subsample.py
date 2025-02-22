from sklearn.cluster import KMeans
from tqdm.notebook import tqdm
import numpy as np



def kmeans_downsample(
    point_cloud: np.ndarray, velocity: np.ndarray, tau: float, downsample_size: int, n_trials: int, pbar: bool = False
):
    phase_space_points = np.concatenate([point_cloud, tau * velocity], axis=1)
    sampled_positions = np.zeros([n_trials, downsample_size, 3])
    sampled_velocities = np.zeros([n_trials, downsample_size, 3])
    sampled_weights = np.zeros([n_trials, downsample_size])

    progress_bar = tqdm(range(n_trials)) if pbar else range(n_trials)
    for t in progress_bar:
        kmeans = KMeans(n_clusters=downsample_size, n_init="auto", random_state=t)
        kmeans.fit(phase_space_points)
        
        _labels, counts = np.unique(kmeans.labels_, return_counts=True)
        weights = counts / counts.sum()

        centers = kmeans.cluster_centers_
        sampled_positions[t] = centers[:, :3]
        sampled_velocities[t] = centers[:, 3:]
        sampled_weights[t] = weights

    return sampled_positions, sampled_velocities, sampled_weights

def kmeans_downsample_points(
    points_list: list[np.ndarray], velocities_list: list[np.ndarray], taus: list[float], downsample_size: int, n_trials: int, pbar: bool = False
):
    sampled_positions_all = []
    sampled_velocities_all = []
    sampled_weights_all = []

    for tau in taus:
        sampled_positions_tau = np.zeros([len(points_list), n_trials, downsample_size, 3])
        sampled_velocities_tau = np.zeros([len(points_list), n_trials, downsample_size, 3])
        sampled_weights_tau = np.zeros([len(points_list), n_trials, downsample_size])

        progress_bar = tqdm(list(enumerate(points_list)), desc=f"Tau={tau}") if pbar else enumerate(points_list)
        for i, point_cloud in progress_bar:
            positions, velocities, weights = kmeans_downsample(
                point_cloud, velocities_list[i], tau, downsample_size, n_trials, pbar=False
            )
            sampled_positions_tau[i] = positions
            sampled_velocities_tau[i] = velocities
            sampled_weights_tau[i] = weights

        sampled_positions_all.append(sampled_positions_tau)
        sampled_velocities_all.append(sampled_velocities_tau)
        sampled_weights_all.append(sampled_weights_tau)

    return sampled_positions_all, sampled_velocities_all, sampled_weights_all



def uniform_downsample(point_cloud, downsample_size, n_trials):
    sampled_weights = np.ones([n_trials, downsample_size]) / downsample_size

    rng = np.random.default_rng(seed=0)

    sampled_point_clouds = rng.choice(
        point_cloud, size=[n_trials, downsample_size], axis=0
    )

    return sampled_point_clouds, sampled_weights





def fix_torus(pts: np.ndarray):
    points = pts.copy()
    coords_to_fix = np.any(points < 1e3, axis=0)
    for coord_to_fix in np.arange(3)[coords_to_fix]:

        shift = 75_000
        split = shift / 2

        to_shift = points[:, coord_to_fix] > split

        points[to_shift, coord_to_fix] = points[to_shift, coord_to_fix] - shift
    return points


def preprocess_pointcloud(pts: np.ndarray):
    points = pts.copy()
    points = fix_torus(points)
    return points


def aggregate_velocities(
    points: np.ndarray, velocities: np.ndarray, subsampled_points: np.ndarray
) -> np.ndarray:
    labels = np.argmin(
        np.linalg.norm(points[None, :] - subsampled_points[:, None], axis=-1), axis=0
    )
    labels_sorter = np.argsort(labels)
    labels_sorted = labels[labels_sorter]

    velocites_sorted = velocities[labels_sorter]

    # Might have an off-by-one error
    label_start_indices = np.concatenate(
        [np.array([0]), np.argwhere(np.diff(labels_sorted))[:, 0]]
    )

    label_counts = np.diff(
        np.concatenate([label_start_indices, np.array([labels.shape[0] - 1])])
    )
    cluster_velocities = (
        np.add.reduceat(velocites_sorted, label_start_indices, 0)
        / label_counts[:, None]
    )
    return cluster_velocities


def recenter(V: np.ndarray) -> np.ndarray:
    return V - V.mean(axis=0)
