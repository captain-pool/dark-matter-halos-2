import numpy as np
# ----------------------------
# Verify the following:
# Code for rescaling the velocities and positions to be unitless

H_0 = 0.07  # km / s ⋅ kpc
OMEGA_m = 0.27  # dimensionless
G = 4.3 * 1e-6  # kpc m_{sun}^{-1} km^2 s^{-2}
rho_0 = OMEGA_m * (3 * (H_0 * H_0)) / (8 * np.pi * G)


def R_200(M_200: np.ndarray):
    # Mass in solar masses
    return np.float_power((3 / 4 * np.pi) * (M_200 / (200 * rho_0)), 1 / 3)


def V_200(M_200: np.ndarray):
    return np.sqrt(G * M_200 / R_200(M_200))


def make_positions_dimensionless(positions: np.ndarray, M_200: np.ndarray):
    return positions / R_200(M_200)[:, None, None]


def make_velocities_dimensionless(velocities: np.ndarray, M_200: np.ndarray):
    return velocities / V_200(M_200)[:, None, None]


def make_dimensionless(
    positions: np.ndarray, velocities: np.ndarray, M_200: np.ndarray
):
    return make_positions_dimensionless(
        positions, M_200
    ), make_velocities_dimensionless(velocities, M_200)


def rescale_by_dispersion(points: np.ndarray):
    return points / np.sqrt(np.sum(np.var(points, axis=1), axis=-1))[:, None, None]


# ----------------------------
