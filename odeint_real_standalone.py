import time
from typing import Tuple

import numpy as np
from numba import cfunc, float64, jit, njit
from numbalsoda import dop853, lsoda, lsoda_sig
from scipy.integrate import odeint

M_SUN = 2.0e33
a_d = 3.0
b_d = 0.28
M_d = 6.8e10 * M_SUN  # Disk+halo mass in [g].
M_b = 5.0e9 * M_SUN  # Bulge mass in [g].
r_b = 1.0
M_n = 1.71e9 * M_SUN  # Nucleus mass in [g].
r_n = 0.07
M_h = 5.4e11 * M_SUN  # Halo mass in [g].
r_h = 15.62

G = 6.67e-8
KPC_TO_CM = 3.08567758e21
YR_TO_S = 3600.0 * 24.0 * 365.0
G_KPC_YR = G / (KPC_TO_CM**3) * YR_TO_S**2


@njit(
    float64[:](
        float64,
    )
)
def shape_parameter(z: float) -> np.ndarray:

    sqrt = np.sqrt(z * z + b_d * b_d)

    K = a_d + sqrt

    dK_dz = z / sqrt

    return np.array([K, dK_dz])


@njit(
    float64(
        float64,
    )
)
def r_derivative_b_potential(r: float) -> float:

    dpot_b_dr = G_KPC_YR * M_b * (r + r_b) ** (-2)

    return dpot_b_dr


@njit(
    float64(
        float64,
    )
)
def r_derivative_n_potential(r: float) -> float:

    dpot_n_dr = G_KPC_YR * M_n * (r + r_n) ** (-2)

    return dpot_n_dr


@njit(
    float64(
        float64,
    )
)
def r_derivative_h_potential(r: float) -> float:

    dpot_h_dr = (
        G_KPC_YR * M_h / r * (1.0 / r * np.log1p(r / r_h) - 1.0 / (r_h + r))
    )

    return dpot_h_dr


@njit(
    float64[:](
        float64,
        float64,
    )
)
def r_z_derivatives_d_potential(r: float, z: float) -> np.ndarray:

    res = shape_parameter(z)
    K = res[0]
    dK_dz = res[1]

    sqrt = (r * r + K * K) ** (-1.5)

    dpot_d_dr = G_KPC_YR * M_d * r * sqrt
    dpot_d_dz = G_KPC_YR * M_d * sqrt * K * dK_dz

    return np.array([dpot_d_dr, dpot_d_dz])


@njit(
    float64[:](
        float64,
        float64,
    )
)
def cylind_coord_gradient_mw_potential(r: float, z: float) -> np.ndarray:

    res = r_z_derivatives_d_potential(r, z)
    dpot_d_dr = res[0]
    dpot_d_dz = res[1]
    dpot_b_dr = r_derivative_b_potential(r)
    dpot_n_dr = r_derivative_n_potential(r)
    dpot_h_dr = r_derivative_h_potential(r)

    dpot_mw_dr = dpot_d_dr + dpot_b_dr + dpot_n_dr + dpot_h_dr
    dpot_mw_dphi = 0.0
    dpot_mw_dz = dpot_d_dz

    return np.array([dpot_mw_dr, dpot_mw_dphi, dpot_mw_dz])


@njit(
    float64[:](
        float64,
        float64[:],
    )
)
def dynamical_eq_system(t: float, initial_cond: np.ndarray) -> np.ndarray:

    r = initial_cond[0]
    z = initial_cond[2]

    gradient_mw_pot = cylind_coord_gradient_mw_potential(r, z)

    # First derivatives.
    dr_dt = initial_cond[3]
    dphi_dt = initial_cond[4]
    dz_dt = initial_cond[5]

    # Second derivatives.
    d2r_dt2 = r * dphi_dt * dphi_dt - gradient_mw_pot[0]
    d2phi_dt2 = -2 * dr_dt * dphi_dt / r - gradient_mw_pot[1]
    d2z_dt2 = -gradient_mw_pot[2]

    derivatives = np.array(
        [dr_dt, dphi_dt, dz_dt, d2r_dt2, d2phi_dt2, d2z_dt2]
    )

    return derivatives


def odeint_call(t0: np.ndarray, u0: np.ndarray) -> np.ndarray:

    solution = odeint(dynamical_eq_system, y0=u0, t=t0, tfirst=True)

    return np.array(solution)


dt = 10000.0
tinit = 0.0

# Stiff
tend = 1.4708565048135614e7
initial_cond = [
    0.1021811020701645,
    3.988563758793204,
    -0.007834562002312783,
    -1.4791582530956602e-7,
    3.8563040038054056e-9,
    1.1408224933522102e-7,
]

time_grid = np.append(
    np.arange(tinit, tend, dt),
    tend,
)
odeint_call(time_grid, initial_cond)


print("Stiff")
start = time.time()
solution = odeint_call(time_grid, initial_cond)
end = time.time()
print(end - start)
print(solution[-1, :])

# No Stiff
tend = 2.163061968120212e7
initial_cond = [
    15.511847064292592,
    0.7326998994038764,
    0.1215349250221546,
    -1.268459376157434e-7,
    4.823630005080897e-9,
    1.7382186419146915e-8,
]

time_grid = np.append(
    np.arange(tinit, tend, dt),
    tend,
)
print("No Stiff")
start = time.time()
solution = odeint_call(time_grid, initial_cond)
end = time.time()
print(end - start)
print(solution[-1, :])


# Super Stiff
tend = 9.165803867040945e6
initial_cond = [
    5.848829707603326e-5,
    0.41856359395982956,
    -0.16966742967708984,
    1.1826612509365937e-7,
    0.00011148690823809339,
    8.199437113692513e-8,
]

time_grid = np.append(
    np.arange(tinit, tend, dt),
    tend,
)
print("Super Stiff")
start = time.time()
solution = odeint_call(time_grid, initial_cond)
end = time.time()
print(end - start)
print(solution[-1, :])
