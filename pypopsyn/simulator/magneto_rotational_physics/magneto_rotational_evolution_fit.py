"""
    Combined evolution of the pulsar period, misalignment angle and magnetic field
    relying on fits to magneto-thermal simulations.

    Authors:

        Vanessa Graber (graber@ice.csic.es)
        Michele Ronchi (ronchi@ice.csic.es)
"""

import pathlib
from typing import Tuple

import numpy as np
import orjson
from numba import float64, jit
from scipy.integrate import odeint

import pypopsyn.simulator.initial_population_edm as ipop
import pypopsyn.simulator.magneto_rotational_physics.misalignment_angle_derivative as madv
import pypopsyn.simulator.magneto_rotational_physics.period_derivative as pdv
from pypopsyn.simulator.config_simulator import cfg


def magnetic_field_evolution_fit_numpy(
    B_initial: float,
    t: np.ndarray,
    B_asymptotic: float,
    a1: float,
    a2: float,
    A1: float,
    A2: float,
    b1: float,
    b2: float,
    tau_late: float,
    a_late: float,
) -> np.ndarray:
    """
    An analytical function for the magnetic field evolution curves from the magneto-thermal evolution simulations.
    This method is compatible with numpy arrays and is used if one wants to save the entire evolution output in the
    magneto_rotational_evolution method.
    The fit parameters for each magneto-thermal model specified in the config_simulator.py file were adjusted by hand
    (see the notebook tutorials/analysis_notebooks/magnetic_field_evolution_fit.ipynb for more details).

    Args:
        B_initial (float): Initial magnetic field strength in [G].
        t (np.ndarray): Time in [s].
        B_asymptotic (float): Asymptotic magnetic field strength at late times in [G].
        a1 (float): Power-law index for the first power-law component for the magnetic-field evolution fit.
        a2 (float): Power-law index for the second power-law component for the magnetic-field evolution fit.
        A1 (float): Normalization for the timescale parameter of the first power-law component for the
            magnetic-field evolution fit.
        A2 (float): Normalization for the timescale parameter of the second power-law component for the
            magnetic-field evolution fit.
        b1 (float): Power-law index for the timescale parameter of the first power-law component for the
            magnetic-field evolution fit.
        b2 (float): Power-law index for the timescale parameter of the second power-law component for the
            magnetic-field evolution fit.
        tau_late (float): Timescale in [yr] when transitioning from the simulated curves to the simple late-time
            power-law evolution of the magnetic field strength.
        a_late (float): Power-law index for the late-time evolution of the magnetic field strength.

    Returns:
        (np.ndarray): Magnetic field evolution in [G] as a function of time t.
    """
    # We consider an evolution determined by three main timescales tau1, tau2 and tau_late.
    # At early times the curves are fixed to reproduce the simulated magnetic field evolution from the magneto-thermal
    # code. At late times after a timescale tau_late we assume that the evolution is determined by a simple power-law.

    B = np.zeros(len(t))

    # Define the two timescales as a function of the initial B field.
    tau1 = A1 * B_initial**b1
    tau2 = A2 * B_initial**b2

    if tau2 < tau_late:
        B = (
            B_initial
            * (1 + t / tau1) ** a1
            * (1 + t / tau2) ** (a2 - a1)
            * (1 + t / tau_late) ** (a_late - a2)
        )

    elif (tau1 < tau_late) & (tau_late < tau2):
        B = (
            B_initial
            * (1 + t / tau1) ** a1
            * (1 + t / tau_late) ** (a_late - a1)
        )
    elif tau_late < tau1:
        B = B_initial * (1 + t / tau_late) ** a_late

    # If the magnetic field becomes lower than an asymptotic value derived from the old millisecond pulsar population,
    # fix the magnetic field to that constant asymptotic value unless the initial field is already lower than this
    # asymptotic value.
    if B_initial < B_asymptotic:
        B = np.ones(len(t)) * B_initial
    elif B_initial > B_asymptotic:
        B[B < B_asymptotic] = B_asymptotic

    return B


@jit(
    [
        float64(
            float64,
            float64,
            float64,
            float64,
            float64,
            float64,
            float64,
            float64,
            float64,
            float64,
            float64,
        )
    ],
    nopython=True,
)
def magnetic_field_evolution_fit(
    B_initial: float,
    t: float,
    B_asymptotic: float,
    a1: float,
    a2: float,
    A1: float,
    A2: float,
    b1: float,
    b2: float,
    tau_late: float,
    a_late: float,
) -> float:
    """
    An analytical fit for the magnetic field evolution curves from the magneto-thermal evolution simulations.
    This method is used when solving the differential equations if one wants to save only the final state in the
    magneto_rotational_evolution method.
    The fit parameters for each magneto-thermal model specified in the config_simulator.py file were adjusted by hand
    (see the notebook tutorials/analysis_notebooks/magnetic_field_evolution_fit.ipynb for more details).

    Args:
        B_initial (float): Initial magnetic field strength in [G].
        t (float): Time in [s].
        B_asymptotic (float): Asymptotic magnetic field strength at late times in [G].
        a1 (float): Power-law index for the first power-law component for the magnetic-field evolution fit.
        a2 (float): Power-law index for the second power-law component for the magnetic-field evolution fit.
        A1 (float): Normalization for the timescale parameter of the first power-law component for the
            magnetic-field evolution fit.
        A2 (float): Normalization for the timescale parameter of the second power-law component for the
            magnetic-field evolution fit.
        b1 (float): Power-law index for the timescale parameter of the first power-law component for the
            magnetic-field evolution fit.
        b2 (float): Power-law index for the timescale parameter of the second power-law component for the
            magnetic-field evolution fit.
        tau_late (float): Timescale in [yr] when transitioning from the simulated curves to the simple late-time
            power-law evolution of the magnetic field strength.
        a_late (float): Power-law index for the late-time evolution of the magnetic field strength.

    Returns:
        (float): Magnetic field value in [G] at time t.
    """
    # We consider an evolution determined by three main timescales tau1, tau2 and tau_late.
    # At early times the curves are fixed to reproduce the simulated magnetic field evolution from the magneto-thermal
    # code. At late times after a timescale tau_late we assume that the evolution is determined by a simple power-law.

    B: float = B_initial

    # Define the two timescales as a function of the initial B field.
    tau1 = A1 * B_initial**b1
    tau2 = A2 * B_initial**b2

    if tau2 < tau_late:
        B = (
            B_initial
            * (1 + t / tau1) ** a1
            * (1 + t / tau2) ** (a2 - a1)
            * (1 + t / tau_late) ** (a_late - a2)
        )

    elif (tau1 < tau_late) & (tau_late < tau2):
        B = (
            B_initial
            * (1 + t / tau1) ** a1
            * (1 + t / tau_late) ** (a_late - a1)
        )
    elif tau_late < tau1:
        B = B_initial * (1 + t / tau_late) ** a_late

    # If the magnetic field becomes lower than an asymptotic value derived from the old millisecond pulsar population,
    # fix the magnetic field to that constant asymptotic value unless the initial field is already lower than this
    # asymptotic value.
    if B_initial < B_asymptotic:
        B = B_initial
    elif B < B_asymptotic:
        B = B_asymptotic

    return B


@jit(
    [
        float64[:](
            float64,
            float64[:],
            float64,
            float64,
            float64,
            float64,
            float64,
            float64,
            float64,
            float64,
            float64,
            float64,
            float64,
            float64,
        )
    ],
    nopython=True,
)
def combined_derivatives(
    t: float,
    y: np.ndarray,
    B_initial: float,
    B_asymptotic: float,
    a1: float,
    a2: float,
    A1: float,
    A2: float,
    b1: float,
    b2: float,
    tau_late: float,
    a_late: float,
    NS_mass: float,
    NS_radius: float,
) -> np.ndarray:
    """
    Combining the two derivative functions for the misalignment angle
    and the spin period (combined into a single two-component vector y) into a single
    function to allow combined integration.

    Args:
        t (float): Unused time variable, required for the integration below.
        y (np.ndarray): Two magneto-rotational parameters, i.e., chi in [rad]
            and P in [s] for a single pulsar at a given time.
        B_initial (float): Initial magnetic field magnitude for one pulsar, measured in [G].
        B_asymptotic (float): Asymptotic magnetic field strength at late times in [G].
        a1 (float): Power-law index for the first power-law component for the magnetic-field evolution fit.
        a2 (float): Power-law index for the second power-law component for the magnetic-field evolution fit.
        A1 (float): Normalization for the timescale parameter of the first power-law component for the
            magnetic-field evolution fit.
        A2 (float): Normalization for the timescale parameter of the second power-law component for the
            magnetic-field evolution fit.
        b1 (float): Power-law index for the timescale parameter of the first power-law component for the
            magnetic-field evolution fit.
        b2 (float): Power-law index for the timescale parameter of the second power-law component for the
            magnetic-field evolution fit.
        tau_late (float): Timescale in [yr] when transitioning from the simulated curves to the simple late-time
            power-law evolution of the magnetic field strength.
        a_late (float): Power-law index for the late-time evolution of the magnetic field strength.
        NS_mass (float): Neutron star mass, measured in [g].
        NS_radius (float): Neutron star radius, measured in [cm].

    Returns:
        (np.ndarray): Derivative of the two magneto-rotational parameters for one pulsar,
        quantities are referred to in respective changes per [yr].
    """

    # Unpacking the two components of the vector y.
    chi, P = y

    # Specifying the two derivatives.
    dy = np.zeros(len(y), dtype=np.float64)

    B = magnetic_field_evolution_fit(
        B_initial, t, B_asymptotic, a1, a2, A1, A2, b1, b2, tau_late, a_late
    )

    dy[0] = madv.misalignment_angle_derivative(B, chi, P, NS_mass, NS_radius)
    dy[1] = pdv.period_derivative(B, chi, P, NS_mass, NS_radius)

    return dy


def magneto_rotational_evolution(
    B_initial: np.ndarray,
    chi_initial: np.ndarray,
    P_initial: np.ndarray,
    t_age: np.ndarray,
    a_late: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Evolving the neutron stars' magnetic fields, misalignment angles and periods
    according to their respective ages forward in time to obtain their current
    magnetic field strengths, misalignment angles and periods. Note that right
    now the times at which these three parameters are evaluated (apart from the
    current time) do not agree for pulsars.

    Args:
        B_initial (np.ndarray): Pulsars' initial magnetic field magnitudes, measured in [G].
        chi_initial (np.ndarray): Pulsars' initial misalignment angles, measured in [rad].
        P_initial (np.ndarray): Pulsars' initial rotation periods, measured in [s].
        t_age (np.ndarray): Array of neutron star ages in [yr].
        a_late (float): Power-law index for the late-time evolution of the magnetic field strength.

    Returns:
        (Tuple[np.ndarray, np.ndarray, np.ndarray, dict]): Tuple consisting of three arrays defining the neutron stars'
            final magnetic field strengths in [G], misalignment angles in [rad] and rotation periods in [s] and a
            dictionary containing the time evolution of these quantities for each neutron star (if the option to save
            the time evolution is enabled).
    """

    # Save the number of simulated neutron stars, which is flexible depending on whether
    # they are simulated all at once or one by one.
    n = len(t_age)

    # Initialization of a dictionary that will contain the evolution in time of B, chi and P.
    evolution_dictionary = {}

    # Initialization of the array for the three parameters.
    B_final = np.zeros(n)
    chi_final = np.zeros(n)
    P_final = np.zeros(n)

    # Initial conditions for the two parameters.
    y_initial = np.column_stack((chi_initial, P_initial))

    # Draw a random asymptotic value of the magnetic field at late times from a log-normal distribution.
    # This asymptotic value is based on the distribution of inferred magnetic fields for the old population
    # of millisecond pulsars.
    B_asymptotic = 10 ** np.random.normal(
        cfg["B_millisec_mean"], cfg["B_millisec_sigma"], n
    )

    for i in range(n):

        # Generating a time grid at which the solution is evaluated. We start to
        # evolve each star at its birth, corresponding to time 0, and do so for
        # its full age in steps of the time_step specified in the configuration
        # file. To obtain the magnetic field at the current time, we append the
        # current age value.
        time_grid = np.append(
            10
            ** np.arange(0, np.log10(t_age[i]), cfg["magrot_time_step_log10"]),
            t_age[i],
        )

        # To integrate the problem, we use scipy's odeint function.
        # We set tfirst=True to unify the structure of the input ODEs in order to be able
        # to compare different scipy functions to solve the ODEs.
        evol_output = np.array(
            odeint(
                combined_derivatives,
                y0=y_initial[i],
                t=time_grid,
                args=(
                    B_initial[i],
                    B_asymptotic[i],
                    cfg["a1"],
                    cfg["a2"],
                    cfg["A1"],
                    cfg["A2"],
                    cfg["b1"],
                    cfg["b2"],
                    cfg["tau_late"],
                    a_late,
                    cfg["NS_mass"],
                    cfg["NS_radius"],
                ),
                tfirst=True,
            )
        )

        if cfg["save_magrot_evolution"]:

            # Evaluate the magnetic field evolution.
            B_t = magnetic_field_evolution_fit_numpy(
                B_initial[i],
                time_grid,
                B_asymptotic[i],
                cfg["a1"],
                cfg["a2"],
                cfg["A1"],
                cfg["A2"],
                cfg["b1"],
                cfg["b2"],
                cfg["tau_late"],
                a_late,
            )

            # Save the evolution output of the i-th neutron star in a dictionary.
            evolution = {
                i: {
                    "t": time_grid.tolist(),
                    "B(t)": B_t.tolist(),
                    "chi(t)": evol_output[:, 0].tolist(),
                    "P(t)": evol_output[:, 1].tolist(),
                }
            }
            # Update the dictionary containing the evolution information.
            evolution_dictionary = {**evolution_dictionary, **evolution}

        # Save the final values of the magnetic field, inclination angle and spin period.
        B_final[i] = magnetic_field_evolution_fit(
            B_initial[i],
            t_age[i],
            B_asymptotic[i],
            cfg["a1"],
            cfg["a2"],
            cfg["A1"],
            cfg["A2"],
            cfg["b1"],
            cfg["b2"],
            cfg["tau_late"],
            a_late,
        )
        chi_final[i] = evol_output[-1, 0]
        P_final[i] = evol_output[-1, 1]

    return B_final, chi_final, P_final, evolution_dictionary


def initialize_population_magrot(age: np.ndarray) -> dict:
    """
    Initialize the magneto-rotational properties of a neutron star population.

    Args:
        age (np.ndarray): An array of ages in [yr] fot the neutron stars that has to be initialized for the
            magneto-rotational evolution.

    Returns:
        (dict): A dictionary containing the initialized magneto-rotational properties of the neutron star population
            in the survey sky coverage.
    """

    # Initialize neutron star population properties.
    pop_initial = ipop.InitialNeutronStarPopulation(NS_number=len(age))

    # Computing the initial field strengths, misalignment angles, and periods.
    B_initial = pop_initial.magnetic_field()
    chi_initial = pop_initial.misalignment_angle()
    P_initial = pop_initial.period()

    dictionary_initial_pop_magrot = {
        "age": age,
        "B_initial": B_initial,
        "chi_initial": chi_initial,
        "P_initial": P_initial,
    }

    return dictionary_initial_pop_magrot


def evolve_population_magrot(
    dict_pop_initial_magrot: dict,
    output_path: pathlib.Path,
) -> dict:
    """
    Evolve the magneto-rotational properties of a neutron star population over time based on initial conditions.

    Args:
        dict_pop_initial_magrot (dict): Dictionary containing initial magneto-rotational properties of the population.
        output_path (pathlib.Path): The path where the evolution data will be saved if enabled in the configuration.

    Returns:
        (dict): A dictionary containing the properties of the evolved neutron star population.
    """
    age = dict_pop_initial_magrot["age"]
    B_initial = dict_pop_initial_magrot["B_initial"]
    chi_initial = dict_pop_initial_magrot["chi_initial"]
    P_initial = dict_pop_initial_magrot["P_initial"]

    a_late = cfg["a_late"]

    # Determine the evolved magnetic field, misalignment angle and rotation period.
    (
        B_final,
        chi_final,
        P_final,
        magrot_evol_dict,
    ) = magneto_rotational_evolution(
        B_initial,
        chi_initial,
        P_initial,
        age,
        a_late,
    )

    if cfg["save_magrot_evolution"]:
        # Save dictionary containing evolution information to output path in a .json file.
        magrot_evolution_dump_path = pathlib.Path().joinpath(
            output_path, "magrot_evolution.json"
        )

        with open(magrot_evolution_dump_path, "wb") as f:
            f.write(
                orjson.dumps(
                    dict(magrot_evol_dict),
                    option=orjson.OPT_SERIALIZE_NUMPY
                    | orjson.OPT_NON_STR_KEYS
                    | orjson.OPT_SORT_KEYS,
                )
            )

    # Determining the final period derivative.
    P_dot_final = pdv.period_derivative_numpy(
        B_final,
        chi_final,
        P_final,
        cfg["NS_mass"],
        cfg["NS_radius"],
    )

    dictionary_final_pop_magrot = {
        "B_initial": B_initial,
        "B": B_final,
        "chi": chi_final,
        "P": P_final,
        "P_dot": P_dot_final,
    }

    return dictionary_final_pop_magrot
