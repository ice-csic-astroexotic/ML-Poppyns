"""
Solver for the combined evolution of the pulsar period, misalignment angle and magnetic field
relying on fits to magneto-thermal simulations in Julia.

Authors:

        Michele Ronchi (ronchi@ice.csic.es)

MIT License

Copyright (c) MAGNESIA (ICE-CSIC) 2020

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

using Distributions
using LSODA
using OrdinaryDiffEq

include("period_derivative.jl")
include("misalignment_angle_derivative.jl")

function magnetic_field_evolution_fit_array(B_initial, t, B_asymptotic)
    """
    An analytical function for the magnetic field evolution curves from the magneto-thermal evolution simulations.
    This method is compatible with arrays and is used if one wants to save the entire evolution output in the
    magneto_rotational_evolution method.

    Args:
        B_initial(float): initial magnetic field strength in [G].
        t(np.ndarray): time in [s].
        B_asymptotic(float): asymptotic magnetic field strength at late times in [G].

    Returns:
        (np.ndarray): magnetic field evolution in [G] as a function of time t.
    """
    # Define the two timescales as a function of the initial B field.
    tau1 = A1_cfg * B_initial ^ (-b1_cfg)
    tau2 = A2_cfg * B_initial ^ (-b2_cfg)

    # We split our time-domain into two regions using different prescriptions for early and late times.
    B = zeros(length(t))
    early_times = t .< t_trans_cfg
    late_times = t .>= t_trans_cfg

    # At early times the curves are fixed to reproduce the simulated magnetic field evolution from the magneto-thermal
    # code.
    B[early_times] = (
        B_initial
        .* (1 .+ t[early_times] ./ tau1) .^ (-a1_cfg)
        .* (1 .+ t[early_times] ./ tau2) .^ (-a2_cfg)
    )
    # At late times we assume a simple power-law evolution.
    B[late_times] = (
        B_initial
        .* (1 .+ t_trans_cfg ./ tau1) .^ (-a1_cfg)
        .* (1 .+ t_trans_cfg ./ tau2) .^ (-a2_cfg)
    ) .* (t[late_times] ./ t_trans_cfg) .^ (-a_late_t_cfg)

    # If the magnetic field becomes lower than an asymptotic value derived from the old millisecond pulsar population,
    # fix the magnetic field to that constant asymptotic value.
    B[B .< B_asymptotic] = B_asymptotic

    return B
end


function magnetic_field_evolution_fit(B_initial, t, B_asymptotic)
    """
    An analytical fit for the magnetic field evolution curves from the magneto-thermal
    evolution simulations.

    Args:
        B_initial(float): initial magnetic field strength in [G].
        t(float): time in [s].
        B_asymptotic(float): asymptotic magnetic field strength at late times in [G].

    Returns:
        (float): magnetic field value in [G] at time t.
    """
    # Define the two timescales as a function of the initial B field.
    tau1 = A1_cfg * B_initial ^ (-b1_cfg)
    tau2 = A2_cfg * B_initial ^ (-b2_cfg)

    # We split our time-domain into two regions using different prescriptions for early and late times.
    # At early times the curves are fixed to reproduce the simulated magnetic field evolution from the magneto-thermal
    # code. At late times we assume a simple power-law evolution.
    if t < t_trans_cfg
        B = (
            B_initial
            * (1 + t / tau1) ^ (-a1_cfg)
            * (1 + t / tau2) ^ (-a2_cfg)
        )
    else
        B = (
            B_initial
            * (1 + t_trans_cfg / tau1) ^ (-a1_cfg)
            * (1 + t_trans_cfg / tau2) ^ (-a2_cfg)
            * (t / t_trans_cfg) ^ (-a_late_t_cfg)
        )
    end

    # If the magnetic field becomes lower than an asymptotic value derived from the old millisecond pulsar population,
    # fix the magnetic field to that constant asymptotic value.
    if B < B_asymptotic
        B = B_asymptotic
    end

    return B
end


function rhs(du, u, p, t)
    """
    Combining the two derivative functions for the misalignment angle
    and the spin period (combined into a single two-component vector y) into a single
    function to allow combined integration.

    Args:
        du (Array): vector where the output is saved (this is called in julia in-place form).
        u (Array): array of 2 components defining the initial
        conditions, i.e., chi in [rad] and P in [s] for a single pulsar at a given time.
        p (Array): parameters of the model, i.e., the initial magnetic field magnitude for
        one pulsar, measured in [G] and the asymptotic magnetic field strength at late times in [G].
        t (float): unused time variable, required for the integration below.

    Returns:
        (np.ndarray): derivative of the two magneto-rotational parameters for one pulsar,
        quantities are referred to in respective changes per [yr].
    """

    # Unpacking the two components of the vector y.
    chi, P = u

    B = magnetic_field_evolution_fit(p[1], t, p[2])

    du[1] = misalignment_angle_derivative(B, chi, P)
    du[2] = period_derivative(B, chi, P)

end


function odeint(u0, p, tr, tol)

    """
    Solve a system of ordinary differential equations with ODEProblem using LSODA.
    The solution is computed in the time range tr and interpolated and saved at the
    final time point (t_age).

    Args:
        u0 (Array): array of 2 components defining the initial
        conditions, i.e., chi in [rad] and P in [s] for a single pulsar at a given time.
        p (Array): parameters of the model, i.e., the initial magnetic field magnitude for
        one pulsar, measured in [G] and the asymptotic magnetic field strength at late times in [G].
        tr (float): time range in which the ODEs are solved.
        tol (float): minimum tolerance to solve the ODEs.

    Returns:
        solution.u (float) : solution of the ODEs evaluated at the final time point.

    """
    prob = ODEProblem(rhs, u0, tr, p, save_everystep=false)

    solution = solve(prob, lsoda(), save_everystep=false, reltol=tol, abstol=tol)

    return solution.u
end


function odeint(t0, u0, p, tr, tol)

    """
    Solve a system of ordinary differential equations with ODEProblem using LSODA.
    The solution is computed in the time range tr and interpolated and saved at the time points specified in the array
    t0.

    Args:
        t0 (Array): array of times where the solution is saved.
        u0 (Array): array of 2 components defining the initial
        conditions, i.e., chi in [rad] and P in [s] for a single pulsar at a given time.
        p (Array): parameters of the model, i.e., the initial magnetic field magnitude for
        one pulsar, measured in [G] and the asymptotic magnetic field strength at late times in [G].
        tr (float): time range in which the ODEs are computed.
        tol (float): minimum tolerance to solve the ODEs.

    Returns:
        solution.u (Array) : solution of the ODEs evaluated at times t0.

    """

    prob = ODEProblem(rhs, u0, p, tr, save_everystep=false, saveat=t0)

    solution = solve(prob, lsoda(), save_everystep=false, reltol=tol, abstol=tol)

    return solution.u
end


function save_evolution(evol_output, B_evol, time_grid)

    """
    Saving the evolved output at times time_grid.

    Args:
        evol_output (Array): array of the magneto-rotational evolution at times time_grid
        with size (time_grid,2).
        B_evol (Array): array of the magnetic field evolution at times time_grid.
        time_grid (Array): array of times at which the output is evaluated.

    Returns:
       (Dict) : Dictionary with the output.

    """
    #= The evol_output is a vector of vectors where each individual vector is the set of parameters at the time step t,
     with the reduce and hcat command of Julia we transform into a matrix of (6,time_grid) size to easily access the
     evolution for each of the parameters. =#

    evol_output = reduce(hcat, evol_output)

    # Save the evolution output of each neutron star in a dictionary.
    return Dict(
        "t" => time_grid,
        "B(t)" => B_evol,
        "chi(t)" => evol_output[1, :],
        "P(t)" => evol_output[2, :],
        )
end


function solver_calls()
    """
    Evolving the neutron stars' magnetic fields, misalignment angles and periods
    according to their respective ages forward in time to obtain their current
    magnetic field strengths, misalignment angles and periods. Note that right
    now the times at which these three parameters are evaluated (apart from the
    current time) do not agree for pulsars.

    Returns:
        (Array, Array, Array, Dict): Tuple consisting of three arrays
        defining the neutron stars' final magnetic field strengths in [G],
        misalignment angles in [rad] and rotation periods in [s] and a dictionary containing
        the time evolution of these quantities for each neutron star (if the option to save
        the time evolution is enabled).
    """

    # Initialization of the three parameters.
    B_final = Float64[]
    chi_final = Float64[]
    P_final = Float64[]

    # Initialization of a dictionary that will contain the evolution in time of B, chi and P.
    evolution_dictionary = Dict()

    for i in 1:n

        # Each star's is evolved for a time equal to its age.
        timerange = (0.0, t_age[i])

        p = (B_initial[i], B_asymptotic[i])

        if save_magrot_evolution
            time_grid = append!(10 .^ collect(0.0:magrot_time_step_log:log10(t_age[i])), t_age[i])
            evol_output = odeint(time_grid, initial_cond_magrot[i, :], p, timerange, tolerance)

            # Evaluate the magnetic field evolution.
            B_t = magnetic_field_evolution_fit_array(
                B_initial[i], time_grid, B_asymptotic[i]
            )

            # Save the evolution output of the i-th neutron star in a dictionary.
            evolution_dictionary[string(i-1)] = save_evolution(evol_output, B_t, time_grid)
        else
            evol_output = odeint(initial_cond_magrot[i, :], p, timerange, tolerance)
        end

        # Save the final position and velocity of the i-th neutron star.
        # Note: We save directly the v_phi velocity component and not the angular velocity omega.
        last_item = evol_output[end]
        append!(B_final, magnetic_field_evolution_fit(
            B_initial[i], t_age[i], B_asymptotic[i]
        ))
        append!(chi_final, last_item[1])
        append!(P_final, last_item[2])

    end

    return B_final, chi_final, P_final, evolution_dictionary
end
