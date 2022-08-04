"""
Dynamical evolution of the neutron stars in the galactic potential in Julia.

We solve the system of dynamical differential equations in cylindrical coordinates,
using a galactocentric reference frame. Here we are using the Julia OrdinaryDiffEq
(https://diffeq.sciml.ai/stable/tutorials/ode_example/) package which uses the method
'LSODA' (https://diffeq.sciml.ai/stable/solvers/ode_solve/#LSODA.jl) (Adams/BDF method
with automatic stiffness detection and switching).
Alternatively, one can use the Tsit5() solver from the OrdinaryDiffEq package.




Authors:

        Borja Miñano (borja.minano@uib.es)
        Celsa Pardo Araujo (pardo @ csic.es)
        Michele Ronchi (ronchi @ ice.csic.es)


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

using OrdinaryDiffEq
using LSODA

include("galactic_model.jl")


function rhs(du, u, p, t)
    """
    System of dynamical equations to solve to determine the orbits of the neutron
    stars in the galactic potential. The differential equation are written in
    cylindrical galactocentric coordinates (r, phi, z).

    Args:
        du (Array): vector where the output is saved (this is called in julia in-place form)
        which contains 6 values of the first order and second order derivatives at each time step.
        u (Array): array of 6 components defining the initial
        conditions in cylindrical coordinates (r0, phi0, z0, v_r0, omega0, v_z0)
        with the following units ([kpc], [rad], [kpc], [kpc/yr], [rad/yr], [kpc/yr]).
        p (Array): unused variable of the parameters of the model.
        t (float): unused time variable, required for the integration below.

    Returns:
        None

    """
    r, _, z, du[1], du[2], du[3] = u

    gradient_mw_pot = cylind_coord_gradient_mw_potential(galactic_model, r, z)

    # Second derivatives.
    du[4] = r * du[2] * du[2] - gradient_mw_pot[1]
    du[5] = -2 * du[1] * du[2] / r - gradient_mw_pot[2]
    du[6] = -gradient_mw_pot[3]

end


function odeint_final_state(u0, tr, tol)

    """
    Solve a system of ordinary differential equations with ODEProblem using LSODA.
    The solution is computed in the time range tr and interpolated and saved at the final time point (t_age).


    Args:
        u0 (Array): array of 6 components defining the initial
        conditions in cylindrical coordinates (r0, phi0, z0, v_r0, omega0, v_z0)
        with the following units ([kpc], [rad], [kpc], [kpc/yr], [rad/yr], [kpc/yr]).
        tr (float): time range in which the ODEs are solved.
        tol (float): minimum tolerance to solve the ODEs.

    Returns:
        solution.u (float) : solution of the ODEs evaluated at the final time point.

    """
    prob = ODEProblem(rhs, u0, tr, save_everystep=false)

    solution = solve(prob, lsoda(), save_everystep=false, reltol=tol, abstol=tol)

    # The sol.u is an array storing the solution at the corresponding time point.
    return solution.u
end


function odeint_full_output(t0, u0, tr, tol)

    """
    Solve a system of ordinary differential equations with ODEProblem using LSODA.
    The solution is computed in the time range tr and interpolated and saved at the time points specified in the array
    t0.

    Args:
        t0 (Array): array of times where the solution is saved.
        u0 (Array): array of 6 components defining the initial
        conditions in cylindrical coordinates (r0, phi0, z0, v_r0, omega0, v_z0)
        with the following units ([kpc], [rad], [kpc], [kpc/yr], [rad/yr], [kpc/yr]).
        tr (float): time range in which the ODEs are computed.
        tol (float): minimum tolerance to solve the ODEs.

    Returns:
        solution.u (Array) : solution of the ODEs evaluated at times t0.

    """

    prob = ODEProblem(rhs, u0, tr, save_everystep=false, saveat=t0)

    solution = solve(prob, lsoda(), save_everystep=false, reltol=tol, abstol=tol)

    # The sol.u is an array storing the solution at the corresponding time points.
    return solution.u
end


function save_evolution(evol_output, time_grid)

    """
    Saving the evolved output at times time_grid.

    Args:
        evol_output (Array): array of the dynamical evolution at times time_grid with size
        (time_grid,6).
        time_grid (Array): array of times at which the output is evaluated.

    Returns:
       (Dict) : Dictionary with the output.

    """
    #= The evol_output is a vector of vectors where each individual vector is the set of parameters at the time step t,
     with the reduce and hcat command of Julia we transform into a matrix of (6,time_grid) size to easily access the
     evolution for each of the parameters. =#

    evol_output = reduce(hcat, evol_output)


    # Convert the velocity evolution output from kpc / yrs into units of km / s.
    v_r_evol = evol_output[4, :] * KPC_TO_KM / YR_TO_S
    v_phi_evol = evol_output[1, :] .* evol_output[5, :] * KPC_TO_KM / YR_TO_S
    v_z_evol = evol_output[6, :] * KPC_TO_KM / YR_TO_S

    # Save the evolution output of each neutron star in a dictionary.
    return Dict(
        "t" => time_grid, 
        "r(t)" => evol_output[1, :], 
        "phi(t)" => evol_output[2, :], 
        "z(t)" => evol_output[3, :],
        "v_r(t)" => v_r_evol,
        "v_phi(t)" => v_phi_evol,
        "v_z(t)" => v_z_evol,
        )

end 


function solver_calls()

    """
    Performing the dynamical evolution of the neutron star population for a given
    galactic potential, starting from a set of initial conditions.

    Returns:
        (Array, dict): Tuple consisting of a two-dimensional array of shape (NS_number, 6)
        defining the neutron stars' final positions r [kpc], phi [rad], z [kpc] and velocities
        in [kpc/yr] in cylindrical coordinates and a dictionary containing the time evolution of
        these quantities for each neutron star (if the option to save the time evolution is enabled).
    """

    r_final = Float64[]
    phi_final = Float64[]
    z_final = Float64[]
    v_r_final = Float64[]
    v_phi_final = Float64[]
    v_z_final = Float64[]

    evolution_dictionary = Dict()

    for i in 1:n

        # Each star's position and velocity is evolved for a time equal to its age.
        timerange = (0.0, t_age[i])

        # The variables t_age, time_step, initial_condition and tolerance are defined via Main in the python script that
        # uses the solver.
        if save_dyn_evolution
            time_grid = append!(collect(0.0:time_step:t_age[i]), t_age[i])
            evol_output = odeint_full_output(time_grid, initial_cond[i, :], timerange, tolerance)

            # Save the evolution output of the i-th neutron star in a dictionary.
            evolution_dictionary[string(i-1)] = save_evolution(evol_output, time_grid)
        else
            evol_output = odeint_final_state(initial_cond[i, :], timerange, tolerance)
        end

        # Save the final position and velocity of the i-th neutron star.
        # Note: We save directly the v_phi velocity component and not the angular velocity omega.
        last_item = evol_output[end]
        append!(r_final, last_item[1])
        append!(phi_final, last_item[2])
        append!(z_final, last_item[3])
        append!(v_r_final, last_item[4])
        omega_final = last_item[5]
        append!(v_phi_final, omega_final * r_final[i])
        append!(v_z_final, last_item[6])

    end

    return r_final, phi_final, z_final, v_r_final, v_phi_final, v_z_final, evolution_dictionary

end
