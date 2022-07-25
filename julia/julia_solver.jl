using OrdinaryDiffEq
using LSODA

include("galactic_model.jl")


function rhs(du,u,p,t)

    """
    System of dynamical equations to solve to determine the orbits of the neutron
    stars in the galactic potential. The differential equation are written in
    cylindrical galactocentric coordinates (r, phi, z).

    Args:
        du (Array): vector where the output is saved (this is called in julia in-place form).
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


function odeint(u0, tr, tol)

    """
    Solve a system of ordinary differential equations with ODEProblem using lsoda.
    The solution is given at time tr.

    Args:
        u0 (Array): array of 6 components defining the initial
        conditions in cylindrical coordinates (r0, phi0, z0, v_r0, omega0, v_z0)
        with the following units ([kpc], [rad], [kpc], [kpc/yr], [rad/yr], [kpc/yr]).
        tr (float): time range for which the ODEs are solved.
        tol (float): minimum tolerance to solve the ODEs.

    Returns:
        solution.u (float) : solution of the ODEs evaluated at time tr.

    """
    prob = ODEProblem(rhs,u0,tr,save_everystep=false)

    solution = solve(prob,lsoda(), save_everystep=false,reltol=tol,abstol=tol)
    return solution.u
end


function odeint(t0, u0, tr, tol)

    """
    Solve a system of ordinary differential equations with ODEProblem using lsoda.
    The solution is given at times t0.

    Args:
        t0 (Array): array of times where the solution is evaluated.
        u0 (Array): array of 6 components defining the initial
        conditions in cylindrical coordinates (r0, phi0, z0, v_r0, omega0, v_z0)
        with the following units ([kpc], [rad], [kpc], [kpc/yr], [rad/yr], [kpc/yr]).
        tr (float): time range for which the ODEs are solved.
        tol (float): minimum tolerance to solve the ODEs.

    Returns:
        solution.u (Array) : solution of the ODEs evaluated at times t0.

    """

    prob = ODEProblem(rhs,u0,tr,save_everystep=false, saveat=t0)

    solution = solve(prob,lsoda(), save_everystep=false,reltol=tol,abstol=tol)
    return solution.u
end



function save_evolution(evol_output, time_grid)

    """
    Saving the evolved output at times time_grid.

    Args:
        evol_output (Array): array of the dynamical evolution at times time_grid with size (time_grid,6).
        time_grid (Array): array of times at which the output is evaluated.

    Returns:
       (Dict) : Dictionary with the output.

    """
    #= The evol_output is a matrix of (time_grid,6), with the reduce and hcat command of Julia we transform into a
    matrix of (6,time_grid) size. =#

    evol_output = reduce(hcat,evol_output)


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
        (np.ndarray, dict): Tuple consisting of a two-dimensional array of shape (NS_number, 6)
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
        
        if save_dyn_evolution
            time_grid = append!(collect(0.0:time_step:t_age[i]), t_age[i])
            evol_output = odeint(time_grid, initial_cond[i, :], timerange, tolerance)
        else
            evol_output = odeint(initial_cond[i, :], timerange, tolerance)
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

        if save_dyn_evolution
            # Save the evolution output of the i-th neutron star in a dictionary.
            evolution_dictionary[string(i-1)] = save_evolution(evol_output, time_grid)
        end


    end


    return r_final, phi_final, z_final, v_r_final, v_phi_final, v_z_final, evolution_dictionary

end