using OrdinaryDiffEq
using LSODA

include("galactic_model.jl")


function rhs(du,u,p,t)

    r, _, z, du[1], du[2], du[3] = u

    gradient_mw_pot = cylind_coord_gradient_mw_potential(galactic_model, r, z)

    # Second derivatives.
    du[4] = r * du[2] * du[2] - gradient_mw_pot[1]
    du[5] = -2 * du[1] * du[2] / r - gradient_mw_pot[2]
    du[6] = -gradient_mw_pot[3]

end


function odeint(u0, tr, tol)

    prob = ODEProblem(rhs,u0,tr,save_everystep=false)

    solution = solve(prob,lsoda(), save_everystep=false,reltol=tol,abstol=tol)
    return solution.u
end


function odeint(t0, u0, tr, tol)

    prob = ODEProblem(rhs,u0,tr,save_everystep=false, saveat=t0)

    solution = solve(prob,lsoda(), save_everystep=false,reltol=tol,abstol=tol)
    return solution.u
end



function save_evolution(evol_output, time_grid)

    evol_output = reduce(hcat,evol_output)

    v_r_evol = evol_output[4, :] * KPC_TO_KM / YR_TO_S
    v_phi_evol = evol_output[1, :] .* evol_output[5, :] * KPC_TO_KM / YR_TO_S
    v_z_evol = evol_output[6, :] * KPC_TO_KM / YR_TO_S

    # Save the evolution output of the i-th neutron star in a dictionary.
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


    r_final = Float64[]
    phi_final = Float64[]
    z_final = Float64[]
    v_r_final = Float64[]
    v_phi_final = Float64[]
    v_z_final = Float64[]

    evolution_dictionary = Dict()

    for i in 1:n

        timerange = (0.0, t_age[i])
        
        if save_dyn_evolution
            time_grid = append!(collect(0.0:time_step:t_age[i]), t_age[i])
            evol_output = odeint(time_grid, initial_cond[i, :], timerange, tolerance)
        else
            evol_output = odeint(initial_cond[i, :], timerange, tolerance)
        end
        
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