using OrdinaryDiffEq
using LSODA
using JSON

const M_SUN = 2.0e33
const a_d = 3.0
const b_d = 0.28
const M_d = 6.8e10 * M_SUN  # Disk+halo mass in [g].
const M_b = 5.0e9 * M_SUN  # Bulge mass in [g].
const r_b = 1.0
const M_n = 1.71e9 * M_SUN  # Nucleus mass in [g].
const r_n = 0.07
const M_h = 5.4e11 * M_SUN  # Halo mass in [g].
const r_h = 15.62

const G = 6.67e-8
const KPC_TO_CM = 3.08567758e21
const YR_TO_S = 3600.0 * 24 * 365
const G_KPC_YR = (G / (KPC_TO_CM ^ 3) * YR_TO_S ^ 2)
const KPC_TO_KM = 3.08567758e16

function shape_parameter(z)

    _sqrt = sqrt(z * z + b_d * b_d)

    K = a_d + _sqrt

    dK_dz = z / _sqrt

    return K, dK_dz
end

function r_derivative_b_potential(r)
    
    dpot_b_dr = G_KPC_YR * M_b * (r + r_b) ^ (-2.0)

    return dpot_b_dr
end

function r_derivative_n_potential(r)
    

    dpot_n_dr = G_KPC_YR * M_n * (r + r_n) ^ (-2.0)

    return dpot_n_dr
end

function r_derivative_h_potential(r)

    dpot_h_dr = (
        G_KPC_YR
        * M_h
        / r
        * (1.0 / r * log1p(r / r_h) - 1.0 / (r_h + r))
    )

    return dpot_h_dr

end

function r_z_derivatives_d_potential(r, z)


    K, dK_dz = shape_parameter(z)

    sqrt =  (r * r + K * K) ^ (-3.0 / 2.0)

    dpot_d_dr = (
        G_KPC_YR * M_d * r * sqrt
    )
    dpot_d_dz = (
        G_KPC_YR
        * M_d
        * sqrt
        * K
        * dK_dz
    )

    return dpot_d_dr, dpot_d_dz
end 

function cylind_coord_gradient_mw_potential(r, z)

    dpot_d_dr, dpot_d_dz = r_z_derivatives_d_potential(r, z)
    dpot_b_dr = r_derivative_b_potential(r)
    dpot_n_dr = r_derivative_n_potential(r)
    dpot_h_dr = r_derivative_h_potential(r)

    dpot_mw_dr = dpot_d_dr + dpot_b_dr + dpot_n_dr + dpot_h_dr
    dpot_mw_dphi = 0.0
    dpot_mw_dz = dpot_d_dz

    return dpot_mw_dr, dpot_mw_dphi, dpot_mw_dz

end

function rober(du,u,p,t)


    r, _, z, du[1], du[2], du[3] = u


    gradient_mw_pot = cylind_coord_gradient_mw_potential(r, z)

    # Second derivatives.
    du[4] = r * du[2] * du[2] - gradient_mw_pot[1]
    du[5] = -2 * du[1] * du[2] / r - gradient_mw_pot[2]
    du[6] = -gradient_mw_pot[3]

end


function odeint(t0, u0, tr)

    prob = ODEProblem(rober,u0,tr,save_everystep=false, saveat=t0) #saveat=t0

    solution = solve(prob,lsoda(), save_everystep=false,reltol=1e-9,abstol=1e-9)
    return solution.u
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

        time_grid = append!(collect(0.0:time_step:t_age[i]), t_age[i])

        timerange = (0.0, t_age[i])
        

        evol_output = odeint(time_grid, initial_cond[i, :], timerange)
        
        last_item = evol_output[end]
        append!(r_final, last_item[1])
        append!(phi_final, last_item[2])
        append!(z_final, last_item[3])
        append!(v_r_final, last_item[4])
        omega_final = last_item[5]
        append!(v_phi_final, omega_final * r_final[i])
        append!(v_z_final, last_item[6])

        if save_dyn_evolution
            evol_output = reduce(hcat,evol_output)

            v_r_evol = evol_output[4, :] * KPC_TO_KM / YR_TO_S
            v_phi_evol = (evol_output[1, :] .* evol_output[5, :] * KPC_TO_KM / YR_TO_S)
            v_z_evol = evol_output[6, :] * KPC_TO_KM / YR_TO_S

            # Save the evolution output of the i-th neutron star in a dictionary.
            evolution_dictionary[i-1] = Dict(
                "t" => time_grid, 
                "r(t)" => evol_output[1, :], 
                "phi(t)" => evol_output[2, :], 
                "z(t)" => evol_output[3, :],
                "v_r(t)" => v_r_evol,
                "v_phi(t)" => v_phi_evol,
                "v_z(t)" => v_z_evol,
                )
        end


    end


    return r_final, phi_final, z_final, v_r_final, v_phi_final, v_z_final, evolution_dictionary

end