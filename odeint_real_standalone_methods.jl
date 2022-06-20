using DifferentialEquations, BenchmarkTools
using LSODA
using Profile
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
const YR_TO_S = 3600.0 * 24.0 * 365.0
const G_KPC_YR = (G / (KPC_TO_CM ^ 3) * YR_TO_S ^ 2)

function shape_parameter(z)

    _sqrt = sqrt(z * z + b_d * b_d)

    K = a_d + _sqrt

    dK_dz = z / _sqrt

    return K, dK_dz
end

r_derivative_b_potential(r) = G_KPC_YR * M_b * (r + r_b) ^ (-2)


r_derivative_n_potential(r) = G_KPC_YR * M_n * (r + r_n) ^ (-2)


r_derivative_h_potential(r) = (
        G_KPC_YR
        * M_h
        / r
        * (1.0 / r * log1p(r / r_h) - 1.0 / (r_h + r))
    )

function r_z_derivatives_d_potential(r, z)


    K, dK_dz = shape_parameter(z)

    sqrt =  (r * r + K * K) ^ (-1.5)

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

function rober(du, u,p,t)

    r, _, z, du[1], du[2], du[3] = u

    gradient_mw_pot = cylind_coord_gradient_mw_potential(r, z)

    du[4] = r * du[2] * du[2] - gradient_mw_pot[1]
    du[5] = -2 * du[1] * du[2] / r - gradient_mw_pot[2]
    du[6] = -gradient_mw_pot[3]
end


function odeint(t0, u0, tr, dt)

    prob = ODEProblem(rober,u0,tr, saveat=t0,save_everystep=false)
    solution = solve(prob,lsoda(),save_everystep=false,reltol=1e-8,abstol=1e-8) # 9.300 μs (26 allocations: 3.03 KiB)

    return solution.u
end



#Stiff
tend = 1.4708565048135614e7
u0 = [0.1021811020701645, 3.988563758793204, -0.007834562002312783, -1.4791582530956602e-7, 3.8563040038054056e-9, 1.1408224933522102e-7]

tinit = 0.0
dt = 10000.0
t0 = append!(collect(tinit:dt:tend), tend)
tr = (tinit, tend)

#t0 = SVector{size(t0,1)}(t0)

#pruebas = [Rosenbrock23, AutoTsit5(Rosenbrock23()), lsoda, Tsit5, BS3, OwrenZen3, AutoVern7, AutoVern7(Rodas4), AutoVern7(KenCarp4), AutoVern7(Rodas5), Rodas4, radau]

#Funcionan: TRBDF2, Rodas4, KenCarp4, KenCarp3, Kvaerno5, RadauIIA3

    @btime solution = odeint(t0, u0, tr, dt)
    println("Stiff")
    solution = odeint(t0, u0, tr, dt)
    println(typeof(solution))

    evol_output = reduce(hcat,solution)
    evolution_dictionary = Dict()
    evolution_dictionary["0"] = Dict(
                "t" => t0[1:6], 
                "r(t)" => evol_output[1, 1:6], 
                )

    println(evolution_dictionary)

    println(JSON.json(evolution_dictionary))

#No Stiff
tend = 2.163061968120212e7
u0 = [15.511847064292592, 0.7326998994038764, 0.1215349250221546, -1.268459376157434e-7, 4.823630005080897e-9, 1.7382186419146915e-8]


tinit = 0.0
dt = 10000.0
t0 = append!(collect(tinit:dt:tend), tend)
tr = (tinit, tend)

    println("No stiff")
    @btime solution = odeint(t0, u0, tr, dt)
    #println(solution[size(solution, 1)])



#Super Stiff

tend = 9.165803867040945e6
u0 = [5.848829707603326e-5, 0.41856359395982956, -0.16966742967708984, 1.1826612509365937e-7, 0.00011148690823809339, 8.199437113692513e-8]


tinit = 0.0
dt = 10000.0
t0 = append!(collect(tinit:dt:tend), tend)
tr = (tinit, tend)

    println("Super stiff")
    @profile  solution = odeint(t0, u0, tr, dt)
    #println(solution[size(solution, 1)])




#@time odeint(t0, u0, tr, false)
#@btime odeint(t0, u0, tr, false)
#solution = odeint(t0, u0, tr, false)



#methods = [OrdinaryDiffEq.Rosenbrock23(), OrdinaryDiffEq.Tsit5(), BS3()]

#println(methods[1])

#for m in methods
    #println(m)
#    try
##        odeint(t0, u0, tr, m)
#    catch e
#        println("Peta metodo " + m)
#    end
#end
