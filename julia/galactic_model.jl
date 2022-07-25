"""
Model for the Milky Way gravitational potential in Julia.

We consider two different models:

1) gmFK06: A galactic structure as in Faucher-Giguère & Kaspi (2006). Their
model consists of three components: a disk-halo, a bulge and a nucleus.
The parameters of the model are taken from Table B1 in Kuijken & Gilmore (1989).

2) gmM19: The galaxy model from Marchetti et al. (2019). This is a four-component
galactic potential model consisting of a Hernquist bulge and nucleus (Hernquist 1990),
a Miyamoto-Nagai disk (Miyamoto & Nagai 1975) and a Navarro-Frenk-White halo (Navarro
et al. 1996). The parameters of the model are taken from Table 1 in Marchetti et al.
(2019) and are chosen to fit the enclosed mass profile of the Milky Way (Bovy 2015).

Authors:
        Borja Miñano (borja.minano@uib.es)
        Vanessa Graber (graber@ice.csic.es)
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


include("constants.jl")

#= Defining an abstract type for the galactic model. Note that Julia is not an object-oriented language, i.e.
we define types not classes. =#

abstract type GalacticModel end

"""
Galaxy model from Marchetti et al. (2019). This is a four-component galactic
potential model consisting of a Hernquist bulge and nucleus (Hernquist 1990),
a Miyamoto-Nagai disk (Miyamoto & Nagai 1975) and a Navarro-Frenk-White halo
(Navarro et al. 1996). The parameters of the model are taken from Table 1 in
Marchetti et al. (2019) and are chosen to fit the enclosed mass profile of
the Milky Way (Bovy 2015).
"""

#= Julia uses the struct similarly to objects in object-oriented languages. With struct we can define the field of each
"class", similarly to the __init__ in Python. Here we define all the variables needed for each galactic model. =#

struct GalaxyModelM19 <:GalacticModel
    a_d::Float64
    b_d::Float64
    M_d::Float64
    M_b::Float64
    r_b::Float64
    M_n::Float64
    r_n::Float64
    M_h::Float64
    r_h::Float64
end

"""
Galaxy model from Faucher-Giguère & Kaspi (2006). This model consists of a
disk-halo component, a bulge component, and a nucleus component. The
parameters of the model are taken from Table 1 in Kuijken & Gilmore (1989)
(in Faucher-Giguère & Kaspi (2006) the nucleus and bulge are erroneously
inverted).
"""
struct GalaxyModelFK06 <:GalacticModel
    a_d::Float64
    h::Vector{Float64}
    beta::Vector{Float64}
    M_dh::Float64
    b_dh::Float64
    M_b::Float64
    b_b::Float64
    M_n::Float64
    b_n::Float64
end


function initialize_galactic_model(input)

    """
    Initializing the galactic model employed in the simulation. We have implemented two
    versions, i.e., the galactic structure of Faucher-Giguère & Kaspi (2006) (with
    parameters from Kuijken & Gilmore (1989)) and Galaxy model from Marchetti et al.
    (2019). The galactic_model variable is made available on a global level.

    Args:
        input (str): Type of Galactic model to be used, i.e. gmFK06 or gmM19.

    Returns:
        GalacticModel (abstract type): GalacticModel abstract type for each of the different galactic model
        we have defined.

    """

    # The ´cmp´ command compares two strings, if they are equal returns 0.
	if cmp(input, "gmM19") == 0
        # Parameters of the model, values from Table 1 in Marchetti et al. (2019).
		return GalaxyModelM19(3.0,              # Scale length of the disk in [kpc].
                            0.28,               # Scale height for the disk.
                            6.8e10 * M_SUN,     # Disk+halo mass in [g].
                            5.0e9 * M_SUN,      # Bulge mass in [g].
                            1.0,                # Core radius of the bulge component in [kpc].
                            1.71e9 * M_SUN,     # Nucleus mass in [g].
                            0.07,               # Core radius of the nucleus component in [kpc].
                            5.4e11 * M_SUN,     # Halo mass in [g].
                            15.62)              # Core radius of the halo component in [kpc].
    end
	if cmp(input, "gmFK06") == 0
        # Parameter values from Table B1 in Kuijken & Gilmore (1989).
		return GalaxyModelFK06(2.4,             # Scale length of the disk in [kpc].
                        [0.325, 0.090, 0.125],  # Array of disk components' scale heights in [kpc].
                        [0.4, 0.5, 0.1],        # Array of weights for the disk components.
                        1.45e11 * M_SUN,        # Disk+halo mass in [g].
                        5.5,                    # Core radius of the halo component in [kpc].
                        1.0e10 * M_SUN,         # Bulge mass in [g].
                        1.5,                    # Core radius of the bulge component in [kpc].
                        9.3e9 * M_SUN,          # Nucleus mass in [g].
                        0.25)                   # Core radius of the nucleus component in [kpc].
    end
    throw("The galactic model does not exist. Choose between gmFK06 or gmM19.")
end

#= Galactic model initialization. Constant variables in Julia are global variables which its type can not change.
# The variable galactic_model_input it is saved in the Main of Julia. To do so, we add this line of code
Main.galactic_model_input = cfg["galactic_model"] in the main script.=#


const galactic_model = initialize_galactic_model(galactic_model_input)



#= Galactic model M19 functions=#

function shape_parameter(galactic_model::GalaxyModelM19, z)
    """
    Shape parameter for the disk potential from Marchetti et al. (2019) and
    its derivative with respect to the height z from the galactic disk.
    Second term in the denominator of eq. (8) in Marchetti et al. (2019).

    Args:
        galactic_model (abstract type) : gmM19 model parameters.
        z (float): height from the galactic disk in [kpc].

    Returns:
        (float, float): value of the shape parameter and its derivative with
        respect to z.
    """

    _sqrt = sqrt.(z .* z .+ (galactic_model.b_d * galactic_model.b_d))

    K = galactic_model.a_d .+ _sqrt

    dK_dz = z ./ _sqrt

    return K, dK_dz
end

function r_derivative_b_potential(galactic_model::GalaxyModelM19, r)
    """
    Derivative with respect to r of the bulge component gravitational potential
    defined in eq. (7) in Marchetti et al. (2019).

    Args:
        galactic_model (abstract type) : gmM19 model parameters.
        r (float): distance in the galactic disk from the galactic centre in [kpc].

    Returns:
        float: derivative with respect to r of the bulge potential.
    """

    dpot_b_dr = G_KPC_YR * galactic_model.M_b * (r .+ galactic_model.r_b) .^ (-2.0)

    return dpot_b_dr
end

function r_derivative_n_potential(galactic_model::GalaxyModelM19, r)
    """
    Derivative with respect to r of the nucleus component gravitational potential
    defined in eq. (7) in Marchetti et al. (2019).

    Args:
        galactic_model (abstract type) : gmM19 model parameters.
        r (float): distance in the galactic disk from the galactic centre in [kpc].

    Returns:
        float: derivative with respect to r of the nucleus potential.
    """

    dpot_n_dr = G_KPC_YR * galactic_model.M_n * (r .+ galactic_model.r_n) .^ (-2.0)

    return dpot_n_dr
end

function r_derivative_h_potential(galactic_model::GalaxyModelM19, r)
    """
    Derivative with respect to r of the halo component gravitational potential
    defined in eq. (9) in Marchetti et al. (2019).

    Args:
        galactic_model (abstract type) : gmM19 model parameters.
        r (float): distance in the galactic disk from the galactic centre in [kpc].

    Returns:
        float: derivative with respect to r of the halo potential.
    """

    dpot_h_dr = (
        G_KPC_YR
        * galactic_model.M_h
        ./ r
        .* (1.0 ./ r .* log1p.(r ./ galactic_model.r_h) .- 1.0 ./ (galactic_model.r_h .+ r))
    )

    return dpot_h_dr

end

function r_z_derivatives_d_potential(galactic_model::GalaxyModelM19, r, z)
    """
    Derivative with respect to r and z of the disk component gravitational
    potential defined in eq. (8) in Marchetti et al. (2019).

    Args:
        galactic_model (abstract type) : gmM19 model parameters.
        r (float): distance in the galactic disk from the galactic centre in [kpc].
        z (float): height from the galactic disk in [kpc].

    Returns:
        (float, float): derivative with respect to r and z of the disk potential.
    """
 
    K, dK_dz = shape_parameter(galactic_model, z)
 
    sqrt =  (r .* r .+ K .* K) .^ (-3.0 / 2.0)
 
    dpot_d_dr = (
        G_KPC_YR * galactic_model.M_d * r .* sqrt
    )
    dpot_d_dz = (
        G_KPC_YR
        * galactic_model.M_d
        * sqrt
        .* K
        .* dK_dz
    )

    return dpot_d_dr, dpot_d_dz
end 


function cylind_coord_gradient_mw_potential(galactic_model::GalaxyModelM19, r, z)
    """
    Gradient in cylindrical coordinates of the Milky Way gravitational potential for
    the components defined in eq. (7,8,9) in Marchetti et al. (2019).

    Args:
        galactic_model (abstract type) : gmM19 model parameters.
        r (float): distance in the galactic disk from the galactic centre in [kpc].
        z (float): height from the galactic disk in [kpc].

    Returns:
        (np.ndarray): gradient of the galactic potential in cylindrical
        coordinates.
    """

    dpot_d_dr, dpot_d_dz = r_z_derivatives_d_potential(galactic_model, r, z)
    dpot_b_dr = r_derivative_b_potential(galactic_model, r)
    dpot_n_dr = r_derivative_n_potential(galactic_model, r)    
    dpot_h_dr = r_derivative_h_potential(galactic_model, r)

    dpot_mw_dr = dpot_d_dr .+ dpot_b_dr .+ dpot_n_dr .+ dpot_h_dr
    dpot_mw_dphi = 0.0
    dpot_mw_dz = dpot_d_dz

    return dpot_mw_dr, dpot_mw_dphi, dpot_mw_dz
end

function d_potential(galactic_model::GalaxyModelM19, r, z)
    """
    The Miyamoto-Nagai disk component gravitational potential defined in eq. (8) in
    Marchetti et al. (2019).

    Args:
        galactic_model (abstract type) : gmM19 model parameters.
        r (np.ndarray): distance in the galactic disk from the galactic centre in [kpc].
        z (np.ndarray): height from the galactic disk in [kpc].

    Returns:
        (np.ndarray): value of the disk-halo potential in [erg/g].
    """

    K, _ = shape_parameter(galactic_model, z)

    pot_d = -G * galactic_model.M_d ./ (sqrt.(K .* K .+ r .* r) * KPC_TO_CM)

    return pot_d
end

function b_potential(galactic_model::GalaxyModelM19, r)
    """
    The Hernquist bulge component gravitational potential defined in eq. (7) in
    Marchetti et al. (2019).

    Args:
        galactic_model (abstract type) : gmM19 model parameters.
        r (np.ndarray): distance in the galactic disk from the galactic centre in [kpc].

    Returns:
        (np.ndarray): value of the bulge potential in [erg/g].
    """

    pot_b = -G * galactic_model.M_b ./ ((galactic_model.r_b .+ r) .* KPC_TO_CM)

    return pot_b
end

function n_potential(galactic_model::GalaxyModelM19, r)
    """
    The Hernquist nucleus component gravitational potential defined in eq. (7) in
    Marchetti et al. (2019).

    Args:
        galactic_model (abstract type) : gmM19 model parameters.
        r (np.ndarray): distance in the galactic disk from the galactic centre in [kpc].

    Returns:
        (np.ndarray): value of the bulge potential in [erg/g].
    """

    pot_n = -G * galactic_model.M_n ./ ((galactic_model.r_n .+ r) .* KPC_TO_CM)

    return pot_n
end

function h_potential(galactic_model::GalaxyModelM19, r)
    """
    The Navarro-Frenk-White halo component gravitational potential defined in
    eq. (9) in Marchetti et al. (2019).

    Args:
        galactic_model (abstract type) : gmM19 model parameters.
        r (np.ndarray): distance in the galactic disk from the galactic centre in [kpc].

    Returns:
        (np.ndarray): value of the halo potential in [erg/g].
    """

    pot_h = -G * galactic_model.M_h ./ (r .* KPC_TO_CM) .* log1p.(r ./ galactic_model.r_h)

    return pot_h
end

function MW_potential(galactic_model::GalaxyModelM19, r, z)
    """
    Total Milky Way gravitational potential in Marchetti et al. (2019).

    Args:
        galactic_model (abstract type) : gmM19 model parameters.
        r (np.ndarray): distance in the galactic disk from the galactic centre in [kpc].
        z (np.ndarray): height from the galactic disk in [kpc].

    Returns:
        (np.ndarray): value of the Galactic potential in [erg].
    """
    
    MW_pot = (
        d_potential(galactic_model, r, z)
        + b_potential(galactic_model, r)
        + n_potential(galactic_model, r)
        + h_potential(galactic_model, r)
    )

    return MW_pot
end


#= Galactic model FK06 functions=#

function shape_parameter(galactic_model::GalaxyModelFK06, z)
    """
    Shape parameter for the disk-halo potential from Carlberg & Innamen
    (1987) and its derivative with respect to the height z from the galactic
    disk. First term in the denominator of eq. (13) in Faucher-Giguère &
    Kaspi (2006).

    Args:
        galactic_model (abstract type) : gmM19 model parameters.
        z (float): height from the galactic disk in [kpc].

    Returns:
        (float, float): value of the shape parameter and its derivative with
        respect to z.
    """
    K = (
        galactic_model.a_d
        .+ galactic_model.beta[1] * sqrt.(z.*z .+ galactic_model.h[1] * galactic_model.h[1])
        .+ galactic_model.beta[2] * sqrt.(z.*z .+ galactic_model.h[2] * galactic_model.h[2])
        .+ galactic_model.beta[3] * sqrt.(z.*z .+ galactic_model.h[3] * galactic_model.h[3])
    )

    dK_dz = (
        galactic_model.beta[1] * z ./ sqrt.(z.*z .+ galactic_model.h[1] * galactic_model.h[1])
        .+ galactic_model.beta[2] * z ./ sqrt.(z.*z .+ galactic_model.h[2] * galactic_model.h[2])
        .+ galactic_model.beta[3] * z ./ sqrt.(z.*z .+ galactic_model.h[3] * galactic_model.h[3])
    )

    return K, dK_dz
end


function r_z_derivatives_dh_potential(galactic_model::GalaxyModelFK06, r, z)
    """
    Derivative with respect to r and z of the disk-halo component gravitational
    potential defined in eq. (14) in Faucher-Giguère & Kaspi (2006).

    Args:
        galactic_model (abstract type) : gmFK06 model parameters.
        r (float): distance in the galactic disk from the galactic centre in [kpc].
        z (float): height from the galactic disk in [kpc].

    Returns:
        (float, float): derivative with respect to r and z of the disk-halo
        potential.
    """
    
    K, dK_dz = shape_parameter(galactic_model, z)
    
    dpot_dh_dr = (
        G_KPC_YR
        * galactic_model.M_dh
        .* r
        .* (K.*K .+ galactic_model.b_dh*galactic_model.b_dh .+ r.*r) .^ (-1.5)
    )
    
    dpot_dh_dz = (
        G_KPC_YR
        * galactic_model.M_dh
        .* (K.*K .+ galactic_model.b_dh*galactic_model.b_dh .+ r.*r) .^ (-1.5)
        .* K
        .* dK_dz
    )

    return dpot_dh_dr, dpot_dh_dz
end

function r_derivative_b_potential(galactic_model::GalaxyModelFK06, r)
    """
    Derivative with respect to r of the bulge component gravitational potential
    defined in eq. (15) in Faucher-Giguère & Kaspi (2006).

    Args:
        galactic_model (abstract type) : gmFK06 model parameters.
        r (float): distance in the galactic disk from the galactic centre in [kpc].

    Returns:
        float: derivative with respect to r of the bulge potential.
    """

    dpot_b_dr = (
        G_KPC_YR * galactic_model.M_b .* r .* (galactic_model.b_b*galactic_model.b_b .+ r.*r) .^ (-1.5)
    )

    return dpot_b_dr
end

function r_derivative_n_potential(galactic_model::GalaxyModelFK06, r)
    """
    Derivative with respect to r of the nucleus component gravitational potential
    defined in eq. (15) in Faucher-Giguère & Kaspi (2006).

    Args:
        galactic_model (abstract type) : gmFK06 model parameters.
        r (float): distance in the galactic disk from the galactic centre in [kpc].

    Returns:
        float: derivative with respect to r of the nucleus potential.
    """
    
    dpot_n_dr = (
        G_KPC_YR * galactic_model.M_n .* r .* (galactic_model.b_n*galactic_model.b_n .+ r.*r) .^ (-1.5)
    )

    return dpot_n_dr
end


function cylind_coord_gradient_mw_potential(galactic_model::GalaxyModelFK06, r, z)
    """
    Gradient in cylindrical coordinates of the Milky Way gravitational potential
    defined in eq. (13) in Faucher-Giguère & Kaspi (2006).

    Args:
        galactic_model (abstract type) : gmFK06 model parameters.
        r (float): distance in the galactic disk from the galactic centre in [kpc].
        z (float): height from the galactic disk in [kpc].

    Returns:
        (array): gradient of the galactic potential in cylindrical
        coordinates.
    """

    dpot_dh_dr, dpot_dh_dz = r_z_derivatives_dh_potential(galactic_model, r, z)
    
    dpot_b_dr = r_derivative_b_potential(galactic_model, r)
    
    dpot_n_dr = r_derivative_n_potential(galactic_model, r)

    dpot_mw_dr = dpot_dh_dr + dpot_b_dr + dpot_n_dr
    dpot_mw_dphi = 0.0
    dpot_mw_dz = dpot_dh_dz

    return dpot_mw_dr, dpot_mw_dphi, dpot_mw_dz
end

function dh_potential(galactic_model::GalaxyModelFK06, r, z)
    """
    The disk-halo component gravitational potential defined in eq. (14) in
    Faucher-Giguère & Kaspi (2006).

    Args:
        galactic_model (abstract type) : gmFK06 model parameters.
        r (np.ndarray): distance in the galactic disk from the galactic centre in [kpc].
        z (np.ndarray): height from the galactic disk in [kpc].

    Returns:
        (np.ndarray): value of the disk-halo potential in [erg/g].
    """

    K, _ = shape_parameter(galactic_model, z)

    pot_dh = (
        -G
        * galactic_model.M_dh
        ./ (sqrt.(K.*K .+ galactic_model.b_dh*galactic_model.b_dh .+ r.*r) * KPC_TO_CM)
    )

    return pot_dh
end

function b_potential(galactic_model::GalaxyModelFK06, r)
    """
    The bulge component gravitational potential defined in eq. (15) in
    Faucher-Giguère & Kaspi (2006).

    Args:
        galactic_model (abstract type) : gmFK06 model parameters.
        r (np.ndarray): distance in the galactic disk from the galactic centre in [kpc].

    Returns:
        (np.ndarray): value of the bulge potential in [erg/g].
    """

    pot_b = -G * galactic_model.M_b ./ (sqrt.(galactic_model.b_b*galactic_model.b_b .+ r.*r) * KPC_TO_CM)

    return pot_b
end

function n_potential(galactic_model::GalaxyModelFK06, r)
    """
    The nucleus component gravitational potential defined in eq. (15) in
    Faucher-Giguère & Kaspi (2006).

    Args:
        galactic_model (abstract type) : gmFK06 model parameters.
        r (np.ndarray): distance in the galactic disk from the galactic centre in [kpc].

    Returns:
        (np.ndarray): value of the nucleus potential.
    """

    pot_n = -G * galactic_model.M_n ./ (sqrt.(galactic_model.b_n*galactic_model.b_n .+ r.*r) * KPC_TO_CM)

    return pot_n
end

function MW_potential(galactic_model::GalaxyModelFK06, r, z)
    """
    Total Milky Way gravitational potential defined in eq. (13) in
    Faucher-Giguère & Kaspi (2006).

    Args:
        galactic_model (abstract type) : gmFK06 model parameters.
        r (np.ndarray): distance in the galactic disk from the galactic centre in [kpc].
        z (np.ndarray): height from the galactic disk in [kpc].

    Returns:
        (np.ndarray): value of the Galactic potential in [erg].
    """

    MW_pot = (
        dh_potential(galactic_model, r, z) .+ b_potential(galactic_model, r) .+ n_potential(galactic_model, r)
    )

    return MW_pot
end

function total_energy(v, r, z)
    """
    Value of the total energy of the system, sum of the total kinetic energy and
    the total gravitational potential energy. We assume here that all the stars
    have unit mass.

    Args:
        v (array): array of magnitudes of the speed of the stars in [km/s].
        r (array): array of distances from the galactic axis in [kpc].
        z (array): array of distances from the galactic disk in [kpc].

    Returns:
        (float): value of the total energy of the system in [erg].
    """
    # Convert speeds into [cm/s].
    v = v * KM_TO_CM
    
    tot_kin_energy = 0.5 * sum(v.*v)
    
    tot_pot_energy = sum(MW_potential(galactic_model, r, z))

    tot_energy = tot_kin_energy + tot_pot_energy

    return tot_energy
end

function total_angular_momentum_z(v_phi, r)
    """
    Value of the z-component of the total angular momentum of the system, which is a
    conserved quantity in axisymmetric potentials. We assume here that all the stars
    have unit mass.

    Args:
        v_phi (array): array of magnitudes of the orbital speed of the stars in [km/s].
        r (array): array of distances from the galactic axis in [kpc].

    Returns:
        (float): value of the total energy of the system in [erg].
    """
    # Convert speeds into [cm/s].
    v_phi = v_phi * KM_TO_CM
    # Convert distances into [cm].
    r = r * KPC_TO_CM

    L_z = sum(r .* v_phi)

    return L_z
end
