# Information on the magneto-thermal evolution simulations

For the magneto-thermal simulations in `BSk24_dip-tor_light-envelope` we used the results of the 2D code in 
[Viganò et al. (2021)](https://ui.adsabs.harvard.edu/abs/2021CoPhC.26508001V/abstract) where the following set-up was employed:
1. The equation of state is BSk24 with a NS mass of 1.4 Msun and radius of 12.59 km. 
2. The impurity parameter in the pasta layer is fixed to 100. For the impurity in the outer and inner crust
(excluding the pasta layer), the fits of Carreau et al. (2020) have been used (see Figure 5 in that paper). 
3. The light envelope model is taken from Potekhin et al. (2015). 
4. Superfluid and superconducting gap parametrizations are taken from Ho et al. (2015): SFB for crustal neutrons, TToa for core neutrons and CCDKp for core protons.
5. The polar and toroidal magnetic field components with moment l=1 are set to have the same strength, but not same magnetic energy.
   With this configuration the polar dipole component contains 90% of the total magnetic energy, the remaining 10% is in the toroidal component.
6. The model for computing the crust failure events is described in [Dehman et al. (2020)](https://ui.adsabs.harvard.edu/abs/2020ApJ...902L..32D/abstract). The files with the crust failure information contain the following columns:
   - the time in [yr] when the failure event occurs;
   - the total magnetic energy in [erg] dissipated during the failure;
   - the position in polar coordinates ($\theta$ in [rad] and $r$ in [km]) where the failure event occurs;
   - the total volume in [cm$^3$] of the crust affected by the failure;
   - the timestep in [yr] for the magnetic field evolution that is used in the simulation.