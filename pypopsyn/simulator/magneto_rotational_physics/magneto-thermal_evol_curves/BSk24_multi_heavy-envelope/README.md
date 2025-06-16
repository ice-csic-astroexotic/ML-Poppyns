# Information on the magneto-thermal evolution simulations

For the magneto-thermal simulations in `BSk24_multi_heavy-envelope` the following set-up was employed:
1. The equation of state is BSk24 with a NS mass of 1.4 Msun and radius of 12.59 km. 
2. The impurity parameter in the pasta layer is fixed to 100. For the impurity in the outer and inner crust
(excluding the pasta layer), the fits of Carreau et al. (2020) have been used (see Figure 5 in that paper). 
3. The heavy envelope model is taken from Potekhin et al. (2015). 
4. Superfluid and superconducting gap parametrizations are taken from Ho et al. (2015): SFB for crustal neutrons, TToa for core neutrons and CCDKp for core protons.
5. The dipole and toroidal magnetic field components are set to have the same strength (but not same magnetic energy). The polar quadrupole strength is set to be two times the dipole strength. 
Our choice of quadrupolar strength is motivated by Fig. 7 of [Reboul-Salze et al. 2021](https://ui.adsabs.harvard.edu/abs/2021A%26A...645A.109R/abstract) (but see also [Dehman et al. 2023](https://ui.adsabs.harvard.edu/abs/2023MNRAS.523.5198D/abstract)), which shows that the initial configuration for the polar dipole and first order moment of the toroidal component carry around 90% of the total magnetic energy. The remaining 10% of the energy is concentrated in the higher-order multipoles, which we neglect in our simplified model here.
We assume that all the energy in the multipolar components is concentrated in the polar quadrupolar component.