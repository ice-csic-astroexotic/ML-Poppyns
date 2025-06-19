# Information on the magneto-thermal evolution simulations

For the magneto-thermal simulations in `BSk24_multi_heavy-envelope` the following set-up was employed:
1. The equation of state is BSk24 with a NS mass of 1.4 Msun and radius of 12.59 km. 
2. The impurity parameter in the pasta layer is fixed to 100. For the impurity in the outer and inner crust
(excluding the pasta layer), the fits of Carreau et al. (2020) have been used (see Figure 5 in that paper). 
3. The heavy envelope model is taken from Potekhin et al. (2015). 
4. Superfluid and superconducting gap parametrizations are taken from Ho et al. (2015): SFB for crustal neutrons, TToa for core neutrons and CCDKp for core protons.
5. The dipole and toroidal magnetic field components are set to have the same strength, but not same magnetic energy.
   The polar quadrupole strength is additionally set to be two times the dipole strength. 
   As a result, 90% of the magnetic energy is contained within the quadrupolar component, while the dipolar and toroidal field strengths contain the remaining 10% (9% in the dipolar and 1% in the toroidal).
   Our choice of quadrupolar strength is motivated by Fig. 7 of [Reboul-Salze et al. 2021](https://ui.adsabs.harvard.edu/abs/2021A%26A...645A.109R/abstract) (but see also [Dehman et al. 2023](https://ui.adsabs.harvard.edu/abs/2023MNRAS.523.5198D/abstract)), 
   which shows that the l=1 and l=2 poloidal and toroidal components contribute roughly 10% to the total magnetic energy, with the latter dominating over the former. 
   The remaining 90% of the energy is concentrated in the higher-order multipoles. 
   To reflect the latter, we make the simplifying assumption that all the energy in the multipolar components is concentrated in the polar quadrupolar component, 
   which is roughly achieved with the assumptions made above.