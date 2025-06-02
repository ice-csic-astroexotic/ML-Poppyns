# Information on the magneto-thermal evolution simulations

For the magneto-thermal simulations the following set-up was employed:
1. The equation of state is BSk24 with a NS mass of 1.4 Msun and radius of 12.59 km. 
2. The impurity parameter in the pasta layer is fixed to 100. For the impurity in the outer and inner crust
(excluding the pasta layer), the fits of Carreau et al. (2020) have been used (see Figure 5 in that paper). 
3. The envelope model is taken from Potekhin et al. (2015). 
4. Superfluid and superconducting gap parametrizations are taken from Ho et al. (2015): SFB for crustal neutrons, TToa for core neutrons and CCDKp for core protons.
5. The dipole and toroidal magnetic field components are set to have the same strength (but not same magnetic energy). The polar quadrupole strength is set to be 2 times the dipole strengths. This guarantees that ~90% of the magnetic energy in the crust is concentrated in that multipole component. This is doneto explore the impact of considering a multipole component for the magnetic field configuration. In Fig. 7 of [Reboul-Salze et al. 2021](https://ui.adsabs.harvard.edu/abs/2021A%26A...645A.109R/abstract) (but see also [Dehman et al. 2023](https://ui.adsabs.harvard.edu/abs/2023MNRAS.523.5198D/abstract)) it seems that the initial configuration the dipole and first order quadrupole hasaround the 10% of the total magnetic energy. All the rest is concentreted in higher order multipoles.
