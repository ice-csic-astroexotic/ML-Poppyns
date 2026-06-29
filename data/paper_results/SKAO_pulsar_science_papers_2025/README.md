The pulsar population synthesis results presented in the 2025 SKAO pulsar science white papers (see 
[Keane et al. (2025)](https://ui.adsabs.harvard.edu/abs/2025OJAp....854256K/abstract) and 
[Levin et al. (2025)](https://ui.adsabs.harvard.edu/abs/2025OJAp....854653L/abstract)) and the 2026 SKAO 
Science Book are based on the sequential inference method and the luminosity prescription of 
[Pardo-Araujo et al. 2025](https://ui.adsabs.harvard.edu/abs/2025A%26A...696A.114P/abstract) used to derive seven 
best-fit parameters for a mean spectral index of -1.45 and standard deviation of 0.15, namely:

- cfg["B_initial_log10_mean"]: float = 13.10 
- cfg["B_initial_log10_sigma"]: float = 0.48
- cfg["P_initial_log10_mean"]: float = -0.71 
- cfg["P_initial_log10_sigma"]: float = 0.58
- cfg["a_late"]: float = -0.92
- cfg["L_radio_log10_mean"]: float = 25.67
- cfg["epsilon_L"]: float = 0.70

Because of SKAO's increased sensitivity compared to existing radio facilities, and hence its ability to detect older, 
fainter radio pulsars, we need to evolve our synthetic neutron stars for longer than was necessary for PMPS, SMPS and 
HTRU. In particular, the computation time for evolving a given population increased from the order of a few hours to 
several days. To mitigate this issue and allow the exploration of possible survey specifications for SKAO, we have 
evolved the underlying population for the above parameters only once using the `simulate_population_full.py` script 
for 4e7 neutron stars with a maximum age of 2e9 yrs. This corresponds to a realistic birth rate of 2 and reproduces 
roughly the observed numbers in existing radio surveys. Once the population has been evolved, we store the full results 
and then apply survey specifications using the`simulate_population_survey_only.py` script for all existing as well as 
possible SKAO configurations.

Because of the increase in evolution time, the resulting data files are also much larger than those created for 
existing radio surveys, making it unfeasible to upload these to GitHub. Instead, we have created a Zenodo repository
with the full dataset required to reproduce the results of the evolutionary population synthesis approach presented in 
the above papers. 

To run the SKAO related jupyter notebooks in the `tutorials` directory, please download and unzip the data and store it 
in `data/paper_results/SKAO_pulsar_science_papers_2025`. The dataset contains the three directories. The first one 
looks as follows:

- `evolultionary_simulations_si145` contains `PardoAraujoL_br2_2e9` with:
  - `configuration.json` describes all relevant simulation parameters.
  - `final_population.pkl.gz` summarises the properties of the final evolved population.
  - `initial_population.pkl.gz` summarises the properties of the initial neutron star population.
  - `profile.json` and `profile.log` contain timing information for the simulation.
  - `survey_HTRU_high_results.pkl.gz` summarises the properties of the pulsars observed in HTRU high.
  - `survey_HTRU_low_mid_results.pkl.gz` summarises the properties of the pulsars observed in HTRU low-mid.
  - `survey_PMPS_results.pkl.gz` summarises the properties of the pulsars observed in PMPS.
  - `survey_SMPS_results.pkl.gz` summarises the properties of the pulsars observed in SMPS.
  
In addition, a second directory `surveys_census_simulations_si145` contains three subdirectories for three
different survey options for the SKA-Low and SKA-Mid telescopes in the AA* and AA4 configurations:

- `survey_option1`: Low: `|b| > 15 deg`; Mid Band 1: `15 deg > |b|> 5 deg`; Mid Band 2: `|b| < 5 deg`
- `survey_option2`: Low: `|b|> 10 deg`; Mid Band 1: -; Mid Band 2: `|b| < 10 deg`
- `survey_option3`: Low: `|b| > 5 deg`; Mid Band 1: -; Mid Band 2: `|b| < 5 deg`

Each of these directories contains 10 subfolders labeled `run_i` for `i = 1, ..., 10` that contain the results of the
population synthesis runs for each of the considered surveys. Note that we run the detection pipeline repeatedly
because of the stochastic nature of our simulation framework, which allows us to determine errors on the number of 
detected sources. As the underlying neutron star population is fixed, the variability does 

The final directory `survey_parameters` contains all relevant survey parameters as JSON files for reproducibility 
purposes, i.e., 

- 

Note that we adjusted the different observing latitudes for SKA-Low and SKA-Mid according to the above three options 
by hand for each run.