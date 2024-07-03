*********
Simulator
*********

Telemetry
#########

.. note:: Commit a6963f4e6b440124c340c348ef2bd0e05a88a6d5 (23/09/2020).

This telemetry comes from the following setup:

  * Ubuntu 20.04.
  * Intel(R) Core(TM) i5-9600K CPU @ 3.70GHz.
  * 16 GiB RAM DDR4.
  * 512 GiB SSD Samsung EVO.

+--------------------+--------------------+--------------------+
| Context            | Time [s]           | Cumulative [s]     |
+====================+====================+====================+
| [InitialPopulation]                                          |
+--------------------+--------------------+--------------------+
| [Initial]          |   1.2920           |   1.2920           |
+--------------------+--------------------+--------------------+
| [Energy]           |   1.4685           |   2.7606           |
+--------------------+--------------------+--------------------+
| [Export]           |   0.2307           |   2.9912           |
+--------------------+--------------------+--------------------+
| [EvolvePopulation]                                           |
+--------------------+--------------------+--------------------+
| [Evolution]        |  12.8988           |  12.8988           |
+--------------------+--------------------+--------------------+
| [Energy]           |   0.0026           |  12.9014           |
+--------------------+--------------------+--------------------+
| [Export]           |   0.3849           |  13.2864           |
+--------------------+--------------------+--------------------+
|                                                              |
+--------------------+--------------------+--------------------+
| Total Time [s]:  16.2776                                     |
+--------------------+--------------------+--------------------+

.. note:: Commit 3f9c4cbd6cb211a26922f5e1793bce7028dfcbc3 (04/02/2021).

This telemetry comes from the following setup:

  * Ubuntu 18.04.
  * Intel(R) Core(TM) i7-10510U CPU @ 1.80GHz × 8.
  * 16 GiB RAM DDR4.
  * 1024 GiB.

+--------------------+--------------------+--------------------+
| Context            | Time [s]           | Cumulative [s]     |
+====================+====================+====================+
| [InitialPopulation]                                          |
+--------------------+--------------------+--------------------+
| [Initial]          |   1.7934           |   1.7934           |
+--------------------+--------------------+--------------------+
| [Energy]           |   1.7615           |   3.5549           |
+--------------------+--------------------+--------------------+
| [Export]           |   0.3486           |   3.9034           |
+--------------------+--------------------+--------------------+
| [EvolvePopulation]                                           |
+--------------------+--------------------+--------------------+
| [Evolution]        |  15.1637           |  15.1637           |
+--------------------+--------------------+--------------------+
| [Energy]           |   0.0102           |  15.1739           |
+--------------------+--------------------+--------------------+
| [Export]           |   0.5263           |  15.7002           |
+--------------------+--------------------+--------------------+
|                                                              |
+--------------------+--------------------+--------------------+
| Total Time [s]:  15.7038                                     |
+--------------------+--------------------+--------------------+

.. note:: Commit 373350bb30a45601cc40f22de78d0ee5e1bc9cab (14/04/2021).

This telemetry comes from the following setup:

  * Ubuntu 18.04.
  * Intel(R) Core(TM) i7-10510U CPU @ 1.80GHz × 8.
  * 16 GiB RAM DDR4.
  * 1024 GiB.

+-------------------------------------------------------------+--------------------+--------------------+
| Context                                                     | Time [s]           | Cumulative [s]     |
+=============================================================+====================+====================+
| [InitialPopulation]                                                                                   |
+-------------------------------------------------------------+--------------------+--------------------+
| [Initial position and velocity]                             |   0.4506           |   0.4506           |
+-------------------------------------------------------------+--------------------+--------------------+
| [Energy]                                                    |   1.8407           |   2.2913           |
+-------------------------------------------------------------+--------------------+--------------------+
| [Initial field strengths, misalignment angles and periods]  |   0.5333           |   2.8246           |
+-------------------------------------------------------------+--------------------+--------------------+
| [Initial period derivatives]                                |   0.0356           |   2.8602           |
+-------------------------------------------------------------+--------------------+--------------------+
| [Export]                                                    |   0.5439           |   3.4041           |
+-------------------------------------------------------------+--------------------+--------------------+
| [EvolvePopulation]                                                                                    |
+-------------------------------------------------------------+--------------------+--------------------+
| [Dynamic evolution]                                         |  16.4773           |  16.4773           |
+-------------------------------------------------------------+--------------------+--------------------+
| [Energy]                                                    |   0.0099           |  16.4872           |
+-------------------------------------------------------------+--------------------+--------------------+
| [Final field strengths, misalignment angles and periods]    |  39.1625           |  55.6497           |
+-------------------------------------------------------------+--------------------+--------------------+
| [Final period derivatives]                                  |   0.0355           |  55.6853           |
+-------------------------------------------------------------+--------------------+--------------------+
| [Export]                                                    |   0.7062           |  56.3914           |
+-------------------------------------------------------------+--------------------+--------------------+
|                                                                                                       |
+-------------------------------------------------------------+--------------------+--------------------+
| Total Time [s]:  59.8273                                                                              |
+-------------------------------------------------------------+--------------------+--------------------+

.. note:: Commit 144046bfbdbb18ccd9ddcf468254671d1d88f360 (12/06/2024).

This telemetry comes from the following setup:

  * Ubuntu 18.04.
  * Intel(R) Core(TM) i7-10510U CPU @ 1.80GHz × 8.
  * 16 GiB RAM DDR4.
  * 1024 GiB.

Timing profile for the :code:`simulate_population_full.py` script with the default configuration in :code:`config_simulator.py`.

+-------------------------------------------------------------+--------------------+--------------------+
| Context                                                     | Time [s]           | Cumulative [s]     |
+=============================================================+====================+====================+
| [InitialPopulation]                                                                                   |
+-------------------------------------------------------------+--------------------+--------------------+
| [Initial position and velocity]                             |   2.5658           |   2.5658           |
+-------------------------------------------------------------+--------------------+--------------------+
| [Initial Energy]                                            |   1.6669           |   4.2327           |
+-------------------------------------------------------------+--------------------+--------------------+
| [Initial angular momentum]                                  |   0.4089           |   4.6416           |
+-------------------------------------------------------------+--------------------+--------------------+
| [Initial field strengths, misalignment angles and periods]  |   0.0736           |   4.7152           |
+-------------------------------------------------------------+--------------------+--------------------+
| [Initial period derivatives]                                |   0.0827           |   4.7979           |
+-------------------------------------------------------------+--------------------+--------------------+
| [Export]                                                    |   0.9706           |   5.7685           |
+-------------------------------------------------------------+--------------------+--------------------+
| [DynamicalEvolution]                                                                                  |
+-------------------------------------------------------------+--------------------+--------------------+
| [Dynamical evolution]                                       | 105.2335           | 105.2335           |
+-------------------------------------------------------------+--------------------+--------------------+
| [Final energy]                                              |   0.0147           | 105.2483           |
+-------------------------------------------------------------+--------------------+--------------------+
| [Final angular momentum]                                    |   0.0018           | 105.2501           |
+-------------------------------------------------------------+--------------------+--------------------+
| [MagnetoRotationalEvolution]                                                                          |
+-------------------------------------------------------------+--------------------+--------------------+
| [Final field strengths, misalignment angles and periods]    | 209.5308           | 209.5308           |
+-------------------------------------------------------------+--------------------+--------------------+
| [Final period derivatives]                                  |   0.1027           | 209.6336           |
+-------------------------------------------------------------+--------------------+--------------------+
| [RadioEmission]                                                                                       |
+-------------------------------------------------------------+--------------------+--------------------+
| [Radio emission]                                            |   0.0977           |   0.0977           |
+-------------------------------------------------------------+--------------------+--------------------+
| [RadioDetection]                                                                                      |
+-------------------------------------------------------------+--------------------+--------------------+
| [Total sky coverage]                                        |   0.0108           |   0.0108           |
+-------------------------------------------------------------+--------------------+--------------------+
| [DM computation]                                            |  37.7504           |  37.7504           |
+-------------------------------------------------------------+--------------------+--------------------+
| [Radio surveys detection]                                   |   0.2871           |  38.0484           |
+-------------------------------------------------------------+--------------------+--------------------+
| [Export]                                                    |   2.3866           |  40.4359           |
+-------------------------------------------------------------+--------------------+--------------------+
|                                                                                                       |
+-------------------------------------------------------------+--------------------+--------------------+
| Total Time [s]:  361.1944                                                                             |
+-------------------------------------------------------------+--------------------+--------------------+

Timing profile for the :code:`simulate_population_dyn.py` script with the default configuration in :code:`config_simulator.py`.

+-------------------------------------------------------------+--------------------+--------------------+
| Context                                                     | Time [s]           | Cumulative [s]     |
+=============================================================+====================+====================+
| [InitialPopulation]                                                                                   |
+-------------------------------------------------------------+--------------------+--------------------+
| [Initial position and velocity]                             |   2.6854           |   2.6854           |
+-------------------------------------------------------------+--------------------+--------------------+
| [Initial Energy]                                            |   2.0433           |   4.7287           |
+-------------------------------------------------------------+--------------------+--------------------+
| [Initial angular momentum]                                  |   0.3334           |   5.0620           |
+-------------------------------------------------------------+--------------------+--------------------+
| [EvolvePopulation]                                                                                    |
+-------------------------------------------------------------+--------------------+--------------------+
| [Dynamical evolution]                                       | 122.3311           | 122.3311           |
+-------------------------------------------------------------+--------------------+--------------------+
| [Final energy]                                              |   0.0200           | 122.3511           |
+-------------------------------------------------------------+--------------------+--------------------+
| [Final angular momentum]                                    |   0.0032           | 122.3544           |
+-------------------------------------------------------------+--------------------+--------------------+
| [Export]                                                    |   3.1363           | 125.4907           |
+-------------------------------------------------------------+--------------------+--------------------+
|                                                                                                       |
+-------------------------------------------------------------+--------------------+--------------------+
| Total Time [s]:  130.5555                                                                             |
+-------------------------------------------------------------+--------------------+--------------------+

Timing profile for the :code:`simulate_population_magrot_det.py` script with the default configuration in :code:`config_simulator.py`.
Here, we report the average time and standard deviation over 7 loops for:

- loading a dynamical database with a total of 300000 neutron stars and sampling these stars during the detection loop with a batch size of 100000.
- performing the magneto-rotational evolution of the sampled stars and applying the detection filters.

Note that in the remaining couple of loops of the detection procedure, the time to perform these calculations is shorter due to the reduced batch size.
We do not take into account these loops to compute the following timing statistics.

+-------------------------------------------------------------+--------------------+
| Context                                                     | Time [s]           |
+=============================================================+====================+
| [InitialPopulation]                                                              |
+-------------------------------------------------------------+--------------------+
| [LoadPopulationDynamics] mean (std)                         |   0.5283 (0.0146)  |
+-------------------------------------------------------------+--------------------+
| [SimulatePopulationDetection] mean (std)                    |   86.9900 (9.6250) |
+-------------------------------------------------------------+--------------------+
| [Export]                                                    |   0.0540           |
+-------------------------------------------------------------+--------------------+
|                                                                                  |
+-------------------------------------------------------------+--------------------+
| Total Time [s]: 634.5910                                                         |
+-------------------------------------------------------------+--------------------+

Basics
#######

Constants
*************

.. automodule:: simulator.basics.constants
  :members: constants

Interstellar Medium
###################

Galactic Electron Density Model
*******************************

.. automodule:: simulator.interstellar_medium.e_density_model
  :members: e_density_model

Hydrogen Density Model
**********************

.. automodule:: simulator.interstellar_medium.nh_model
  :members: nh_model

X-ray Absorption Cross section
******************************

.. automodule:: simulator.interstellar_medium.xray_abs_cross_section
  :members: xray_abs_cross_section

Magneto-rotational Physics
#############################

Initial Period Distribution
****************************

.. automodule:: simulator.magneto_rotational_physics.initial_period
  :members: initial_period

Magnetic Field Derivative
**************************

.. automodule:: simulator.magneto_rotational_physics.magnetic_field_derivative
  :members: magnetic_field_derivative

Magneto-rotational Evolution
****************************

.. automodule:: simulator.magneto_rotational_physics.magneto_rotational_evolution
  :members: magneto_rotational_evolution

Magneto-rotational Evolution with B-field Evolution from Magneto-thermal Simulations
************************************************************************************

.. automodule:: simulator.magneto_rotational_physics.magneto_rotational_evolution_fit
  :members: magneto_rotational_evolution_fit

Misalignment Angle Derivative
*****************************

.. automodule:: simulator.magneto_rotational_physics.misalignment_angle_derivative
  :members: misalignment_angle_derivative

Period Derivative
*****************

.. automodule:: simulator.magneto_rotational_physics.period_derivative
  :members: period_derivative

Multi-band Electromagnetic Emission
###################################

Emission Radio
**************

.. automodule:: simulator.multiband_emission.emission_radio
  :members: emission_radio

Multi-band Surveys
##################

Survey Radio
************

.. automodule:: simulator.multiband_surveys.survey_radio
  :members: survey_radio

Stellar Dynamics
###################

Coordinate Conversions
************************

.. automodule:: simulator.stellar_dynamics.coordinate_conversions
  :members: coordinate_conversions

Dynamical Evolution
*******************

.. automodule:: simulator.stellar_dynamics.dynamical_evolution
  :members: dynamical_evolution

Galactic Model
**************

.. automodule:: simulator.stellar_dynamics.galactic_model
  :members: galactic_model

Initial Position
****************

.. automodule:: simulator.stellar_dynamics.initial_position
  :members: initial_position

Initial Velocity
*******************

.. automodule:: simulator.stellar_dynamics.initial_velocity
  :members: initial_velocity

Spiral Model
************

.. automodule:: simulator.stellar_dynamics.spiral_model
  :members: spiral_model

Initialising Mock Population
#############################

Configuration File
*******************

.. automodule:: pypopsyn.simulator.config_simulator
  :members: configuration

Initial NS Population from spiral-arm model
*******************************************

.. automodule:: pypopsyn.simulator.initial_population_sam
  :members: initial_population_sam

Initial NS Population from electron density model
*************************************************

.. automodule:: pypopsyn.simulator.initial_population_edm
  :members: initial_population_edm

Evolve NS Population
####################

Simulate only the dynamical evolution of a NS Population
********************************************************

.. automodule:: pypopsyn.simulator.simulate_population_dyn
  :members: simulate_population_dyn

Simulate the full evolution of a NS Population
**********************************************

.. automodule:: pypopsyn.simulator.simulate_population_full
  :members: simulate_population_full

Simulate the magneto-rotational evolution and detection of a NS Population
**************************************************************************

.. automodule:: pypopsyn.simulator.simulate_population_magrot_det
  :members: simulate_population_magrot_det