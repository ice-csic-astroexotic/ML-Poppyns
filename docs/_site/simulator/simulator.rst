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

Initialising Mock Population
#############################

Configuration File
*******************

.. automodule:: pypopsyn.simulator.configuration
  :members: configuration

Initial NS Population
**********************

.. automodule:: pypopsyn.simulator.initial_population
  :members: initial_population

Initial NS Population from electron density model
*************************************************

.. automodule:: pypopsyn.simulator.initial_population_edm
  :members: initial_population_edm

Basics
#######

Random Sampler
**************

.. automodule:: simulator.basics.random_sampler
  :members: random_sampler

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

Magneto-rotational Evolution with B-field Evolution from Magneto-thermal simulations
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