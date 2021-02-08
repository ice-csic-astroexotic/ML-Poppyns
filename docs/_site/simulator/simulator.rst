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

Basics
#######

Configuration
**************

.. automodule:: pypopsyn.simulator.configuration
  :members: configuration

CDF Calculator
***************

.. automodule:: simulator.cdf_calculator
  :members: cdf_calculator

Constants
*************

.. automodule:: simulator.constants
  :members: constants

Coordinate Conversions
************************

.. automodule:: simulator.coordinate_conversions
  :members: coordinate_conversions


Stellar Dynamics
###################

Dynamical Evolution
*******************

.. automodule:: simulator.dynamical_evolution
  :members: dynamical_evolution

Galactic Model
**************

.. automodule:: simulator.galactic_model
  :members: galactic_model

Initial Position
****************

.. automodule:: simulator.initial_position
  :members: initial_position

Initial Velocity
*******************

.. automodule:: simulator.initial_velocity
  :members: initial_velocity


Initial NS Population
#####################

.. automodule:: simulator.initial_population
  :members: