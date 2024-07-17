# Telemetry

**NOTE:** Commit 144046bfbdbb18ccd9ddcf468254671d1d88f360 (12/06/2024).

This telemetry comes from the following setup:

* Ubuntu 18.04.
* Intel(R) Core(TM) i7-10510U CPU @ 1.80GHz × 8.
* 16 GiB RAM DDR4.
* 1024 GiB.

Timing profile for the `simulate_population_full.py` script with the default configuration in `config_simulator.py`.

| Context                                                    | Time [s] | Cumulative [s]    |
|------------------------------------------------------------|----------|-------------------|
| [InitialPopulation]                                        |          |                   |
| [Initial position and velocity]                            | 2.5658   | 2.5658            |
| [Initial Energy]                                           | 1.6669   | 4.2327            |
| [Initial angular momentum]                                 | 0.4089   | 4.6416            |
| [Initial field strengths, misalignment angles and periods] | 0.0736   |   4.7152          |
| [Initial period derivatives]                               | 0.0827   |   4.7979          |
| [Export]                                                   | 0.9706   |   5.7685          |
|                                                            |          |                   |
| [DynamicalEvolution]                                       |          |                   |
| [Dynamical evolution]                                      | 105.2335 | 105.2335          |
| [Final energy]                                             | 0.0147   | 105.2483          |
| [Final angular momentum]                                   | 0.0018   | 105.2501          |
|                                                            |          |                   |
| [MagnetoRotationalEvolution]                               |          |                   |
| [Final field strengths, misalignment angles and periods]   | 209.5308 | 209.5308          |
| [Final period derivatives]                                 | 0.1027   | 209.6336          |
| [RadioEmission]                                            |          |                   |
| [Radio emission]                                           | 0.0977   |   0.0977          |
|                                                            |          |                   |
| [RadioDetection]                                           |          |                   |
| [Total sky coverage]                                       | 0.0108   |   0.0108          |
| [DM computation]                                           | 37.7504  |  37.7504          |
| [Radio surveys detection]                                  | 0.2871   |  38.0484          |
| [Export]                                                   | 2.3866   |  40.4359          |
|                                                            |          |                   |
| Total Time                                                 |          |  361.1944         |

Timing profile for the `simulate_population_dyn.py` script with the default configuration in `config_simulator.py`.


| Context                         | Time [s]           | Cumulative [s]     |
|---------------------------------|--------------------|--------------------|
| [InitialPopulation]             |                    |                    |
| [Initial position and velocity] |   2.6854           | 2.6854             |
| [Initial Energy]                |   2.0433           | 4.7287             |
| [Initial angular momentum]      |   0.3334           | 5.0620             |
|                                 |                    |                    |
| [EvolvePopulation]              |                    |                    |
| [Dynamical evolution]           | 122.3311           | 122.3311           |
| [Final energy]                  |   0.0200           | 122.3511           |
| [Final angular momentum]        |   0.0032           | 122.3544           |
| [Export]                        |   3.1363           | 125.4907           |
|                                 |                    |                    |
| Total Time                      |                    | 130.5555           |


Timing profile for the `simulate_population_magrot_det.py` script with the default configuration in `config_simulator.py`.
Here, we report the average time and standard deviation over 7 loops for:

- loading a dynamical database with a total of 300000 neutron stars and sampling these stars during the detection loop with a batch size of 100000.
- performing the magneto-rotational evolution of the sampled stars and applying the detection filters.

Note that in the remaining couple of loops of the detection procedure, the time to perform these calculations is shorter due to the reduced batch size.
We do not take into account these loops to compute the following timing statistics.

| Context                                  | Time [s]         |
|------------------------------------------|------------------|
| [LoadPopulationDynamics] mean (std)      | 0.5283 (0.0146)  |
| [SimulatePopulationDetection] mean (std) | 86.9900 (9.6250) |
| [Export]                                 | 0.0540           |
|                                          |                  |
| Total Time                               | 634.5910         |