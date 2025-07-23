"""
    Test for the initial_period module.

        Authors:

            Vanessa Graber (graber @ ice.csic.es)
"""

import pypopsyn.simulator.magneto_rotational_physics.initial_period as ipd
from pypopsyn.simulator.config_simulator import cfg

# Set the configuration parameters for testing purposes.
cfg["NS_number"]: int = 30
cfg["P_initial_mean"]: float = 0.3
cfg["P_initial_sigma"]: float = 0.2
cfg["P_initial_log10_mean"]: float = -0.67
cfg["P_initial_log10_sigma"]: float = 0.55


def test_pdf_period_normal():
    """
    Check that all initial periods are indeed positive and the resulting array
    has the correct length, corresponding to the number of pulsars in our sample.
    """

    P_initial_out = ipd.pdf_period_normal(
        cfg["P_initial_mean"], cfg["P_initial_sigma"], cfg["NS_number"]
    )

    assert len(P_initial_out) == cfg["NS_number"]
    assert (P_initial_out > 0).all()


def test_pdf_period_lognormal():
    """
    Check that the array of initial periods has the correct length, corresponding
    to the number of pulsars in our sample.
    """
    P_initial_out = ipd.pdf_period_lognormal(
        cfg["P_initial_log10_mean"],
        cfg["P_initial_log10_sigma"],
        cfg["NS_number"],
    )

    assert len(P_initial_out) == cfg["NS_number"]
