"""
Test for the magneto_rotational_physics/initial_period module.

    Authors:

        Vanessa Graber (graber @ ice.csic.es)

MIT License

Copyright (c) MAGNESIA (ICE-CSIC) 2020

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import pypopsyn.simulator.magneto_rotational_physics.initial_period as ipd
from pypopsyn.simulator.configuration import cfg


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
