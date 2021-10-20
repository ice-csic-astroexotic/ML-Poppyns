"""
Model for the pulsar gamma surveys.
We consider the following surveys:
1) fermi LAT
Authors:
        Michele Ronchi (ronchi@ice.csic.es)
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

import abc
from typing import Tuple

import numpy as np

import pypopsyn.simulator.basics.constants as const


def detect(S_gamma: np.ndarray, detected_radio: np.ndarray) -> np.ndarray:
    """
    Simulate a detection: if the measured gamma flux surpasses the threshold of the survey
    then the pulsar is detected at high energy.
    We assume that for blind searches the detection threshold is higher as in Johnston et al. 2020.

    Args:
        S_gamma (np.ndarray): gamma flux in [erg cm^(-2) s^(-1)].
        detected_radio (np.ndarray): boolean array indicating if the stars are already detected in radio.

    Returns:
        (np.ndarray): array of boolean variables: true if the pulsar is detected, false if not.
    """

    # Gamma flux threshold for blind searches in [erg cm^(-2) s^(-1)] (see Johnston et al. 2020).
    S_gamma_th_blind = 16.0e-12
    S_gamma_th = 4.0e-12

    detected = S_gamma > S_gamma_th_blind
    detected[detected_radio] = S_gamma[detected_radio] > S_gamma_th

    return detected
