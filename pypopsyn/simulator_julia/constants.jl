"""
Constants module in Julia.

    Authors:

        Borja Miñano (borja.minano@uib.es)
        Celsa Pardo Araujo (pardo @ csic.es)
        Michele Ronchi (ronchi @ ice.csic.es)

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

# Defining global constants needed for the dynamical evolution.

const M_SUN = 2.0e33
const G = 6.67e-8
const KPC_TO_CM = 3.08567758e21
const YR_TO_S = 3600.0 * 24 * 365
const G_KPC_YR = (G / (KPC_TO_CM ^ 3) * YR_TO_S ^ 2)
const KPC_TO_KM = 3.08567758e16
const KM_TO_CM = 100000.0