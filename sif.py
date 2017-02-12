""" This is the spin ice farm itself """
import numpy as np

# -*- coding: utf-8 -*-
__author__ = 'Sergei Liashko'

from scipy.signal import argrelextrema
import numpy as np
from numpy import pi
from numpy import sqrt
import json
import matplotlib.pyplot as plt
from functools import partial # we don't want to pass all parameters of the system
from sys import argv

import energy as e
import minimization as mm
import geometry as g
import paramsfactory

try:
    script, params_file, runtime_settings, vcoords_file = argv
except:
    print("ERROR: You didn't provide the params file for the programm", argv[0])

output_path="data/"
params = paramsfactory.load_hamiltonian_params(params_file)
n = params['n']
M = params['M']
V = params['V']
anisotropy_angles = params['anisotropy_angles']
K = params['K']
field_angle = params['field_angle']
H = params['H']
R = params['R']
aa = params['state_0']
bb = params['state_m']

nebsettings = paramsfactory.load_runtime_settings(runtime_settings)
(m,k_spring, dt, epsilon, c_i,f_i) = nebsettings['general_settings']
(m_min, k_spring_min, dt_min, epsilon_min ,c_i_min,f_i_min)=nebsettings['min_settings']
(m_max, k_spring_max, dt_max, epsilon_max ,c_i_max,f_i_max)=nebsettings['max_settings']

factor = 1.0/1.60217657e-12 # convert erg to eV

vectorcoords = g.load_vectorcoordinates_from_txt(vcoords_file)
centerscoords = g.generate_centercoords(vectorcoords)

distances = g.generate_distances(centerscoords, R)
distance_unit_vectors = g.generate_distance_vectors(centerscoords)


find_mep = partial(e.calculate_gradient,
        M,
        V,
        anisotropy_angles,
        K,
        H,
        field_angle,
        distances,
        distance_unit_vectors,
        factor)
