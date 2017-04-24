# -*- coding: utf-8 -*-
__author__ = 'Sergei Liashko'

import numpy as np

# N M V A_an A_v F_an F_v
def load_hamiltonian_params(filename):
    params = np.genfromtxt(filename)
    n = len(params[0])
    return {'n':n,
            'M':params[0],
            'V':params[1],
            'anisotropy_angles':params[2],
            'K':params[3],
            'field_angle':params[4],
            'H':params[5],
            'R':(params[6])[0],
            'K_out':params[7],
            'state_0':params[8],
            'state_m':params[9]}


def load_runtime_settings(filename):
    settings = np.genfromtxt(filename, dtype=[int,float,float,float,bool, bool])
    return {'general_settings':settings[0],
            'min_settings':settings[1],
            'max_settings':settings[2]}
