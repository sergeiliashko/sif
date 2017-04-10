# -*- coding: utf-8 -*-
__author__ = 'Sergei Liashko'

""" You need a start for everything """

import numpy as np
from numpy import pi

def generate_coherent_rotation_path(initState,finalState, numberOfSteps):
    dt = np.array([finalState-initState])/(numberOfSteps-1)
    path = np.array([initState])
    for i in range(numberOfSteps-1):
        path = np.vstack((path,(path[i,:]+dt)))
    return (path % (2*pi)).T

# total m = n*z-n+1=n(z-1)+1 so,  
# m = n*z-n+1 => z = (m + n -1)/n
# m = 9  n=2 z = 5
# m = 13 n=3 z = 5
# m = 17 n=4 z = 5
# m = 21 n=5 z = 5
# m =25  n=6 z = 5

def generate_one_by_one_rotation_path(initState, finalState, numberOfIntermidSteps):
    m = numberOfIntermidSteps
    dt = np.array([np.array(finalState)-np.array(initState)])/(m-1)
    path = np.array([initState])
    for i in range(len(initState)):
        change = np.zeros_like(dt)
        change[0,i] = dt[0,i]
        for j in range(m-1):
            #k = i*len(initState) + j
            k = i*(m-1)+j
            path = np.vstack((path,(path[k,:]+change)))
    return (path % (2*pi)).T

#def generate_between_states_path()
