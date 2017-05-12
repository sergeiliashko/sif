# -*- coding: utf-8 -*-
__author__ = 'Sergei Liashko'

""" You need a start for everything """

import numpy as np
from numpy import pi

def generate_coherent_rotation_path(initState,finalState, size,rotation_orientation):
    def isLargerThan2Quandrants(init,finish):
        if(abs(init-finish)<pi):
            return False
        else:
            return True
    if(size < 2):
        raise ValueError('desired size is less than 3. You cant make a path')
    #path=np.linspace(initState[0],finalState[0],size)
    #for i in range(1,len(initState)):
    #    if isLargerThan2Quandrants(initState[i],finalState[i]):
    #        if(
    #        path = np.vstack((path,np.linspace(finalState[i],rotation_orientation[i]*initState[i],size)))
    #    else:
    #        path = np.vstack((path,np.linspace(initState[i],rotation_orientation[i]*finalState[i],size)))



    interimNumberOfSteps = size - 2
    dt = (np.array(finalState)-np.array(initState))/(interimNumberOfSteps+1)
    for i in range(len(dt)):
        if isLargerThan2Quandrants(initState[i],finalState[i]):
            dt[i] *=-1.
        else:
            dt[i] *= 1.
    pos_max = np.argmax(np.abs(dt))
    dt[pos_max] *= rotation_orientation[pos_max]
    path = np.array([initState])
    for i in range(interimNumberOfSteps):
        path = np.vstack((path,(path[i,:]+dt)))
    path = np.vstack((path,finalState))
    return (path % (2*pi)).T
    #return (path % (2*pi))

# total m = n*z-n+1=n(z-1)+1 so,  
# m = n*z-n+1 => z = (m + n -1)/n
def generate_one_by_one_rotation_path(initState, finalState, numberOfIntermidSteps,rotation_orientation):
    m = numberOfIntermidSteps
    dt = (np.array(finalState)-np.array(initState))/(m-1)
    print(dt)
    print(rotation_orientation)
    dt = dt*np.array(rotation_orientation)
    path = np.array([initState])
    t_i = 0
    for i in range(len(initState)):
        change = np.zeros_like(dt)
        change[i] = dt[i]
        if(np.abs(change[i]) > 1e-3):
            for j in range(m-1):
                #k = i*len(initState) + j
                k = t_i*(m-1)+j
                path = np.vstack((path,(path[k,:]+change)))
            t_i+=1
    return (path % (2*pi)).T

#def generate_between_states_path()
