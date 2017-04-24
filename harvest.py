""" This is the spin ice farm itself """
# -*- coding: utf-8 -*-
__author__ = 'Sergei Liashko'

import os
#import sys
#abs_path_sif = os.path.dirname(os.path.realpath(__file__)) #get the sif current directory
#sys.path.append(f"{abs_path_sif}/external_libs") #add the exteranl lib to our sys.path so we could export it as a module

from   sif.external_libs import energy_module as em
from   sif.external_libs import pathminimizer_module as pm
from   sif.pathgenerator import pathgenerator as pg
from   sif.paramsfactory import geometry as g
from   sif.paramsfactory import paramsfactory as paramsfactory
from   sif.transition_rate import htstratecalculator as hr

import numpy as np
from scipy.optimize import minimize
from scipy.signal import argrelextrema
import numpy as np
from numpy import pi
from numpy import sqrt
import matplotlib.pyplot as plt
from functools import partial # we don't want to pass all parameters of the system
from sys import argv


def getMinimaAndSaddlesPositions(energy_path):
    _minimums = argrelextrema(energy_path, np.less_equal)[0]
    _saddles = argrelextrema(energy_path, np.greater)[0]
    return (_minimums,_saddles)

def getConstants():
    return {'gyromagnetic_ratio':1.76e7, # divide by M and V to get G
            'boltzmann':8.617e-5, #eV
            'erg2eV':624150932378.196}

def calculateBarriersMatrix(minimaEnergies,saddlesEnergies):
    n = len(minimaEnergies)
    _res = np.zeros((n,n))
    for i in range(n-1):
        init = minimaEnergies[i]
        final = minimaEnergies[i+1]
        barrier = saddlesEnergies[i]
        _res[i,i+1] = barrier - init
        _res[i+1,i] = barrier - final
    return _res

def calculateMEPandEnergiesAlongIt(path,params,distances, distance_unit_vectors,ci=False):
        factor = 1.0/1.60217657e-12 # convert erg to eV
        n = params['n']
        M = params['M']
        V = params['V']
        anisotropy_angles = params['anisotropy_angles']
        K = params['K']
        field_angle = params['field_angle']
        H = params['H']
        _tmp_path = np.copy(path)
        _tmp_en  = em.calculateEnergyAndGradient_func(_tmp_path,M,anisotropy_angles,K,0.0,0.0,V,distances,distance_unit_vectors,factor)[0]
        pm.find_mep_func(
                distances,
                distance_unit_vectors,
                M,
                V,
                K,
                anisotropy_angles,
                factor,
                _tmp_path,
                _tmp_en,
                0.0,
                0.0,
                0.1,
                1e-10,
                1.0,
                200001,
                ci,
                False)
        _tmp_en = em.calculateEnergyAndGradient_func(_tmp_path,M,anisotropy_angles,K,0.0,0.0,V,distances,distance_unit_vectors,factor)[0]
        return (np.copy(_tmp_path),np.copy(_tmp_en))


def main(params_file,vcoords_file):
    def calculateEnergy(system_coords):
            return em.calculateEnergyAndGradient_func((np.array([system_coords])).T,M,anisotropy_angles,K,0.0,0.0,V,distances,distance_unit_vectors,factor)[0]
    def calculateJac(system_coords):
            return em.calculateEnergyAndGradient_func((np.array([system_coords])).T,M,anisotropy_angles,K,0.0,0.0,V,distances,distance_unit_vectors,factor)[1]

    run_path = os.getcwd()
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

    #nebsettings = paramsfactory.load_runtime_settings(runtime_settings)
    #(m,k_spring, dt, epsilon, c_i,f_i) = nebsettings['general_settings']
    #(m_min, k_spring_min, dt_min, epsilon_min ,c_i_min,f_i_min)=nebsettings['min_settings']
    #(m_max, k_spring_max, dt_max, epsilon_max ,c_i_max,f_i_max)=nebsettings['max_settings']
    #
    factor = 1.0/1.60217657e-12 # convert erg to eV

    vectorcoords = g.load_vectorcoordinates_from_txt(vcoords_file)
    centerscoords = g.generate_centercoords(vectorcoords)

    distances = g.generate_distances(centerscoords, R)
    distance_unit_vectors = g.generate_distance_vectors(centerscoords)


    # other possible transition
    #aa = np.array([pi,pi,pi/2,pi/2,0.,0.,3*pi/2,3*pi/2,pi/2])
    #bb = (aa+pi)%(2*pi)

    #rot_or = [-1,1,1,1,1,-1,-1,-1,-1]# that one is for init
    #rot_or=np.array([-1,-1,-1,-1,-1,-1,-1,-1,-1]) # this one is for othe state
    rot_or = np.array([1,1,1,1,-1,1])
    path = pg.generate_one_by_one_rotation_path(aa,bb, 12,rot_or)
    #path = pg.generate_coherent_rotation_path(aa,bb,100)

    #==================== Find intial MEP - START ====================
    overallMepPath,overallMepEnergies = calculateMEPandEnergiesAlongIt(path,params,distances,distance_unit_vectors)
    min_pos, sad_pos = getMinimaAndSaddlesPositions(overallMepEnergies)
    minimaCoords = ((overallMepPath[:,min_pos]))
    saddlesCoords = ((overallMepPath[:,sad_pos]))
    #==================== Find intial MEP - DONE =====================

    #================== Minimization prcedures - START ===============
    coordsBounds = tuple((0,2.03*np.pi) for x in aa)
    minimizedEnergies = np.empty(1)[0:-1]
    minimizedCoords = np.empty((n,1))[:,0:-1]
    for coord in minimaCoords.T:
        res = minimize(calculateEnergy, coord, method='SLSQP',bounds=coordsBounds, options={ 'disp': False},jac=calculateJac)
        minimizedEnergies = np.append(minimizedEnergies,res.fun)
        minimizedCoords = np.append(minimizedCoords, np.array([res.x]).T, axis=1)
    #np.savetxt(f"{path}/calculated_data/pp",minimizedCoords)
    #np.savetxt("ep",minimizedEnergies)
    #================== Minimization prcedures - DONE ===============

    #================== Saddle points calcs - START ===============
    rot_or_in = np.array([-1,1,1,1,1,1])
    rot_or_out = np.array([1,-1,-1,-1,-1,-1])
    energies_along_path = np.empty(1)
    path = np.empty((n,1))
    for i in range(len(minimizedEnergies)-1):
        start = minimizedCoords[:,i]
        final = minimizedCoords[:,i+1]
        interimPath = pg.generate_coherent_rotation_path(start,final,13,rot_or_in)
        tmpp, tmpe = calculateMEPandEnergiesAlongIt(interimPath,params,distances,distance_unit_vectors,ci=True)
        energies_along_path = np.append(energies_along_path[0:-1],tmpe)
        path = np.append(path[:,0:-1], tmpp, axis=1)
    #================== New NEB calcs - DONE  ===============
    min_pos, sad_pos = getMinimaAndSaddlesPositions(energies_along_path)
    minimaCoords = ((path[:,min_pos]))
    saddlesCoords = ((path[:,sad_pos]))

    #print(path.shape)
    #print(energies_along_path.shape)
    np.savetxt(f'{run_path}/calculated_data/path.txt',path)
    np.savetxt(f'{run_path}/calculated_data/energy.txt',energies_along_path)

    path_coefficients_matrix_1 = np.loadtxt(f"{run_path}/init_data/pathstats1.txt")
    path_coefficients_matrix_2 = np.loadtxt(f"{run_path}/init_data/pathstats2.txt")
    constants = getConstants()
    barriers_matrix = calculateBarriersMatrix(energies_along_path[min_pos],energies_along_path[sad_pos])
    print(energies_along_path[min_pos])
    print(energies_along_path[sad_pos])
    print(barriers_matrix)

    #np.savetxt('path.txt',path)
    #np.savetxt('energy.txt',en)


    start_T = 62
    end_T   = 420
    step_T  = 50
    hessianParams = params

    hr.main(start_T, end_T,step_T,
            path_coefficients_matrix_1,
            path_coefficients_matrix_2,
            hessianParams, distances, distance_unit_vectors,
            constants,
            minimaCoords, saddlesCoords,
            barriers_matrix)

    print("Done")

if __name__ == "__main__":
    try:
        print(argv[0],argv[1],argv[2])
        main(argv[1], argv[2])
        #script, params_file, runtime_settings, vcoords_file = argv
    except:
        print("ERROR: You didn't provide the params file for the programm", argv[0])
