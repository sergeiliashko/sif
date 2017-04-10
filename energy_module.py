"""Let's try to wrap a C++ energy to python
    Why would we need that you ask? Well, you
    always need to make a check what's going on
    at particular step. The acces to energy functions
    make that a trivial task"""

import numpy as np
import numpy.ctypeslib as npct
from ctypes import c_int
from ctypes import c_double

# input type for the
# must be a double array, with single dimension that is contiguous
array_1d_double = npct.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')

# load the library, using numpy mechanisms
libcd = npct.load_library("energy", ".")


libcd.shape_anisotropy_energy.restype = None
libcd.shape_anisotropy_energy.argtypes = [
        array_1d_double,
        array_1d_double,
        array_1d_double,
        array_1d_double,
        c_double,
        c_int,
        array_1d_double]

def shape_anisotropy_energy_func(
        system_angles,
        anisotropy_angles,
        anisotropy_values,
        islands_volumes,
        factor
        #int size will be get from array properties
        #out_array
        ):
        tmp_out_array = np.zeros_like(system_angles)
        libcd.shape_anisotropy_energy(system_angles, anisotropy_angles, anisotropy_values, islands_volumes, factor,len(system_angles), tmp_out_array)
        return tmp_out_array


libcd.zeeman_energy.restype = None
libcd.zeeman_energy.argtypes = [
        array_1d_double,
        array_1d_double,
        array_1d_double,
        array_1d_double,
        array_1d_double,
        c_double,
        c_int,
        array_1d_double]

def zeeman_energy_func(
        system_angles,
        ext_field_angles,
        ext_field_values,
        islands_volumes,
        magnetisation_values,
        factor
        # size is omited and will be get from array length
        # out_array
        ):
        tmp_out_array = np.zeros_like(system_angles)
        libcd.zeeman_energy(
                system_angles,
                ext_field_angles,
                ext_field_values,
                islands_volumes,
                magnetisation_values,
                factor,
                len(system_angles),
                tmp_out_array)
        return tmp_out_array

libcd.dipole_dipole_energy.restype = None
libcd.dipole_dipole_energy.argtypes = [
        array_1d_double,
        array_1d_double,
        array_1d_double,
        array_1d_double,
        array_1d_double,
        c_double,
        c_int,
        array_1d_double]

def dipole_dipole_energy_func(
        system_angles,
        magnetisation_values,
        islands_volumes,
        distances,
        distance_unit_vectors,
        factor
        # size is ommited will be gotten from arra size
        # out_array
        ):
    tmp_out_array = np.zeros_like(system_angles)
    libcd.dipole_dipole_energy(
            system_angles,
            magnetisation_values,
            islands_volumes,
            distances.flatten(),
            distance_unit_vectors.flatten(),
            factor,
            len(system_angles),
            tmp_out_array)
    return tmp_out_array

libcd.calculateEnergyAndGradient.restype = None
libcd.calculateEnergyAndGradient.argtypes = [
        array_1d_double, # path
        array_1d_double, # M
        array_1d_double, # Kangles
        array_1d_double, # K
        c_double,        # H
        c_double,        # H_ang
        array_1d_double, # V
        array_1d_double, # dist
        array_1d_double, # dist_unit
        c_double,        # factor
        c_int,           # m
        c_int,           # n
        array_1d_double, # energ_out
        array_1d_double] # gradi_out

def calculateEnergyAndGradient_func(
        set_of_images,
        magnetisation_values,
        anisotropy_angles,
        anisotropy_values,
        ext_field_value,
        ext_field_angle,
        islands_volumes,
        distances,
        distance_unit_vectors,
        factor
        # m,
        # n,
        # output_energy,
        # output_gradietn
        ):
    n = set_of_images.shape[0]
    m = set_of_images.shape[1]
    tmp_out_energy = np.zeros(m,dtype=np.float)
    tmp_out_gradient = np.zeros(n*m,dtype=np.float)
    libcd.calculateEnergyAndGradient(
            set_of_images.flatten(),
            magnetisation_values,
            anisotropy_angles,
            anisotropy_values,
            ext_field_value,
            ext_field_angle,
            islands_volumes,
            distances.flatten(),
            distance_unit_vectors.flatten(),
            factor,
            m,
            n,
            tmp_out_energy,
            tmp_out_gradient)
    return [tmp_out_energy, tmp_out_gradient.reshape((n,m))]

libcd.calculateTangetAndSpringForces.restype = None
libcd.calculateTangetAndSpringForces.argtypes = [
        array_1d_double,
        array_1d_double,
        c_int,
        c_int,
        c_double,
        array_1d_double,
        array_1d_double]

def calculateTangetAndSpringForces_func(
        set_of_images,
        energy_path,
        # n,
        # m,
        springConstant
        #output_tangent,
        #output_springForces
        ):
    n = set_of_images.shape[0]
    m = set_of_images.shape[1]
    tmp_out_tangent = np.zeros(n*m, dtype=np.float)
    tmp_out_springForces = np.zeros(n*m, dtype=np.float)
    libcd.calculateTangetAndSpringForces(
            set_of_images.flatten(),
            energy_path,
            n,# n,
            m,# m,
            springConstant,
            tmp_out_tangent,
            tmp_out_springForces)
    return [tmp_out_tangent.reshape((n,m)), tmp_out_springForces.reshape((n,m))]

libcd.calculatePerpendicularForces.restype = None
libcd.calculatePerpendicularForces.argtypes = [
        array_1d_double,
        array_1d_double,
        c_int,
        c_int,
        array_1d_double]

def calculatePerpendicularForces_func(
        gradient_path,
        norm_tangets,
        #n,
        #m,
        #output_perpendicularForces
        ):
    n = gradient_path.shape[0] #n,
    m = gradient_path.shape[1] #m,
    tmp_out_perpForce = np.zeros(n*m,dtype=np.float)
    libcd.calculatePerpendicularForces(
            gradient_path.flatten(),
            norm_tangets.flatten(),
            n,
            m,
            tmp_out_perpForce)
    return tmp_out_perpForce.reshape((n,m))

libcd.calculateTrueForces.restype = None
libcd.calculateTrueForces.argtypes = [
        array_1d_double,
        array_1d_double,
        c_int,
        c_int,
        array_1d_double]

def calculateTrueForces_func(
        springForces,
        perpendicularForces,
        #  n,
        #  m,
        # output_trueForces
        ):
    n = perpendicularForces.shape[0] #n,
    m = perpendicularForces.shape[1] #m,
    tmp_out_trueForces = np.zeros_like(perpendicularForces)
    libcd.calculateTrueForces(
            springForces.flatten(),
            perpendicularForces.flatten(),
            n,
            m,
            tmp_out_trueForces)
    return tmp_out_trueForces.reshape((n,m))

libcd.makeStep.restype = None
libcd.makeStep.argtypes = [
        array_1d_double,
        array_1d_double,
        c_int,
        c_int,
        c_double]

def makeStep_func(
        path,
        forces,
        # n,
        # m,
        dt,
        out,
        ):
    n = path.shape[0]
    m = path.shape[1]
    if(out.isEmpty):
        libcd.makestep(
                path.flatten(),
                forces.flatten(),
                n,
                m,
                dt)
    else:
        libcd.makestep(
                out.flatten(),
                forces.flatten(),
                n,
                m,
                dt)
