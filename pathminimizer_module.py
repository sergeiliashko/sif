"""Let's try to wrap a C++ minimization to python
   NEB method with all the glory along with it."""

import numpy as np
import numpy.ctypeslib as npct
from ctypes import c_int
from ctypes import c_double
from ctypes import c_bool

# input type for the cos_doubles function
# must be a double array, with single dimension that is contiguous
array_1d_double = npct.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')

# load the library, using numpy mechanisms
libcd = npct.load_library("pathminimizer", ".")

# setup the return types and argument types
libcd.find_mep.restype = None
libcd.find_mep.argtypes = [
        array_1d_double, #distances
        array_1d_double, #distances_unit_vectors
        array_1d_double, #M array
        array_1d_double, #V array
        array_1d_double, #K array
        c_double,        #H
        c_double,        #H_angle
        array_1d_double, #K angles array
        array_1d_double, #M angles array
        c_double,        #factor we use to change units for the energy
        c_int,           #n dimension
        c_int,             #m dimension
        array_1d_double, #path
        array_1d_double, #energy path
        c_double,        #dt
        c_double,        #epsilon
        c_double,        #k_spring
        c_int,           #maxiter
        c_int,           #printInfoAfter
        c_bool,          #use_ci
        c_int,           #useCIAfter
        c_bool          #printInfo
        ]
#
def find_mep(
        distances,
        distances_unit_vectors,
        m_values,
        v_values,
        k_values,
        h_value,
        h_angle,
        k_angles,
        m_angles,
        factor,
        #m and n goes inside the func
        path,
        energy_path,
        dt,
        epsilon,
        k_spring,
        maxiter,
        printInfoAfter,
        use_ci,
        useCIAfter,
        printInfo
        ):
        return libcd.cos_doubles(
                distances,
                distances_unit_vectors,
                m_values,
                v_values,
                k_values,
                h_value,
                h_angle,
                k_angles,
                m_angles,
                factor,
                m_path.shape[0],
                m_path.shpae[1],
                path,
                energy_path,
                dt,
                epsilon,
                k_spring,
                maxiter,
                printInfoAfter,
                use_ci,
                useCIAfter,
                printInfo)
