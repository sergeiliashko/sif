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
libcd = npct.load_library("neb", ".")

# setup the return types and argument types
libcd.find_mep.restype = None
libcd.find_mep.argtypes = [
        array_1d_double, #distances
        array_1d_double, #distances_unit_vectors
        array_1d_double, #M array
        array_1d_double, #V array
        array_1d_double, #K array
        array_1d_double, #K angles array
        c_double,        #factor
        c_int,           #n dimension
        c_int,           #m dimension
        array_1d_double, #path
        array_1d_double, #energy path
        c_double,        #H
        c_double,        #H_angle
        c_double,        #dt
        c_double,        #epsilon
        c_double,        #k_spring
        c_int,           #maxiter
        c_bool,          #use_ci
        c_bool          #use_fi
        ]

def find_mep_func(
        distances,
        distances_unit_vectors,
        magnetisation_values,
        islands_volumes,
        anisotropy_values,
        anisotropy_angles,
        factor,
        # n goes inside the func
        # m  goes inside the func
        path,
        energy_path,
        H,
        H_angle,
        dt,
        epsilon,
        k_spring,
        maxiter,
        use_ci,
        use_fi
        ):
        return libcd.find_mep(
                distances.flatten(),
                distances_unit_vectors.flatten(),
                magnetisation_values,
                islands_volumes,
                anisotropy_values,
                anisotropy_angles,
                factor,
                path.shape[0], # n
                path.shape[1], # m 
                path.flatten(),
                energy_path,
                H,
                H_angle,
                dt,
                epsilon,
                k_spring,
                maxiter,
                use_ci,
                use_fi)
