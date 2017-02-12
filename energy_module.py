"""Let's try to wrap a C++ energy to python
    Why would we need that you ask? Well, you
    always need to make a check what's going on
    at particular step. The acces to energy functions
    make that trivial task"""

import numpy as np
import numpy.ctypeslib as npct
from ctypes import c_int
from ctypes import c_double

# input type for the
# must be a double array, with single dimension that is contiguous
array_1d_double = npct.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')

# load the library, using numpy mechanisms
libcd = npct.load_library("pathminimizer", ".")


libcd.shape_anisotropy_energy.restype = None
libcd.shape_anisotropy_energy.argtypes = [
        array_1d_double,
        array_1d_double,
        array_1d_double,
        array_1d_double,
        c_double,
        c_int,
        array_1d_double]

def shape_anisotropy_energy_func(system_angles, anisotropy_angles, anisotropy_values, islands_volumes, factor, out_array):
        return libcd.shape_anisotropy_energy(system_angles, anisotropy_angles, anisotropy_values, islands_volumes, factor,len(system_angles), out_array)
