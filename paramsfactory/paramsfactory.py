# -*- coding: utf-8 -*-
__author__ = 'Sergei Liashko'

import numpy as np
from   sif.paramsfactory import geometry as g

#
def reduce_metric_prefix(value,unit):
    #print('init unit',unit)
    _metric_prefixes = {'d':1e-1, 'c':1e-2, 'm':1e-3,'μ':1e-6,'n':1e-9}
    _metric_prefixes_exceptions = ['deg']
    if(len(unit)<2):
        return (value,unit) #nothing to do here, return initial values
    else:
        _tmp = unit.partition('^')
        _multiplier = 1. if( not _tmp[2].isnumeric() or (unit.count('/')>0)) else float(_tmp[2])
        _answer_v = value
        _answer_u = unit
        #print('init answer', _answer_u)
        for prefix,factor in _metric_prefixes.items():
            if(unit.startswith(prefix)):
                if(_answer_v != value):
                    raise ValueError(f"Prefix matrix has abigious elements for the unit {unit}. Error raised at prefix {prefix}. Reduced units are {_answer_u}")
                else:
                    _answer_v *=factor**_multiplier
                    _answer_u = unit[len(prefix):]
        #print('final answer',_answer_u)
        return (_answer_v,_answer_u)

#choose convert what to what
def convert_to_internal_units(value_unit_zipped):
    def _cnv2def(value,unit):
        _belongs2default = _default.count(unit) > 0
        _belongs2si = list(_ci2cgs.keys()).count(unit) > 0
        #print(unit,_belongs2default, _belongs2si)
        if(_belongs2default or _belongs2si):
            if(_belongs2default):
                return (value,unit)
            else:
                return (_ci2cgs[unit][0]*value,_ci2cgs[unit][1])
        else:
            raise ValueError(f"Input unit {unit} is not recongised and cannot be used")

    _cgs = ['s','g','cm','cm^3','erg','G','erg/G/cm^3','erg/cm^3']
    _g2default = []
    _ci2cgs = {'m':(1e2,'cm'),'kg':(1e3,'g'),'J':(1e7,'erg'),'T':(1e4,'G'),'A/m':(1e-3,'erg/G/cm^3'),'J/m^3':(1e-2,'erg/cm^3'),'m^3':(1e6,'cm^3')}
    _default = _cgs
    _reduced_prefixes = [reduce_metric_prefix(value,unit) for value,unit in value_unit_zipped]
    #print(_reduced_prefixes)
    _result = [_cnv2def(value,unit)[0] for value,unit in _reduced_prefixes]
    return _result


def degree2radian(deg,unit):
    if unit =='deg':
        return deg*np.pi/180.
    elif unit == 'rad':
        return deg
    else:
        raise ValueError(f'Angle unit {unit} is not recognisible')


def volume_calculation(dim,shape):
    _h, _w, _t = np.array(dim, dtype=float)
    if(shape=='stadium'):
        return (np.pi*(_w/2.)**2 + (_h-_w/2.)*(_w/2.))*_t
    elif(shape=='rectangle'):
        return _h*_w*_t
    else:
        raise ValueError(f"There is no such type as {shape}")


def new_load_hamiltonian_params(filename):
    _lattice_params = np.genfromtxt(filename,skip_footer=1,delimiter =',',dtype=[('vectorcoords','U500'),('lattice_constant','float'),('lattice_constant_units','U3')])
    _vectorcoords = g.load_vectorcoordinates_from_txt(str(_lattice_params['vectorcoords']))
    _centerscoords = g.generate_centercoords(_vectorcoords)
    distances = g.generate_distances(_centerscoords, convert_to_internal_units([(_lattice_params['lattice_constant'],str(_lattice_params['lattice_constant_units']))])[0])
    distance_unit_vectors = g.generate_distance_vectors(_centerscoords)

    _island_params = np.genfromtxt(filename,skip_header=2,delimiter=',',autostrip=True, dtype=[ ('M','float'),
        ('M_units','U25'),
        ('dim','U35'),
        ('V_units','U25'),
        ('shape','U25'),
        ('K_angle','float'),
        ('K_angle_units','U3'),
        ('K','float'),
        ('K_units','U25'),
        ('K_out','float'),
        ('K_out_units','U25'),
        ('H_angle','float'),
        ('H_angle_units','U3'),
        ('H','float'),
        ('H_units','U25')],
        converters = {
            1:lambda s:  s.decode("utf-8"),
            2:lambda s:  s.decode("utf-8"),
            3:lambda s:  s.decode("utf-8"),
            4:lambda s:  s.decode("utf-8"),
            6:lambda s:  s.decode("utf-8"),
            8:lambda s:  s.decode("utf-8"),
            10:lambda s: s.decode("utf-8"),
            12:lambda s: s.decode("utf-8"),
            14:lambda s: s.decode("utf-8")})
    # do volumes
    _volumes = [volume_calculation(np.array(dim.split('x'),dtype='float'),shape) for dim,shape in zip(_island_params['dim'],_island_params['shape'])]
    V = convert_to_internal_units(zip(_volumes, _island_params['V_units']))

    # convert other guys
    M = convert_to_internal_units(zip(_island_params['M'],_island_params['M_units']))
    K = convert_to_internal_units(zip(_island_params['K'],_island_params['K_units']))
    K_out = convert_to_internal_units(zip(_island_params['K_out'],_island_params['K_out_units']))
    H = convert_to_internal_units(zip(_island_params['H'],_island_params['H_units']))

    # anlges is the different beast
    K_angle = [degree2radian(value,unit) for value, unit in zip(_island_params['K_angle'],_island_params['K_angle_units'])]
    H_angle = [degree2radian(value,unit) for value, unit in zip(_island_params['H_angle'],_island_params['H_angle_units'])]
    return {'n':len(M),
            'M':np.array(M),
            'V':np.array(V),
            'anisotropy_angles':np.array(K_angle),
            'K':np.array(K),
            'field_angle':np.array(H_angle),
            'H':np.array(H),
            'K_out':np.array(K_out),
            'distance_unit_vectors':np.array(distance_unit_vectors),
            'distances':np.array(distances)}




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

def load_init_path(filename):
    params = np.genfromtxt(filename)
    n = len(params[0])
    return {'rot_or' :params[0],
            'state_0':params[1],
            'state_m':params[2] }

def load_runtime_settings(filename):
    settings = np.genfromtxt(filename, dtype=[int,float,float,float,bool, bool])
    return {'general_settings':settings[0],
            'min_settings':settings[1],
            'max_settings':settings[2]}
