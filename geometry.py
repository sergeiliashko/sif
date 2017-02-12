# -*- coding: utf-8 -*-
import numpy as np
from scipy.spatial import distance
from numpy import sqrt
from numpy import pi

def load_vectorcoordinates_from_txt(inputfile):
    coords = np.loadtxt(inputfile)
    n,i = coords.shape
    return coords.reshape((n,i/2,2))

def generate_centercoords(vectorcoords):
    # assuming it has (N,m,2) dimensions
    return (vectorcoords[:,0] + vectorcoords[:,1])/2.

def generate_vector_weight(vectors):
    # assuming it (N,m)
    norms = np.apply_along_axis(np.linalg.norm,1,vectors)
    return norms/np.min(norms)

def generate_angle_with_Ox(vectorcoords):
    vectors = (vectorcoords[:,1] - vectorcoords[:,0])
    n,m = vectors.shape
    normed_vectors = vectors/np.apply_along_axis(np.linalg.norm,1, vectors).reshape((n,1))
    x_axis = np.repeat(0., m)
    x_axis[0] = 1.
    angles = np.array([], dtype=float)
    for unit_vec in normed_vectors:
        if unit_vec[1]<0 :
            angles = np.append(angles, -np.arccos(np.clip(np.dot(unit_vec,x_axis),-1.0,1.0)))
        else:
            angles = np.append(angles, +np.arccos(np.clip(np.dot(unit_vec,x_axis),-1.0,1.0)))

    return angles

def generate_distance_vectors(xycenterdots, equalhalfs=True):
    # equalhalfs, means the lower triangle equals to upper triangle
    result = np.array([])
    for xy1 in xycenterdots:
        for xy2 in xycenterdots:
            vector = xy2 - xy1
            if vector[0] < 0:
                result = np.append(result, np.array([np.arctan(np.divide(vector[1], vector[0]))])+np.pi)
            elif vector[1] < 0:
                result = np.append(result, np.array([np.arctan(np.divide(vector[1], vector[0]))])+2*np.pi)
            else:
                result = np.append(result, np.array([np.arctan(np.divide(vector[1], vector[0]))]))
    n = len(xycenterdots)
    result = result.reshape((n,n))
    if equalhalfs:
        result[np.tril_indices(n)]=(result[np.tril_indices(n)] + pi)%(2*pi)
    return np.nan_to_num(result)

def generate_distances(xycenterdots, lattice_constant):
    return distance.cdist(xycenterdots, xycenterdots, 'euclidean')*lattice_constant

def get_kagome_properties(lattice_constant):
    r3 = 3/4.; s3 = sqrt(3)/2. #temporary vars
    kagome_coords = np.array([[0,s3], [r3,s3/2], [r3,-s3/2], [0,-s3], [-r3,-s3/2], [-r3,s3/2]])
    dd = generate_distances(kagome_coords, 2*lattice_constant/sqrt(3.))
    uv = generate_distance_vectors(kagome_coords)
    return (dd, uv)
