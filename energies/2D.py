# -*- coding: utf-8 -*-
__author__ = 'Sergei Liashko'

import numpy as np
from numpy import cos
from numpy import sin

def shape_anisotropy_energy(
        system_angles: np.ndarray,
        anisotropy_angles:np.ndarray,
        anisotropy_parameters: np.ndarray,
        island_volumes: np.ndarray,
        factor
        ) -> float:
    """Return the shape anisotropy energy that correspond to this angles.[erg].

    Parameters
    ----------
    system_angles: np.ndarray
        a np array of (N,1) dimensionality with magnetisation's angles.[Rad].
    anisotropy_parameters:np.ndarray
        anisotropy parameters for all islands derrived from selfinteraction energy.[erg/cm^3].
    anisotropy_angles: np.ndarray
        a np array of (N,1) dimensionality with anisotropy angles.[Rad].
    island_volumes: np.ndarray
        corepsonding volumes of magnetic islands.[cm^3].
    """
    return (factor * island_volumes *  anisotropy_parameters * (cos(system_angles - anisotropy_angles) ** 2)).sum()


def zeeman_energy(
        system_angles: np.ndarray,
        ext_field_angle: np.ndarray,
        ext_field_value: np.ndarray,
        island_volumes: np.ndarray,
        magnetisation_values: np.ndarray,
        factor
        ) -> float:
    """Return Zeeman energy of the system that is corresponded to this angles.[erg].

    Parameters
    ----------
    system_angles: np.ndarray
        corresponding to magnetisation's angles.[Rad].
    magnetisation_values: np.ndarray
        corresponding to magnetisation values of islands.[emu/cm^3].
    island_volumes: np.ndarray
        corresponding volumes of a magnetic islands [cm^3].
    ext_field_value: float
        the value define the strength of the apllied external field.[G].
    ext_field_angle: float
        define the direction angle of the external field.[Rad].
    """
    return  ( ext_field_value *factor * island_volumes *  magnetisation_values * cos(system_angles - ext_field_angle)).sum()

def dipole_dipole_energy(
        system_angles: np.ndarray,
        magnetisation_values: np.ndarray,
        island_volumes: np.ndarray,
        distances: np.ndarray,
        distance_unit_vectors: np.ndarray,
        factor
        ) -> float:
    """Return Dipole Dipole energy of the system that is corresponded to this angles.[erg].

    Parameters
    ----------
    system_angles: np.ndarray
        corresponding to magnetisation's angles.[Rad].
    magnetisation_values: np.ndarray
        corresponding to magnetisation values of islands.[emu/cm^3].
    island_volumes: np.ndarray
        corresponding volumes of a magnetic islands.[cm^3].
    distances: np.ndarray
        matrix of values of distances between isladns.[cm].
    distance_unit_vectors: np.ndarray
        matrix of distance unit vectors between islands.[Rad].
    """
    indices = np.arange(len(system_angles))
    index_pairs = np.transpose([np.tile(indices, len(indices)), np.repeat(indices, len(indices))]) # all pairs
    zz =(index_pairs.T[0] - index_pairs.T[1])
    index_pairs = index_pairs[np.where( zz > 0)] # remove pairs i == j and repeated ones
    all_i = index_pairs.T[0]
    all_j = index_pairs.T[1]

    i_angles = system_angles[all_i]
    j_angles = system_angles[all_j]

    ij_distances = distances[all_i, all_j]
    ij_distance_unit_vectors = distance_unit_vectors[all_i, all_j]

    i_magnetisation_values = magnetisation_values[all_i]
    j_magnetisation_values = magnetisation_values[all_j]

    i_island_volumes = island_volumes[all_i]
    j_island_volumes = island_volumes[all_j]

    return ((3*cos(i_angles - ij_distance_unit_vectors)
        * cos(j_angles - ij_distance_unit_vectors)
        - cos(i_angles - j_angles))
        * (factor * 4*np.pi*((i_magnetisation_values
        * j_magnetisation_values
        * i_island_volumes
        * j_island_volumes)
        / ij_distances**3))).sum()

def calculate_energy(
        magnetisation_values: np.ndarray,
        island_volumes: np.ndarray,
        anisotropy_angles: np.ndarray,
        anisotropy_parameters: np.ndarray,
        ext_field_angle: np.ndarray,
        ext_field_value: np.ndarray,
        distances: np.ndarray,
        distance_unit_vectors: np.ndarray,
        factor: float,
        system_angles: np.ndarray
        ) -> float:
    return (- shape_anisotropy_energy(system_angles,
                                    anisotropy_angles,
                                    anisotropy_parameters,
                                    island_volumes,
                                    factor)
            - zeeman_energy(system_angles,
                            ext_field_angle,
                            ext_field_value,
                            island_volumes,
                            magnetisation_values,
                            factor)
            - dipole_dipole_energy(system_angles,
                                   magnetisation_values,
                                   island_volumes,
                                   distances,
                                   distance_unit_vectors,
                                   factor))

def g_d_e(
        system_angles: np.ndarray,
        magnetisation_values: np.ndarray,
        island_volumes: np.ndarray,
        distances: np.ndarray,
        distance_unit_vectors: np.ndarray,
        factor
        ) -> float:
    n = len(system_angles)
    indices = np.arange(n)
    index_pairs = np.transpose([np.tile(indices, len(indices)), np.repeat(indices, len(indices))]) # all pairs
    zz =(index_pairs.T[0] - index_pairs.T[1])
    index_pairs = index_pairs[np.where( zz != 0)] # remove pairs i == j and repeated ones
    z = np.apply_along_axis(lambda x: np.array(np.where(index_pairs.T[0] == x[0])[0]), # I get indices by number
            1, # I get indices along rows since I've reshaped them
            indices.reshape((n,1)) # just in order to get nice aligmnet of the array
            )
    c = index_pairs[z]
    def _helper_to_get_elem(index_pair):
        i = index_pair[0]
        j = index_pair[1]

        i_angle = system_angles[i]
        j_angle = system_angles[j]
        #print('i_a',i_angle)
        #print('j_a',j_angle)

        ij_distance = distances[i, j]
        ij_distance_unit_vector = distance_unit_vectors[i, j]
        #print('ij_distance',ij_distance)
        #print('ij_distance_unit_vector',ij_distance_unit_vector)

        i_magnetisation_value = magnetisation_values[i]
        j_magnetisation_value = magnetisation_values[j]
        #print('i_m',i_magnetisation_value)
        #print('j_m',j_magnetisation_value)

        i_island_volume = island_volumes[i]
        j_island_volume = island_volumes[j]

        return ((sin(i_angle - j_angle)
                - 3. * sin(i_angle - ij_distance_unit_vector)
                * cos(j_angle - ij_distance_unit_vector))
                * factor*4*np.pi*i_magnetisation_value
                * j_magnetisation_value
                * i_island_volume
                * j_island_volume
                / ij_distance**3)
    return np.apply_along_axis(_helper_to_get_elem,2,c).sum(1)

def g_a_e(system_angles,
        anisotropy_parameters,
        island_volumes,
        anisotropy_angles,
        factor
        ) -> np.ndarray:
    return anisotropy_parameters * factor * island_volumes * sin(2.0 * (system_angles - anisotropy_angles))

def g_z_e(system_angles,
        ext_field_value,
        ext_field_angle,
        island_volumes,
        magnetisation_values,
        factor
        ) -> np.ndarray:
    return factor*ext_field_value * magnetisation_values * island_volumes * sin(system_angles - ext_field_angle)

def calculate_gradient(
        magnetisation_values: np.ndarray,
        island_volumes: np.ndarray,
        anisotropy_angles: np.ndarray,
        anisotropy_parameters: np.ndarray,
        ext_field_angle: np.ndarray,
        ext_field_value: np.ndarray,
        distances: np.ndarray,
        distance_unit_vectors: np.ndarray,
        factor: float,
        system_angles: np.ndarray
        ) -> float:
    return (+ g_a_e(system_angles,
                    anisotropy_parameters,
                    island_volumes,
                    anisotropy_angles,
                    factor)
            + g_z_e(system_angles,
                    ext_field_angle,
                    ext_field_value,
                    island_volumes,
                    magnetisation_values,
                    factor)
            - g_d_e(system_angles,
                    magnetisation_values,
                    island_volumes,
                    distances,
                    distance_unit_vectors,
                    factor))
