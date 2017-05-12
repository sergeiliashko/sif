import matplotlib.pyplot as plt
from scipy import stats
import sympy
import numpy as np
from sympy.parsing.sympy_parser import (parse_expr, implicit_multiplication_application,standard_transformations)

#(4.98e+09)*exp(-0.792/8.617e-5/420) n number of islands
def constructSymbolicEnergyAndHessian(n):
    def uniti_anisotropy_member(i):
        return f"K_{i}*V_{i}*(sin(th_{i})cos(phi_{i} - phi_K_{i}))**2"
    def uniti_easy_plane_member(i):
        return f"K_p*V_{i}*cos(th_{i})**2"
    def uniti_zeeman_member(i):
        return f"M_{i}*V_{i}*H_{i}*sin(th_{i})cos(phi_{i}-phi_H_{i})"
    def uniti_dipole_memeber(i,k):
        _first = f"sin(th_{i})*sin(th_{k})*cos(phi_{i} - phi_{k}) + cos(th_{i})*cos(th_{k})"
        _second = f"3sin(th_{i})cos(phi_{i}-phi_r_{i}{k})sin(th_{k})cos(phi_{k}-phi_r_{i}{k})"
        #return f"({_second}-{_first})4*pi*M_{i}*M_{k}*V_{i}*V_{k}/r_{i}{k}**3"
        return f"({_second}-{_first})*M_{i}*M_{k}*V_{i}*V_{k}/r_{i}{k}**3"


    def desiredParser(string):
        return parse_expr(string, transformations=(standard_transformations + (implicit_multiplication_application,)))

    _res_anis = ""
    _res_zeeman = ""
    _res_dipole = ""
    _res_easyplane = ""
    for h in range(n):
        _res_anis += "-" + uniti_anisotropy_member(h)
        _res_easyplane += "+" + uniti_easy_plane_member(h)
        _res_zeeman += "-" + uniti_zeeman_member(h)
        for k in range(n):
            if(h<k): _res_dipole += "-" + uniti_dipole_memeber(h,k)

    _symbolic_energy = desiredParser(_res_anis+_res_easyplane+_res_dipole+_res_zeeman)
    _symbolic_hessian = sympy.hessian(_symbolic_energy,[parse_expr(f"phi_{i}") for i in range(n)] + [parse_expr(f"th_{i}") for i in range(n)])
    return (_symbolic_energy, _symbolic_hessian )

def constructNumericEnergyAndHessian(hessianParams,distances,distance_unit_vectors):
    n = hessianParams["n"]
    #construct hessian
    _symbolicEnergy, _symbolicHessian = constructSymbolicEnergyAndHessian(n)

    #theta_r_ij is always pi/2
    _th_r_ij_subs = [(parse_expr(f"th_r_{i}{j}"),sympy.pi/2) for i in range(n) for j in range(n) if i<j]
    _res_hessian = _symbolicHessian.subs(_th_r_ij_subs)
    _res_energy = _symbolicEnergy.subs(_th_r_ij_subs)

    # K_out
    _res_hessian = _res_hessian.subs([(parse_expr("K_p"),hessianParams["K_out"][0])])
    _res_energy = _res_energy.subs([(parse_expr("K_p"),hessianParams["K_out"][0])])

    #H H_angle
    _field_angles = hessianParams["field_angle"]
    _field_values = hessianParams["H"]

    _phi_H_i_subs = [(parse_expr(f"phi_H_{i}"),_field_angles[i]) for i in range(n)]
    _H_i_subs = [(parse_expr(f"H_{i}"),_field_values[i]) for i in range(n)]

    _res_hessian=_res_hessian.subs(_phi_H_i_subs)
    _res_energy =_res_energy.subs(_phi_H_i_subs)

    _res_hessian=_res_hessian.subs(_H_i_subs)
    _res_energy =_res_energy.subs(_H_i_subs)

    #r_ij has a matrix distances
    _r_ij_subs = [(parse_expr(f"r_{i}{j}"),distances[i,j]) for i in range(n) for j in range(n) if i<j]
    _res_hessian=_res_hessian.subs(_r_ij_subs)
    _res_energy = _res_energy.subs(_r_ij_subs)

    #phi_r_ij has a matrix of unit vectors
    _phi_r_ij_subs = [(parse_expr(f"phi_r_{i}{j}"),distance_unit_vectors[i,j]) for i in range(n) for j in range(n) if i<j]
    _res_hessian=_res_hessian.subs(_phi_r_ij_subs)
    _res_energy = _res_energy.subs(_phi_r_ij_subs)

    #theta_i is always pi/2
    _th_i_subs = [(parse_expr(f"th_{i}"),sympy.pi/2) for i in range(n)]
    _res_hessian=_res_hessian.subs(_th_i_subs)
    _res_energy = _res_energy.subs(_th_i_subs)

    #phi_K_i has values
    _anisotropy_angles = hessianParams["anisotropy_angles"]
    _phi_K_i_subs = [(parse_expr(f"phi_K_{i}"),_anisotropy_angles[i]) for i in range(n)]
    _res_hessian=_res_hessian.subs(_phi_K_i_subs)
    _res_energy = _res_energy.subs(_phi_K_i_subs)

    #K_i
    _K_values = hessianParams["K"]
    _K_i_subs = [(parse_expr(f"K_{i}"),_K_values[i]) for i in range(n)]
    _res_hessian=_res_hessian.subs(_K_i_subs)
    _res_energy = _res_energy.subs(_K_i_subs)

    #V_i
    _V_values = hessianParams["V"]
    _V_i_subs = [(parse_expr(f"V_{i}"),_V_values[i]) for i in range(n)]
    _res_hessian=_res_hessian.subs(_V_i_subs)
    _res_energy = _res_energy.subs(_V_i_subs)
    #M_i
    _M_values = hessianParams["M"]
    _M_i_subs = [(parse_expr(f"M_{i}"),_M_values[i]) for i in range(n)]
    _res_hessian=_res_hessian.subs(_M_i_subs)
    _res_energy = _res_energy.subs(_M_i_subs)


    _phi_i_vars = [parse_expr(f"phi_{i}") for i in range(n)]
    _res_hessian = sympy.lambdify([_phi_i_vars],_res_hessian,"numpy")
    _res_energy = sympy.lambdify([_phi_i_vars],_res_energy, "numpy")
    return (_res_energy, _res_hessian)

def calculatePreFactor(numericHessian, minimumCoord, saddleCoord, y):
    n = minimumCoord.shape[0]
    hessian_min = numericHessian(minimumCoord)
    hessian_sad = numericHessian(saddleCoord)
    eigen_vals_min = np.linalg.eig(hessian_min)[0]
    eigen_vals_sad, eigen_vecs_sad = np.linalg.eig(hessian_sad)
    negative_val_pos_min = np.where(eigen_vals_min<0)[0]
    negative_val_pos_sad = np.where(eigen_vals_sad<0)[0]

    if(len(negative_val_pos_min)>0):
        raise ValueError('ERROR.Got negative eigen value at minima')
    if(len(negative_val_pos_sad)>1):
        raise ValueError('ERROR.Got more than 1 negative eigen value at saddle point')
    negative_val_pos_sad = negative_val_pos_sad[0]

    velocities = np.zeros_like(hessian_min)
    velocities[0:n,0:n*2] = y*hessian_sad[n:2*n:,0:2*n]
    velocities[n:2*n,0:n*2] = -y*hessian_sad[0:n:,0:2*n]

    tmp = np.dot(np.linalg.inv(eigen_vecs_sad),velocities)
    tmp = np.dot(tmp, eigen_vecs_sad)
    a_coef = np.copy(tmp[negative_val_pos_sad,:])
    a_coef[negative_val_pos_sad] = 0.

    adjusted_eigen_vals_sad = np.copy(eigen_vals_sad)
    adjusted_eigen_vals_sad[negative_val_pos_sad] = 1.

    det_min = np.sqrt(eigen_vals_min.prod())
    det_sad = np.sqrt(adjusted_eigen_vals_sad.prod())
    pr = np.sqrt((a_coef**2/adjusted_eigen_vals_sad).sum())

    pre_exp = (1./2./np.pi)*pr*(det_min/det_sad)

    return pre_exp

def calculateAllPreFactors(numericHessian, minima, saddles,y):
    n = minima.shape[1]
    pre_factors_matrix = np.zeros((n,n))
    for i in range(n-1):
        init = minima[:,i]
        final = minima[:,i+1]
        barrier = saddles[:,i]
        pre_factors_matrix[i,i+1] = calculatePreFactor(numericHessian,init,barrier,y)
        pre_factors_matrix[i+1,i] = calculatePreFactor(numericHessian,final,barrier,y)
    return pre_factors_matrix

def calculateTransitionRateForT(pre_factor,barrier,T,kb):
        return pre_factor*np.exp(-barrier/(T*kb))

def calculateAllTransitionRatesForT(pre_factors_matrix, barriers_matrix, T, kb):
    transition_rates_matrix = pre_factors_matrix*np.exp(-barriers_matrix/(T*kb))
    return transition_rates_matrix


# n-1 = number of barriers 
def calculateOverallRateForT(transition_rates_matrix_for_T, path_coefficients_matrix_1,
        path_coefficients_matrix_2):
    def b(i):
        if(i==0):
            return k1[0,1]
        else:
            return  k2[i,i-1]+k1[i,i+1]
    def a(i): return -k2[i-1,i]*k1[i,i-1]

    def A(i):
        if(i==0):
            return b(0)
        elif(i==1):
            return b(1)*b(0) + a(1)
        else:
            return b(i)*A(i-1)+a(i)*A(i-2)

    def B(i):
        if(i==0):
            return 1.0
        elif(i==1):
            return b(1)
        else:
            return b(i)*B(i-1)+a(i)*B(i-2)
    n = transition_rates_matrix_for_T.shape[0]
    k1 = transition_rates_matrix_for_T*path_coefficients_matrix_1
    k2 = transition_rates_matrix_for_T*path_coefficients_matrix_2
    #print(k1)
    #print('===')
    #print(k2)
    #print('=====')
    #print(transition_rates_matrix_for_T)
    #print('=====')
    return A(n-2)/B(n-2)

def calculateEffectiveBarrierAndPreFactor(temperature_range,
        path_coefficients_matrix_1,path_coefficients_matrix_2,
        hessianParams,pre_factors_matrix,barriers_matrix,constants):
    overall_rates=[]
    kb = constants["boltzmann"]
    for t in temperature_range:
        transition_rates_matrix_for_T = calculateAllTransitionRatesForT(pre_factors_matrix, barriers_matrix, t, kb)
        _overall_r = calculateOverallRateForT(transition_rates_matrix_for_T,path_coefficients_matrix_1,path_coefficients_matrix_2)
        overall_rates.append(_overall_r)
    temp1 = np.log(np.array(overall_rates))
    slope, intercept, r_value, p_value, std_err = stats.linregress(1./temperature_range,temp1)
    eff_pre_factor = np.exp(intercept)
    eff_barrier = -slope*kb
    #plt.plot(1./temperature_range,temp1 , '-b')
    #plt.show()
    return (eff_barrier,eff_pre_factor,overall_rates)


def main(start_T, end_T,step_T,
        path_coefficients_matrix_1,
        path_coefficients_matrix_2,
        hessianParams, distances, distance_unit_vectors,
        constants,
        minimaCoords, saddlesCoords,
        barriers_matrix):
    print("Start prefactor")

    numericEnergy, numericHessian = constructNumericEnergyAndHessian(
            hessianParams,
            distances,
            distance_unit_vectors)
    y = constants["gyromagnetic_ratio"]/(hessianParams["V"]*hessianParams["M"])
    y = np.swapaxes([y],1,0) # I need a vector with the shape (n,1)
    pre_factors_matrix = calculateAllPreFactors(numericHessian, minimaCoords, saddlesCoords,y)
    _temperature_range = np.linspace(start_T,end_T,step_T)

    np.set_printoptions(precision=3)
    print("=======")
    print("Pre factors matrix:",pre_factors_matrix)
    print("=======")
    eff_barrier_stat, eff_pre_factor_stat,o_r_s = calculateEffectiveBarrierAndPreFactor(
            _temperature_range,
            path_coefficients_matrix_1,
            path_coefficients_matrix_2,
            hessianParams,
            pre_factors_matrix,
            barriers_matrix,
            constants)
    print(f"overall rates across T[{start_T},{end_T}] ")
    print(np.array(o_r_s))
    print(f"eff_barrier with stat is {'%.3f' % eff_barrier_stat}")
    print(f"eff_pre_factor with stat is {'%.2e' % eff_pre_factor_stat}")
    print("=======")
    #eff_barrier_no_stat, eff_pre_factor_no_stat, o_r_ns = calculateEffectiveBarrierAndPreFactor(
    #        _temperature_range,
    #        np.ones_like(pre_factors_matrix),
    #        np.ones_like(pre_factors_matrix),
    #        hessianParams,
    #        pre_factors_matrix,
    #        barriers_matrix,
    #        constants)

    #print(f"overall rates across T[{start_T},{end_T}] ")
    #print(np.array(o_r_ns))
    #print(f"eff_barrier without stat is {'%.3f' % eff_barrier_no_stat}")
    #print(f"eff_pre_factor without stat is {'%.2e' % eff_pre_factor_no_stat}")

