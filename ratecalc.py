from scipy import stats
from numpy import pi as pi
from numpy import e as e
from scipy.signal import argrelextrema
import sympy
import matplotlib.pyplot as plt
import numpy as np
import json

#with open('paramsFarhan.json') as json_data:
with open('params.json') as json_data:
    params = json.load(json_data)
    json_data.close()

temperatures = np.linspace(300,420,6)
temperatures = np.linspace(50,420,15)
#temperatures = [400.]
systemrates = []
systemrates_s = []

W01, W10, W12, W21, W23, W32, W34, W43, W45, W54, W56 = sympy.symbols('W01 W10 W12 W21 W23 W32 W34 W43 W45 W54 W56')

_between_gs = 12*W01*W12*W23*W34*W45*W56/(2*W10*W21*W32*W43*W54 + W10*W21*W32*W43*W56 + W10*W21*W32*W45*W56 + W10*W21*W34*W45*W56 + W10*W23*W34*W45*W56 + 2*W12*W23*W34*W45*W56)

_between_gs_no_stat = W01*W12*W23*W34*W45*W56/(W10*W21*W32*W43*W54 + W10*W21*W32*W43*W56 + W10*W21*W32*W45*W56 + W10*W21*W34*W45*W56 + W10*W23*W34*W45*W56 + W12*W23*W34*W45*W56)

_calc_between_gs = sympy.lambdify([[W01, W10, W12, W21, W23, W32, W34, W43, W45, W54, W56]], _between_gs, "numpy")

_calc_between_gs_no_stat = sympy.lambdify([[W01, W10, W12, W21, W23, W32, W34, W43, W45, W54, W56]], _between_gs_no_stat, "numpy")

for T in temperatures:

    withField = False #define wehter we use field in our calculations
    _phi_H = params['system']['H_angle']
    _H = params['system']['H']

    phik = np.array([0., 5.*pi/3., 4.*pi/3., pi, 2.*pi/3., pi/3.]) #anisotropy values

    _K = params['system']['K']
    _KP = params['system']['K_p']
    _V = params['system']['V']
    _R = params['system']['R']
    _M = params['system']['M']
    y = 1.76e7/_M/_V # in G
    kb = 8.617e-5

    #define constants
    K, V, M, R, H = sympy.symbols("K V M R H")
    r_12, r_13, r_14, r_15, r_16, r_23, r_24, r_25, r_26, r_34, r_35, r_36, r_45, r_46, r_56 = sympy.symbols("r_12 r_13 r_14 r_15 r_16 r_23 r_24 r_25 r_26 r_34 r_35 r_36 r_45 r_46 r_56")
    theta_r_12, theta_r_13, theta_r_14, theta_r_15, theta_r_16, theta_r_23, theta_r_24, theta_r_25, theta_r_26, theta_r_34, theta_r_35, theta_r_36, theta_r_45, theta_r_46, theta_r_56 = sympy.symbols("theta_r_12 theta_r_13 theta_r_14 theta_r_15 theta_r_16 theta_r_23 theta_r_24 theta_r_25 theta_r_26 theta_r_34 theta_r_35 theta_r_36 theta_r_45 theta_r_46 theta_r_56")
    phi_r_12, phi_r_13, phi_r_14, phi_r_15, phi_r_16, phi_r_23, phi_r_24, phi_r_25, phi_r_26, phi_r_34, phi_r_35, phi_r_36, phi_r_45, phi_r_46, phi_r_56 = sympy.symbols("phi_r_12 phi_r_13 phi_r_14 phi_r_15 phi_r_16 phi_r_23 phi_r_24 phi_r_25 phi_r_26 phi_r_34 phi_r_35 phi_r_36 phi_r_45 phi_r_46 phi_r_56")

    #define coordinates
    theta_1, theta_2, theta_3, theta_4, theta_5, theta_6 = sympy.symbols("theta_1 theta_2 theta_3 theta_4 theta_5 theta_6")
    phi_1, phi_2, phi_3, phi_4, phi_5, phi_6 = sympy.symbols("phi_1 phi_2 phi_3 phi_4 phi_5 phi_6")

    #define anisotropy coordinates
    theta_K_1, theta_K_2, theta_K_3, theta_K_4, theta_K_5, theta_K_6 = sympy.symbols("theta_K_1 theta_K_2 theta_K_3 theta_K_4 theta_K_5 theta_K_6")
    phi_K_1, phi_K_2, phi_K_3, phi_K_4, phi_K_5, phi_K_6 = sympy.symbols("phi_K_1 phi_K_2 phi_K_3 phi_K_4 phi_K_5 phi_K_6")

    #Define external field coordinates
    theta_H, phi_H = sympy.symbols("theta_H phi_H")


    def spherical_dot_product(th_1, phi_1, r_1, th_2, phi_2, r_2):
        return sympy.sin(th_1)*sympy.sin(th_2)*sympy.cos(phi_1 - phi_2) + sympy.cos(th_1)*sympy.cos(th_2)

    #========== Anisotropy ===========
    def uniti_anisotropy_member(th_i, phi_i, phi_K_i):
        return K*V*(sympy.sin(th_i)*sympy.cos(phi_i - phi_K_i))**2

    anis = 0
    for t_i, p_i, p_i_k in zip([theta_1, theta_2, theta_3, theta_4, theta_5, theta_6],[phi_1, phi_2, phi_3, phi_4, phi_5, phi_6], [phi_K_1, phi_K_2, phi_K_3, phi_K_4, phi_K_5, phi_K_6]):
        anis += uniti_anisotropy_member(t_i, p_i, p_i_k )

    anis = anis.subs([
    (phi_K_1,phik[0]),
    (phi_K_2,phik[1]),
    (phi_K_3,phik[2]),
    (phi_K_4,phik[3]),
    (phi_K_5,phik[4]),
    (phi_K_6,phik[5])])
    #========== Anisotropy ===========

    #========== Zeeman     ===========
    def uniti_zeeman_member(th_i,phi_i, th_H, ph_H):
        return M*V*H*sympy.sin(th_i)*sympy.cos(phi_i-ph_H)
    zeeman = 0
    for t_i, p_i in zip([theta_1, theta_2, theta_3, theta_4, theta_5, theta_6],[phi_1, phi_2, phi_3, phi_4, phi_5, phi_6]):
        zeeman += uniti_zeeman_member(t_i, p_i, theta_H, phi_H)
    #========== Zeeman     ===========


    #========== Dipolar  =============
    def unit_dipolar_member(th_m, ph_m, th_n, ph_n, th_r_nm, ph_r_nm, r_nm):
        _first = spherical_dot_product( th_m, ph_m, 1,th_n, ph_n, 1)
        #_second = 3*spherical_dot_product( th_m, ph_m, 1,th_r_nm, ph_r_nm, 1)*spherical_dot_product( th_n, ph_n, 1,th_r_nm, ph_r_nm, 1)
        _second = 3*sympy.sin(th_m)*sympy.cos(ph_m-ph_r_nm)*sympy.sin(th_n)*sympy.cos(ph_n-ph_r_nm)
        return (_second - _first)*4*pi*M*M*V*V/r_nm/r_nm/r_nm


    _d_1_2 = unit_dipolar_member(theta_1, phi_1, theta_2, phi_2, theta_r_12, phi_r_12, r_12)
    _d_1_3 = unit_dipolar_member(theta_1, phi_1, theta_3, phi_3, theta_r_13, phi_r_13, r_13)
    _d_1_4 = unit_dipolar_member(theta_1, phi_1, theta_4, phi_4, theta_r_14, phi_r_14, r_14)
    _d_1_5 = unit_dipolar_member(theta_1, phi_1, theta_5, phi_5, theta_r_15, phi_r_15, r_15)
    _d_1_6 = unit_dipolar_member(theta_1, phi_1, theta_6, phi_6, theta_r_16, phi_r_16, r_16)

    _d_2_3 = unit_dipolar_member(theta_3, phi_2, theta_3, phi_3, theta_r_23, phi_r_23, r_23)
    _d_2_4 = unit_dipolar_member(theta_2, phi_2, theta_4, phi_4, theta_r_24, phi_r_24, r_24)
    _d_2_5 = unit_dipolar_member(theta_2, phi_2, theta_5, phi_5, theta_r_25, phi_r_25, r_25)
    _d_2_6 = unit_dipolar_member(theta_2, phi_2, theta_6, phi_6, theta_r_26, phi_r_26, r_26)

    _d_3_4 = unit_dipolar_member(theta_3, phi_3, theta_4, phi_4, theta_r_34, phi_r_34, r_34)
    _d_3_5 = unit_dipolar_member(theta_3, phi_3, theta_5, phi_5, theta_r_35, phi_r_35, r_35)
    _d_3_6 = unit_dipolar_member(theta_3, phi_3, theta_6, phi_6, theta_r_36, phi_r_36, r_36)

    _d_4_5 = unit_dipolar_member(theta_4, phi_4, theta_5, phi_5, theta_r_45, phi_r_45, r_45)
    _d_4_6 = unit_dipolar_member(theta_4, phi_4, theta_6, phi_6, theta_r_46, phi_r_46, r_46)

    _d_5_6 = unit_dipolar_member(theta_5, phi_5, theta_6, phi_6, theta_r_56, phi_r_56, r_56)

    dipole = _d_1_2 + _d_1_3 + _d_1_4 + _d_1_5 + _d_1_6 + _d_2_3 + _d_2_4 + \
             _d_2_5 + _d_2_6 +_d_3_4 + _d_3_5 + _d_3_6 + _d_4_5 + _d_4_6 + _d_5_6

    dipole = dipole.subs([
    (theta_r_12, sympy.pi/2),
    (theta_r_13, sympy.pi/2),
    (theta_r_14, sympy.pi/2),
    (theta_r_15, sympy.pi/2),
    (theta_r_16, sympy.pi/2),
    (theta_r_23, sympy.pi/2),
    (theta_r_24, sympy.pi/2),
    (theta_r_25, sympy.pi/2),
    (theta_r_26, sympy.pi/2),
    (theta_r_34, sympy.pi/2),
    (theta_r_35, sympy.pi/2),
    (theta_r_36, sympy.pi/2),
    (theta_r_45, sympy.pi/2),
    (theta_r_46, sympy.pi/2),
    (theta_r_56, sympy.pi/2)])

    dipole = dipole.subs([
    (r_12, R),
    (r_13, sympy.sqrt(3) * R),
    (r_14, 2 * R),
    (r_15, sympy.sqrt(3) * R),
    (r_16, R),
    (r_23, R),
    (r_24, sympy.sqrt(3) * R),
    (r_25, 2 * R),
    (r_26, sympy.sqrt(3) * R),
    (r_34, R),
    (r_35, sympy.sqrt(3) * R),
    (r_36, 2 * R ),
    (r_45, R),
    (r_46, sympy.sqrt(3) * R),
    (r_56, R)])

    dipole = dipole.subs([
    (phi_r_12,5 * sympy.pi / 6),
    (phi_r_13,4 * sympy.pi / 6),
    (phi_r_14,3 * sympy.pi / 6),
    (phi_r_15,2 * sympy.pi / 6),
    (phi_r_16,sympy.pi / 6),
    (phi_r_23,3 * sympy.pi / 6 ),
    (phi_r_24,2 * sympy.pi / 6),
    (phi_r_25,sympy.pi / 6),
    (phi_r_26,0),
    (phi_r_34,sympy.pi / 6),
    (phi_r_35,0.),
    (phi_r_36,5 * sympy.pi / 6),
    (phi_r_45,5 * sympy.pi / 6),
    (phi_r_46,4 * sympy.pi / 6),
    (phi_r_56,3 * sympy.pi / 6)])
    #========== Dipolar  =============

    #========== Easy plane============
    def uniti_easy_plane_member(th_i):
        return V*_KP*sympy.cos(th_i)*sympy.cos(th_i)

    easy_plane = 0.
    for t_i in [theta_1, theta_2, theta_3, theta_4, theta_5, theta_6]:
        easy_plane += uniti_easy_plane_member(t_i)

    #========== Easy plane============
    if withField:
        energy = -anis-dipole-zeeman+easy_plane
        energy = energy.subs([
            (phi_H, _phi_H),
            (H, _H)])
    else:
        energy = -anis-dipole+easy_plane

    hessian = sympy.hessian(energy,[phi_1, phi_2, phi_3, phi_4, phi_5, phi_6,theta_1, theta_2, theta_3, theta_4, theta_5, theta_6])


    #load path data
    #path = np.load("/Users/Sergei/path{:+f}{:+f}vF.npy".format(_H, _phi_H))
    energy_path = np.load("{}energy_path{:.2f}+{:.1f}.npy".format(params['programm']['output_path'], params['system']['H_angle'],params['system']['H']))
    path = np.load("{}path{:.2f}+{:.1f}.npy".format(params['programm']['output_path'], params['system']['H_angle'], params['system']['H']))


    #Input data 
    h_a_and_d = hessian.subs([
        (theta_1, sympy.pi/2),
        (theta_2, sympy.pi/2),
        (theta_3, sympy.pi/2),
        (theta_4, sympy.pi/2),
        (theta_5, sympy.pi/2),
        (theta_6, sympy.pi/2)])

    en = energy.subs([
        (theta_1, sympy.pi/2),
        (theta_2, sympy.pi/2),
        (theta_3, sympy.pi/2),
        (theta_4, sympy.pi/2),
        (theta_5, sympy.pi/2),
        (theta_6, sympy.pi/2)])


    fac = 1.
    result_hessian = h_a_and_d.subs([
        (K,(_K)*fac),
        (V,(_V)/fac),
        (R,_R),
        (M,_M)])

    result_energy = en.subs([
        (K,(_K)*fac),
        (V,(_V)/fac),
        (R,_R),
        (M,_M)])


    #list minimums
    minimums = (path[:,argrelextrema(energy_path, np.less_equal)[0]])

    #list sadles
    maximums = (path[:,argrelextrema(energy_path, np.greater)[0]])

    rates = []
    rates_s = []

    for i in range(0, 6):
        tmp_min_l = minimums[:,i]
        tmp_min_r = minimums[:,i+1]

        tmp_sad = maximums[:,i]

        hes_min_l = result_hessian.subs([
            (phi_1, tmp_min_l[0]),
            (phi_2, tmp_min_l[1]),
            (phi_3, tmp_min_l[2]),
            (phi_4, tmp_min_l[3]),
            (phi_5, tmp_min_l[4]),
            (phi_6, tmp_min_l[5])])

        hes_min_r = result_hessian.subs([
            (phi_1, tmp_min_r[0]),
            (phi_2, tmp_min_r[1]),
            (phi_3, tmp_min_r[2]),
            (phi_4, tmp_min_r[3]),
            (phi_5, tmp_min_r[4]),
            (phi_6, tmp_min_r[5])])

        hes_sad = result_hessian.subs([
            (phi_1, tmp_sad[0]),
            (phi_2, tmp_sad[1]),
            (phi_3, tmp_sad[2]),
            (phi_4, tmp_sad[3]),
            (phi_5, tmp_sad[4]),
            (phi_6, tmp_sad[5])])

        en_min_l = result_energy.subs([
            (phi_1, tmp_min_l[0]),
            (phi_2, tmp_min_l[1]),
            (phi_3, tmp_min_l[2]),
            (phi_4, tmp_min_l[3]),
            (phi_5, tmp_min_l[4]),
            (phi_6, tmp_min_l[5])]).evalf()/1.60217657e-12

        en_min_r = result_energy.subs([
            (phi_1, tmp_min_r[0]),
            (phi_2, tmp_min_r[1]),
            (phi_3, tmp_min_r[2]),
            (phi_4, tmp_min_r[3]),
            (phi_5, tmp_min_r[4]),
            (phi_6, tmp_min_r[5])]).evalf()/1.60217657e-12

        en_s = result_energy.subs([
            (phi_1, tmp_sad[0]),
            (phi_2, tmp_sad[1]),
            (phi_3, tmp_sad[2]),
            (phi_4, tmp_sad[3]),
            (phi_5, tmp_sad[4]),
            (phi_6, tmp_sad[5])]).evalf()/1.60217657e-12

        de_l = abs(en_min_l)-abs(en_s)
        de_r = abs(en_min_r)-abs(en_s)

        he_min_l = np.array(np.array(hes_min_l.evalf()).astype(float))
        he_min_r = np.array(np.array(hes_min_r.evalf()).astype(float))
        he_sad = np.array(np.array(hes_sad.evalf()).astype(float))

        eig_min_l = np.linalg.eig(he_min_l)[0]
        eig_min_r = np.linalg.eig(he_min_r)[0]

        eig_sad = np.linalg.eig(he_sad)[0]
        eig_v_sad = np.linalg.eig(he_sad)[1]

        Vel = np.zeros((12,12))
        V_r = np.copy(he_sad)

       # tt0 = V_r[6,:]
       # tt1 = V_r[7,:]
       # tt2 = V_r[8,:]
       # tt3 = V_r[9,:]
       # tt4 = V_r[10,:]
       # tt5 = V_r[11,:]
       #
       # tt6 = V_r[0,:]
       # tt7 = V_r[1,:]
       # tt8 = V_r[2,:]
       # tt9 = V_r[3,:]
       # tt10 = V_r[4,:]
       # tt11 = V_r[5,:]
       #
       # Vel[0,:]=y*np.array([tt0[0],tt0[6], tt0[1],tt0[7], tt0[2],tt0[8], tt0[3],tt0[9], tt0[4],tt0[10], tt0[5],tt0[11]])
       # Vel[1,:]=y*np.array([tt1[0],tt1[6], tt1[1],tt1[7], tt1[2],tt1[8], tt1[3],tt1[9], tt1[4],tt1[10], tt1[5],tt1[11]])
       # Vel[2,:]=y*np.array([tt2[0],tt2[6], tt2[1],tt2[7], tt2[2],tt2[8], tt2[3],tt2[9], tt2[4],tt2[10], tt2[5],tt2[11]])
       # Vel[3,:]=y*np.array([tt3[0],tt3[6], tt3[1],tt3[7], tt3[2],tt3[8], tt3[3],tt3[9], tt3[4],tt3[10], tt3[5],tt3[11]])
       # Vel[4,:]=y*np.array([tt4[0],tt4[6], tt4[1],tt4[7], tt4[2],tt4[8], tt4[3],tt4[9], tt4[4],tt4[10], tt4[5],tt4[11]])
       # Vel[5,:]=y*np.array([tt5[0],tt5[6], tt5[1],tt5[7], tt5[2],tt5[8], tt5[3],tt5[9], tt5[4],tt5[10], tt5[5],tt5[11]])
       #
       # Vel[6,:]=-y*np.array([tt6[0],tt6[6], tt6[1],tt6[7], tt6[2],tt6[8], tt6[3],tt6[9], tt6[4],tt6[10], tt6[5],tt6[11]])
       # Vel[7,:]=-y*np.array([tt7[0],tt7[6], tt7[1],tt7[7], tt7[2],tt7[8], tt7[3],tt7[9], tt7[4],tt7[10], tt7[5],tt7[11]])
       # Vel[8,:]=-y*np.array([tt8[0],tt8[6], tt8[1],tt8[7], tt8[2],tt8[8], tt8[3],tt8[9], tt8[4],tt8[10], tt8[5],tt8[11]])
       # Vel[9,:]=-y*np.array([tt9[0],tt9[6], tt9[1],tt9[7], tt9[2],tt9[8], tt9[3],tt9[9], tt9[4],tt9[10], tt9[5],tt9[11]])
       # Vel[10,:]=-y*np.array([tt10[0],tt10[6], tt10[1],tt10[7], tt10[2],tt10[8], tt10[3],tt10[9], tt10[4],tt10[10], tt10[5],tt10[11]])
       # Vel[11,:]=-y*np.array([tt11[0],tt11[6], tt11[1],tt11[7], tt11[2],tt11[8], tt11[3],tt11[9], tt11[4],tt11[10], tt11[5],tt11[11]])

        Vel[0:6:,0:12] = y*V_r[6:12:,0:12]
        Vel[6:12:,0:12] = -y*V_r[0:6:,0:12]

        A = np.dot(np.linalg.inv(eig_v_sad), Vel)
        A = np.dot(A, eig_v_sad)
        a = np.copy(A[0,:])
        a[0] = 0.

        e_m_l = np.copy(eig_min_l)
        e_m_r = np.copy(eig_min_r)
        e_s = np.copy(eig_sad)
        e_s[0] = 1

        det_m_l = np.sqrt(e_m_l.prod())
        det_m_r = np.sqrt(e_m_r.prod())
        det_s = np.sqrt(e_s.prod())

        pr = np.sqrt(((a**2)/e_s).sum())


        pre_exp_l = (1./2./pi)*pr*(det_m_l/det_s)
        pre_exp_r = (1./2./pi)*pr*(det_m_r/det_s)


        exponent_l =  e**(-de_l/(kb*T))
        exponent_r =  e**(-de_r/(kb*T))

        rates.append(exponent_l*pre_exp_l)
        rates.append(exponent_r*pre_exp_r)
        print( pre_exp_l/1e8, de_l)
        print( pre_exp_r/1e8, de_r)

    print(T)
    print( rates)


    #system_rate = rates[0] - (rates[0]*rates[1]/(rates[1]+rates[2] - (rates[2]*rates[3]/(rates[3] + rates[4] - (rates[4]*rates[5]/(rates[5]+rates[6] - (rates[6]*rates[7]/(rates[7] + rates[8] - rates[8]*rates[9]/(rates[9] + rates[10])))))))))
    #system_rate = _calc_between_gs(rates[0:11])
    system_rate = _calc_between_gs_no_stat(rates[0:11])
    systemrates.append(system_rate)

print('results:')
print(systemrates)
print(temperatures)

temp1 = np.log(np.array(systemrates,dtype=float))

slope, intercept, r_value, p_value, std_err = stats.linregress(1./temperatures,temp1)

print('A as a prefactor', e**intercept/1e8)
print('effective barrier', -slope*kb)
print('life time, including all paths at 420K', 1/(e**intercept*e**(slope/420)))

plt.plot(1000./temperatures,temp1 , '-b')
plt.show()
