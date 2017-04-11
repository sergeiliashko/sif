from sympy.parsing.sympy_parser import (parse_expr, implicit_multiplication_application)
# n-1 = number of barriers 
def overallRate(k,n):
    return A(n)/B(n)

    def a(i): return -k[i,i+1]*k[i+1,1]
    def b(i): if(i==0): return k[1,2] else: return  k[i+1,i]+k[i+1,i+2]

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

# n number of islands
def constructSymbolicHessian(n):
    def uniti_anisotropy_member(i):
        return f"KV(sin(th_{i})cos(phi_{i} - phi_K_{i}))**2"
    def uniti_easy_plane_member(i):
        return f"K_p*V*cos(th_{i})cos(th_{i})"
    def uniti_zeeman_member(i):
        return f"MVHsin(th_{i})cos(phi_{i}-phi_H"
    def uniti_dipole_memeber(i,k):
        _first = f"sin(th_{i})*sin(th_{k})*cos(phi_{i} - phi_{k}) + cos(th_{i})*cos(th_{k})"
        _second = f"3sin(th_{i})cos(ph_{i}-ph_r_{i}{k})sin(th_{k})cos(ph_{k}-ph_r_{i}{k})"
        return (_second+"-" + _first) + f"4*pi*M**2*V**2/r_{i}{k}**3"

    def desiredParser(string):
        return parse_expr(string, transformations=(standard_transformations + (implicit_multiplication_application,)))

    _res_anis = ""
    _res_zeeman = ""
    _res_dipole = ""
    _res_easyplane = ""
    for h in range(n):
        _res_anis += "-" + uniti_anisotropy_member(h)
        _res_easyplane += "+" + uniti_easy_plane_member(h)
        _res_zeeman += "-" + uniti_easy_plane_member(h)
        for k in range(n):
            if(h<k): _res_dipole += "-" + uniti_dipole_memeber(h,k)
    _symbolic_energy = desiredParser(_res_anis+_res_easyplane+_res_dipole+_res_dipole)
    _symbolic_hessian = sympy.hessian(_symbolic_energy,[parse_expr(f"phi_{i}") for i in range(n)] + [parse_expr(f"theta_{i}") for i in range(n)])
    return _symbolic_hessian



def calculateRate(T, hamiltonianParams, MEP, mep_coordinates):

