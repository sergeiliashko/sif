#ifndef ENERGY_H
#define ENERGY_H
void find_mep(
    const double *distances,
    const double *distance_unit_vectors,
    const double *M,
    const double *V,
    const double *K,
    const double *kAngles,
    const double *mAngels,
    double factor,
    int n,
    int m,
    double *path,
    double *energy_path,
    double dt,
    double epsilon,
    double k_spring,
    int maxiter,
    bool use_ci,
    bool use_fi);

#endif
