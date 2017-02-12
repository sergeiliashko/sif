#ifndef ENERGY_H
#define ENERGY_H
void shape_anisotropy_energy(
    const double *system_angles,
    const double *anisotropy_angles,
    const double *anisotropy_values,
    const double *islands_volumes,
    const double factor,
    const int size,
    double *output);

void zeeman_energy(
    const double *system_angles,
    const double *ext_field_angles,
    const double *ext_field_values,
    const double *islands_volumes,
    const double *magnetisation_values,
    const double factor,
    const int size,
    double *output);

void dipole_dipole_energy(
    const double *system_angles,
    const double *magnetisation_values,
    const double *islands_volumes,
    const double *distances,
    const double *distance_unit_vectors,
    const double factor,
    const int size,
    double *output);

void calculateEnergyAndGradient(
    const double *set_of_images, // should be (m*n) size
    const double *magnetisation_values,
    const double *anisotropy_angles,
    const double *anisotropy_values,
    const double ext_field_value,
    const double ext_field_angle,
    const double *islands_volumes,
    const double *distances,
    const double *distance_unit_vectors,
    const double factor,
    const int m,
    const int n,
    double *result,
    double *gresult);

void calculateTangetAndSpringForces(
    const double *set_of_images, // should be (m*n) size
    const double *energy_path,
    int n,
    int m,
    double springConstant,
    double *result,
    double *springForces);

void calculatePerpendicularForces(
    const double *gradient_path,
    const double *norm_tangets,
    int n,
    int m,
    double *result);

void calculateTrueForces(
    const double *springForces,
    const double *perpendicularForces,
    int n,
    int m,
    double *result);

void makeStep(
    double *path,
    const double *forces,
    double n,
    double m,
    double dt);
#endif
