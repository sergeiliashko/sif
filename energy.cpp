#include <iostream>
#include <mathimf.h>
#include "mkl.h"

inline int k(int i, int j, int size)
{
  return i*size+j; // row column
  //return i+j*size;// column row
}
// What to say about this thing
#ifdef __cplusplus
extern "C" void shape_anisotropy_energy( const double *system_angles, const double *anisotropy_angles, const double *anisotropy_values, const double *islands_volumes, const double factor, const int size, double *output)
#else
void shape_anisotropy_energy(
    const double *system_angles,
    const double *anisotropy_angles,
    const double *anisotropy_values,
    const double *islands_volumes,
    const double factor,
    const int size,
    double *output)
#endif
{
  double *mid = new double[size]{};
  for(int i = 0; i < size; i++)
  {
    mid[i] = system_angles[i] - anisotropy_angles[i];
  }

  for(int i = 0; i < size; i++)
  {
    output[i] = factor * anisotropy_values[i] * islands_volumes[i]
        * pow(cos(mid[i]), 2.0);
  }
  delete[] mid;
  mid = nullptr;
}

#ifdef __cplusplus
extern "C" void zeeman_energy(
    const double *system_angles,
    const double *ext_field_angles,
    const double *ext_field_values,
    const double *islands_volumes,
    const double *magnetisation_values,
    const double factor,
    const int size,
    double *output)
#else
void zeeman_energy(
    const double *system_angles,
    const double *ext_field_angles,
    const double *ext_field_values,
    const double *islands_volumes,
    const double *magnetisation_values,
    const double factor,
    const int size,
    double *output)
#endif
{
  double *mid = new double[size]{};
    for (int i = 0; i < size; i++) {
        mid[i] = system_angles[i] - ext_field_angles[i];
    }
  //vdSub(size, system_angles, ext_field_angles, mid);
  #pragma omp parallel for
  for(int i = 0; i < size; i++)
  {
    output[i] = factor * ext_field_values[i] * islands_volumes[i] * magnetisation_values[i] * cos(mid[i]);
  }
  delete[] mid;
  mid = nullptr;
}

#ifdef __cplusplus
extern "C" void dipole_dipole_energy(
    const double *system_angles,
    const double *magnetisation_values,
    const double *islands_volumes,
    const double *distances,
    const double *distance_unit_vectors,
    const double factor,
    const int size,
    double *output)
#else
void dipole_dipole_energy(
    const double *system_angles,
    const double *magnetisation_values,
    const double *islands_volumes,
    const double *distances,
    const double *distance_unit_vectors,
    const double factor,
    const int size,
    double *output)
#endif
{
  double pi = 4 * atan(1.0); // Accurate pi calculation
  double *result_j = new double[size*size]{}; // This array is used for inner loope vectorisation
  #pragma omp parallel for
  for(int i = 0; i < size; i++)
  {
    double x = system_angles[i];
    double x_m = magnetisation_values[i];
    double x_v = islands_volumes[i];

    for(int j = 0; j < size; j++)
    {
      if(i != j)
      {
        double y = system_angles[j];
        double y_m = magnetisation_values[j];
        double y_v = islands_volumes[j];
        double d_xy = distances[k(i,j,size)];
        double d_xy_vec = distance_unit_vectors[k(i,j,size)];

        result_j[k(i,j,size)] = (3.0*cos(x - d_xy_vec) * cos(y - d_xy_vec) - cos(x - y)) *
          (4.0*pi*x_m*y_m*x_v*y_v*factor / pow(d_xy,3.0));
      }
    }
  }
    // Obtain result
    for(int a = 0; a < size; a++)
    {
      for(int b = 0; b < size; b++)
      {
        output[a] += result_j[k(a,b,size)];
      }
      output[a] *= 0.5;
    }
  delete[] result_j; // Release the array
  result_j = nullptr; // Set the array pointer to null
}

#ifdef __cplusplus
extern "C" void calculateEnergyAndGradient(
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
    double *gresult)
#else
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
    double *gresult)
#endif
{
  double *output = new double[n*m]{}; // This array is used for inner loop vectorisation
  double pi = 4 * atan(1.0); // Accurate pi calculation
  for(int image = 0; image < m; image++)
  {
    double *mid_dip = new double[n*n]{}; // This array is used for inner loop vectorisation dipole
    double *mid_g_dip = new double[n*n]{}; // This array is used for inner loop vectorisation dipole
    for(int island = 0; island < n; island++)
    {
      // Anisotropy calculation
      output[k(island,image,m)] =
       -factor * anisotropy_values[island] * islands_volumes[island] *
        pow(cos(set_of_images[k(island,image,m)] - anisotropy_angles[island]), 2.0);

      // Anisotropy gradient
      gresult[k(island,image,m)] =
        factor * anisotropy_values[island] * islands_volumes[island] *
        sin(2.0*(set_of_images[k(island,image,m)] - anisotropy_angles[island]));

      // We Always have the anisotropy and dipole but field could be zero.
      if(ext_field_value != 0)
      {
        output[k(island,image,m)] +=
          -factor * ext_field_value * islands_volumes[island] * magnetisation_values[island] *
          cos(set_of_images[k(island,image,m)] - ext_field_angle);

        gresult[k(island,image,m)] +=
          factor * ext_field_value * islands_volumes[island] * magnetisation_values[island] *
          sin(set_of_images[k(island,image,m)] - ext_field_angle);
      }

      // Dipole calculation
      double x = set_of_images[k(island,image,m)];
      double x_m = magnetisation_values[island];
      double x_v = islands_volumes[island];
      for(int other_island = 0; other_island < n; other_island++)
      {
        if(island != other_island)
        {
          double y = set_of_images[k(other_island,image,m)];
          double y_m = magnetisation_values[other_island];
          double y_v = islands_volumes[other_island];
          double d_xy = distances[k(island,other_island,n)];
          double d_xy_vec = distance_unit_vectors[k(island,other_island,n)];

          double coef = (4.0*pi*x_m*y_m*x_v*y_v*factor / pow(d_xy,3.0));
          mid_dip[k(island,other_island,n)] = coef*(3.0*cos(x-d_xy_vec)*cos(y-d_xy_vec) - cos(x-y));
          mid_g_dip[k(island,other_island,n)] = coef*(sin(x-y)-3.0*sin(x-d_xy_vec)*cos(y-d_xy_vec));
        }
      }
    }
    // Still dipole
    for(int a = 0; a < n; a++)
    {
      for(int b = 0; b < n; b++)
      {
        output[k(a,image,m)] += -0.5*mid_dip[k(a,b,n)];
        gresult[k(a,image,m)] += -mid_g_dip[k(a,b,n)];
      }
    }
    delete[] mid_dip;
    delete[] mid_g_dip;
    mid_dip = nullptr;
    mid_g_dip = nullptr;
  }
  for(int image = 0; image < m; image++)
  {
    for(int island = 0; island < n; island++)
    {
      result[image] += output[k(island,image,m)];
    }
  }
  delete[] output;
  output = nullptr;
}

#ifdef __cplusplus
extern "C" void calculateTangetAndSpringForces(
    const double *set_of_images, // should be (m*n) size
    const double *energy_path,
    int n,
    int m,
    double springConstant,
    double *result,
    double *springForces)
#else
void calculateTangetAndSpringForces(
    const double *set_of_images, // should be (m*n) size
    const double *energy_path,
    int n,
    int m,
    double springConstant,
    double *result,
    double *springForces)
#endif
{
  double *tanget_plus = new double[n*(m)]{};
  double *tanget_minus = new double[n*(m)]{};
  for (int image = 1; image < m-1; image++)
  {
    for(int island = 0; island < n; island++)
    {
      tanget_plus[k(island,image,m)] =
        set_of_images[k(island,image+1,m)] - set_of_images[k(island,image,m)];
      tanget_minus[k(island,image,m)] =
        set_of_images[k(island,image,m)] - set_of_images[k(island,image-1,m)];
    }
  }
  for(int image = 1; image < m-1; image++)
  {
    for(int island = 0; island < n; island++)
    {
      double e0 = energy_path[image-1];
      double e1 = energy_path[image];
      double e2 = energy_path[image+1];
      if(e2 > e1 && e1 > e0)
      {
        result[k(island,image,m)] = tanget_plus[k(island,image,m)];
      }
      else if(e2 < e1 && e1 < e0)
      {
        result[k(island,image,m)] = tanget_minus[k(island,image,m)];
      }
      else
      {
        double eMax = fmax(fabs(e2 - e1),fabs(e0 - e1));
        double eMin = fmin(fabs(e2 - e1),fabs(e0 - e1));
        if(e2 > e0)
        {
          result[k(island,image,m)] = tanget_plus[k(island,image,m)]*eMax
            + tanget_minus[k(island,image,m)]*eMin;
        }
        else
        {
          result[k(island,image,m)] = tanget_plus[k(island,image,m)]*eMin
            + tanget_minus[k(island,image,m)]*eMax;
        }
      }
    }
  }
  for(int image = 1;image < m-1; image++ )
  {
    double thisImageNorm = cblas_dnrm2(n,&result[k(0,image,m)],m);
    for(int island = 0; island < n; island++)
    {
      if(thisImageNorm == 0)
      {
        result[k(island,image,m)] = 1.0;
      }
      else
      {
        result[k(island,image,m)] /= thisImageNorm;
      }
    }
  }

  for(int image = 1;image < m-1; image++)
  {
    double thisNormSpring =
      cblas_dnrm2(n,&tanget_plus[k(0,image,m)],m) -
      cblas_dnrm2(n,&tanget_minus[k(0,image,m)],m);
    thisNormSpring *= springConstant;

    for(int island = 0; island < n; island++)
    {
      springForces[k(island,image,m)] = thisNormSpring*result[k(island,image,m)];
    }
  }
 delete[] tanget_plus;
 tanget_plus=nullptr;
 delete[] tanget_minus;
 tanget_minus=nullptr;
}

#ifdef __cplusplus
extern "C" void calculatePerpendicularForces(
    const double *gradient_path,
    const double *norm_tangets,
    int n,
    int m,
    double *result)
#else
void calculatePerpendicularForces(
    const double *gradient_path,
    const double *norm_tangets,
    int n,
    int m,
    double *result)
#endif
{
  double *tangets_gradient_dot_product = new double[m]{};

  for(int image = 1; image < m-1; image++)
  {
    tangets_gradient_dot_product[image] =
      cblas_ddot(n, &gradient_path[k(0,image,m)],m,&norm_tangets[k(0,image,m)],m);
  }

  for(int image = 1; image < m-1; image++)
  {
    for(int island = 0; island < n; island++)
    {
      result[k(island,image,m)] = gradient_path[k(island,image,m)] - norm_tangets[k(island,image,m)]*tangets_gradient_dot_product[image];
    }
  }
  delete[] tangets_gradient_dot_product;
  tangets_gradient_dot_product=nullptr;
}

#ifdef __cplusplus
extern "C" void calculateTrueForces(
    const double *springForces,
    const double *perpendicularForces,
    int n,
    int m,
    double *result)
#else
void calculateTrueForces(
    const double *springForces,
    const double *perpendicularForces,
    int n,
    int m,
    double *result)
#endif
{
  vdSub(n*m,springForces,perpendicularForces,result);
}

#ifdef __cplusplus
extern "C" void makeStep(
    double *path,
    const double *forces,
    int n,
    int m,
    double dt)
#else
void makeStep(
    double *path,
    const double *forces,
    int n,
    int m,
    double dt)
#endif
{
  cblas_daxpy(n*m,dt,forces,1,path,1);
  for (int image = 1; image < m-1; image++)
  {
    for (int island = 0; island < n; island++)
    {
      path[k(island,image,m)] = fmod(path[k(island,image,m)],atan(1.0)*8.0); //2*pi
    }
  }
}
