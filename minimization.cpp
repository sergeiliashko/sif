#include <iostream>
#include "energy.h"
#include <mathimf.h>
#include <fstream>
#include "mkl.h"
#include <vector>


std::vector<int> argrelextrem(const double *arr, const int size, std::function<bool (double,double)> comparator)
{
  std::vector<int> result;
  for(int i = 0; i < size; i++)
  {
    if(i == 0 && comparator(arr[i],arr[i+1]))
    {
      result.push_back(i);
    }
    else if(i == (size - 1) && comparator(arr[i], arr[i-1]))
    {
      result.push_back(i);
    }
    else if(comparator(arr[i] , arr[i+1]) && comparator(arr[i] , arr[i-1]))
    {
      result.push_back(i);
    }
  }
  return result;
}
inline void print2DArray(
    const double *arr, int rows, int columns,
    const std::string &fName,
    int offset_column_l = 0, int offset_column_r = 0,
    int offset_row_t = 0, int offset_row_b = 0)
{
  std::ofstream myfile;
  myfile.open (fName);
  for(int i = offset_row_t; i < rows-offset_row_b; i++)
  {
    for(int j = offset_column_l; j < columns-offset_column_r; j++)
    {
      myfile << arr[i*columns+j] << " ";
    }
    myfile << "\n";
  }
  myfile.close();
}

inline void print1DArray(const double *arr, const int size, const std::string &fName, int offset_l = 0, int offset_r = 0)
{
  std::ofstream myfile;
  myfile.open (fName);
  for(int i = offset_l; i < size-offset_r; i++)
  {
    myfile << arr[i] << " ";
  }
  myfile.close();
}

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
    double dt=0.1,
    double epsilon=1e-8,
    double k_spring=1.0,
    int maxiter=10000,
    bool use_ci=false,
    bool use_fi=false)
{
  double tol = fmax(epsilon, pow(m,-4.0));
  double maxForce = 1.0;
  int iteration = 1;
  double *goutput= new double[m*n]{};
  double *spforces= new double[m*n]{};
  double *tangets= new double[m*n]{};
  double *perforces= new double[m*n]{};
  double *trueforces= new double[m*n]{};
  while(iteration < maxiter && fabs(maxForce) > tol)
  {
    //double *energy_path = new double[m]{};
    for(int image = 0; image < m; image++)
    {
      energy_path[image] = 0.0;
    }

    if(iteration % 100 == 0)
    {
      std::cout << "iter: "<< iteration <<"\n"; 
      std::cout << "mfor: "<< maxForce <<"\n"; 
    }

    calculateEnergyAndGradient(path,M,kAngles,K,V,distances,distance_unit_vectors,factor,m,n,energy_path,goutput);
    calculateTangetAndSpringForces(path,energy_path,n,m,k_spring,tangets,spforces);
    calculatePerpendicularForces(goutput,tangets,n,m,perforces);
    calculateTrueForces(spforces,perforces,n,m,trueforces);

    //TODO a-x =c => x = a- c
    //1. find all minima/maximums
    //2.1 for each max extremum run trueforces[extremum] = -gradient[ex_numb] + (2* dot(gradient[ex_numb], tan_unit_vector[ex_numb]])*tan_uni_vector[ex_numb]) - ci
    //2.2 for each min extremum run trueforces[extremum] = -gradient[ex_numb] - (2* dot(gradient[ex_numb], tan_unit_vector[ex_numb]])*tan_uni_vector[ex_numb]) - cf
    if(use_ci && iteration > 300)
    {
      std::vector<int> maxima = argrelextrem(energy_path, m, ([](double a,double b){return a>b;}));
      for(int image: maxima)
      {
        for(int island = 0; island < n; island++)
        {
          double tmp = goutput[island*m+image] - perforces[island*m+image];
          trueforces[island*m+image] = -goutput[island*m+image] + 2*tmp;
        }
      }
    }

    if(use_fi)
    {
      std::vector<int> minima = argrelextrem(energy_path, m, ([](double a,double b){return a<b;}));
      for(int image: minima)
      {
        for(int island = 0; island < n; island++)
        {
          //trueforces[island*m+image] = -2.0*perforces[island*m+image];
          //trueforces[island*m+image] = -goutput[island*m+image];
          double tmp = goutput[island*m+image] - perforces[island*m+image];
          trueforces[island*m+image] = -goutput[island*m+image]- 2*tmp;
        }
      }
    }

    makeStep(path,trueforces,n,m,dt);

    iteration += 1;
    maxForce = trueforces[cblas_idamax(n,trueforces,1)];
  }
  delete[] goutput;
  delete[] spforces;
  delete[] tangets;
  delete[] perforces;
  delete[] trueforces;
  goutput   =nullptr;
  spforces  =nullptr;
  tangets   =nullptr;
  perforces =nullptr;
  trueforces=nullptr;
}
