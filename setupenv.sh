#!/bin/bash

# This script helps to set up all neccary enviroment vars for the 
# intel mkl and compilers. Cannot put that to the make file since 
# on run make doesn't change the enviroment.

export DYLD_LIBRARY_PATH="/opt/intel/compilers_and_libraries/mac/lib"

echo "Start"
echo "Reading config for intel compiler and mkl" >&2
source intelpath.cfg

echo "Seting up env vars for the mkl" >&2
source "$mklpath/mklvars.sh" $platform

echo "Seting up env vars for the intel compilers" >&2
source "$compilerpath/compilervars.sh" $platform

<<<<<<< Updated upstream
export DYLD_LIBRARY_PATH="/opt/intel/compilers_and_libraries/mac/lib"
=======
>>>>>>> Stashed changes

echo "Succes"
