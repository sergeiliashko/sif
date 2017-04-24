#!/bin/bash

# This script helps to set up all neccary enviroment vars for the 
# intel mkl and compilers. Cannot put that to the make file since 
# on run make doesn't change the enviroment.

echo "Start"
echo "Reading config for intel and neb dylib " >&2
source "required_params/intelpath.cfg"
source "required_params/nebpath.cfg"

export DYLD_LIBRARY_PATH="$intel_dyldlib_path:$neb_dyldlib_path"

echo "Succes"
