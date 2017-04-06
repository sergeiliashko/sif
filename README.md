The Spin Ice Farm is here. It helps you to grow an artificial 2D spin ice system and harvest such things as remagnetisation processes, activation barriers and lifetimes.
No magic or earth gods, just NEB algorithm, polar coordinate system and Harmonic Transition State Theory(HTST).


## How to use
1. fill in intelpath.cfg file
2. run make buildsharedneblib 
3. set up parameters
  - system params
  - lattice coords
4. ???
5. PROFIT


## What's included
- *intelpath.cfg* Is the place to put your intel compile vars
- *setupenv.sh* Is the script that should set up all enviroment variables are required for the intel compiler to run 
- *Makefile* 
- CPP part
  - *energy* here we calculate all properties related to the energy(Gradients, NEB Forces and Energy itslef)
  - *minimization* well, the name is self explanatory. Here we use NEB with stepest decent
- Python part
  - *geometry* this script gets different properties from the lattice geometry file (centers of islands, angles they have with Ox, distances they have between centers etc.)
  - *paramsfactory* get a list of params for programm and energy from params files
  - *plotenergy* get a plot of the enrgy vs MEP
  - *plotsates* get minima/maxima states along MEP
  - *ratecalc* calculate transition rate
  - *sif* ???
  - *[energy|pathminimizer]_module* interface between python and cpp files 

