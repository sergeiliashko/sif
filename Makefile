sharedenergylib:
	icpc -dynamiclib -std=c++14 -fPIC -o energy.dylib energy.cpp  -qopenmp-stubs -g -mkl -O3
sharedneblib:
	icpc -dynamiclib -std=c++14 -fPIC -o neb.dylib minimization.cpp /Users/Sergei/Dropbox/Documents/@Archive/2017/SoftDev/sif/energy.dylib -qopenmp-stubs -g -mkl -O3


clean:	
	rm -r *.dSYM
	rm -r *.dylib
