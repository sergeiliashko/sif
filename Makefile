buildsharedneblib:
	icpc -dynamiclib -std=c++14 -fPIC -o pathminimizer.dylib energy.cpp -qopenmp-stubs -g -mkl -O3

clean:	
	rm -r *.dSYM
	rm -r *.dylib
