CXX  = g++ -Wall -std=c++11 -fPIC
CXXFLAGS = -g -O2 -I ../tensor/inc
LIB = -L../tensor -ltensor -lcublas -lcudart -lopenblas
.PHONY: clean doc cleandoc main.o all neural

all: neural

neural: main.o
	$(CXX) $(CXXFLAGS) -o $@ main.o $(LIB) -Wl,--hash-style=sysv,-rpath,./

main.o:	src/main.cc
	$(CXX) -c $(CXXFLAGS) -o $@ $(firstword $^)

##

clean:
	rm -rf neural *.o *.a

doc:
	rm -rf doc/*
	doxygen Doxyfile.in
	ln -s html/index.html doc/index.html

cleandoc:
	rm -rf doc/*
