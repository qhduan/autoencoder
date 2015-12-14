NVCC = nvcc -arch sm_30 --compiler-options '-fPIC -Wall'
CXX  = g++  -Wall -std=c++11 -fPIC
CXXFLAGS = -g -O2
LIB =  -L. -ltensor -lcublas -lcudart -lopenblas
.PHONY: clean doc cleandoc main.o test all tensor libtensor.so

all: tensor libtensor.so

tensor: main.o libtensor.so
	$(CXX) $(CXXFLAGS) -o $@ main.o $(LIB) -Wl,-rpath,./

test: test_mat_cpu test_mat_gpu test_vec_cpu test_vec_gpu

test_mat_cpu: libtensor.so
	$(CXX) $(CXXFLOAGS) -o mtc src/mattest.cc -D MATTEST_CPU $(LIB) -lboost_unit_test_framework -Wl,-rpath,./

test_mat_gpu: libtensor.so
	$(CXX) $(CXXFLOAGS) -o mtg src/mattest.cc -D MATTEST_GPU $(LIB) -lboost_unit_test_framework -Wl,-rpath,./

test_vec_cpu: libtensor.so
	$(CXX) $(CXXFLOAGS) -o vtc src/vectest.cc -D MATTEST_CPU $(LIB) -lboost_unit_test_framework -Wl,-rpath,./

test_vec_gpu: libtensor.so
	$(CXX) $(CXXFLOAGS) -o vtg src/vectest.cc -D MATTEST_GPU $(LIB) -lboost_unit_test_framework -Wl,-rpath,./

libtensor.so: mem.o mempool.o blas.o operation.o tool.o
	$(CXX) -shared -o $@ $^

mempool.o: src/mempool.cc inc/mempool.h
	$(CXX) -c $(CXXFLAGS) -o $@ $(firstword $^)

main.o:	src/main.cc
	$(CXX) -c $(CXXFLAGS) -o $@ $(firstword $^)

## for cuda:

tool.o: src/tool.cu inc/tool.h
	$(NVCC) -c $(CXXFLAGS) -o $@ $(firstword $^)

mem.o: src/mem.cu inc/mem.h
	$(NVCC) -c $(CXXFLAGS) -o $@ $(firstword $^)

blas.o: src/blas.cu inc/blas.h
	$(NVCC) -c $(CXXFLAGS) -o $@ $(firstword $^)

operation.o: src/operation.cu inc/cpu.hpp inc/gpu.hpp
	$(NVCC) -c $(CXXFLAGS) -o $@ $(firstword $^)

##

clean:
	rm -rf tensor *.o *.a *.so doc/* mtg mtc

doc:
	rm -rf doc/*
	doxygen Doxyfile.in
	ln -s html/index.html doc/index.html

cleandoc:
	rm -rf doc/*
