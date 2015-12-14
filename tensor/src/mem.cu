#include "../inc/mem.h"

#include <cstdlib>
#include <stdexcept>
#include <iostream>

namespace tensor {

#define CUDA_MEM_CALL(x) cudaError_t err = (x); if (err != cudaSuccess) { std::cerr<<"error: "<<cudaGetErrorString(err)<<std::endl; throw std::bad_alloc(); }

template<>
void* Mem<CPU>::malloc(int bytes) {
	void* result;
	CUDA_MEM_CALL(cudaMallocHost((void**)&result, bytes));
	return result;
}

template<>
void Mem<CPU>::free(void* ptr) {
  CUDA_MEM_CALL(cudaFreeHost(ptr));
}


template<>
void* Mem<GPU>::malloc(int bytes) {
	void* result;
	CUDA_MEM_CALL(cudaMalloc((void**)&result, bytes));
	return result;
}


template<>
void Mem<GPU>::free(void* ptr) {
  CUDA_MEM_CALL(cudaFree(ptr));
}

}
