#ifndef GPU_HPP
#define GPU_HPP
/*!
 * \file gpu.h
 * \brief some GPU operation funcions
 * 		2D convolution
 * 		set a row of matrix
 * 		matrix transpose
 * \author qhduan.com
 * \date 2014-08-08
 */

#include <cuda.h>

#include "../inc/device.hpp"

namespace tensor {


template <typename OP>
__global__ void gpu_vector_call (OP op) {
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  op(i);
}


template <typename OP>
__global__ void gpu_matrix_call (OP op) {
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
	const int j = blockDim.y * blockIdx.y + threadIdx.y;
  op(i, j);
}


template <typename T, typename OP>
__device__ void tile_reduce (int n, T* array, int tid, int i) {
  const int size = BLOCK_NUM_THREADS;

  for (unsigned int s = size; s > 0; s >>= 1) {
    if (tid < s && (tid + s) < size && (i * size + s) < n) {
      array[tid] = OP()(array[tid], array[tid + s]);
    }
    __syncthreads();
  }
}


template <typename T, typename OP>
__device__ void vector_reduce (int n, const T* x, int incx, T* result, const int tid, T* tile) {

  int times = (n - 1) / BLOCK_NUM_THREADS + 1;

  for (int i = 0; i < times; i++) {
    if ( (i * BLOCK_NUM_THREADS + tid) < n ) {
      int index = i * BLOCK_NUM_THREADS + tid;
      tile[tid] = x[index * incx];
    } else {
      tile[tid] = 0;
    }

    tile[tid] = OP().pre(tile[tid]);

    __syncthreads();

    tile_reduce<T, OP>(n, tile, tid, i);

    __syncthreads();

    if (tid == 0) {
      if (i == 0) {
        *result = tile[0];
      } else {
        *result = OP()(*result, tile[0]);
      }
    }
  }

  __syncthreads();

}


template <typename T, typename OP>
__global__ void gpu_x_reduce (int n, const T* x, int incx, T* result) {
  const int tid = threadIdx.x;

  __shared__ T tile[BLOCK_NUM_THREADS];

  vector_reduce<T, OP>(n, x, incx, result, tid, tile);
}


template <typename T, typename OP>
__global__ void gpu_A_each_row_reduce_to_x (int m, int n, const T* A, int lda, T* x, int incx) {
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;

  __shared__ T tile[BLOCK_NUM_THREADS];

  const T* row = &A[bid + 0 * lda]; // column first
  vector_reduce<T, OP>(n, row, lda, &x[bid * incx], tid, tile);
}


template <typename T, typename OP>
__global__ void gpu_A_each_col_reduce_to_x (int m, int n, const T* A, int lda, T* x, int incx) {
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;

  __shared__ T tile[BLOCK_NUM_THREADS];

  const T* col = &A[0 + bid * lda]; // column first
  vector_reduce<T, OP>(m, col, 1, &x[bid * incx], tid, tile);
}


template <typename T, typename OP>
__global__ void gpu_A_reduce (int m, int n, const T* A, int lda, T* result) {
  const int tid = threadIdx.x;
  const int size = BLOCK_NUM_THREADS;

  __shared__ T tile[BLOCK_NUM_THREADS];

  int times = (m * n - 1) / size + 1;

  for (int i = 0; i < times; i++) {

    if ( (i * size + tid) < (m*n)) {
      int index = i * size + tid;
      tile[tid] = A[(index % m) + (index / m) * lda];
    } else {
      tile[tid] = 0;
    }

    tile[tid] = OP().pre(tile[tid]);

    __syncthreads();

    tile_reduce<T, OP>(m*n, tile, tid, i);

    __syncthreads();

    if (tid == 0) {
      if (i == 0) {
        *result = tile[0];
      } else {
        *result = OP()(*result, tile[0]);
      }
    }

  }
}


template <typename T>
__global__ void gpu_A_conv2_B_to_C (const T* A, int ha, int wa, int lda,
  const T* B, int hb, int wb, int ldb, T* C, int hc, int wc, int ldc) {
	const int i = blockDim.x * blockIdx.x + threadIdx.x;// row of C
	const int j = blockDim.y * blockIdx.y + threadIdx.y;// col of C

	if (i < hc && j < wc) {

		T value = 0;
    #pragma unroll
		for (int x = 0; x < hb; x++) {// x: row of B
      #pragma unroll
			for (int y = 0; y < wb; y++) {// y: col of B
				int m = i + x;// row of A
				int n = j + y;// col of A
				value += A[m + lda * n] * B[x + ldb * y];
			}
		}

		C[i + ldc * j] = value;

	}
}


template <typename T, typename OP>
__global__ void gpu_A_rows_op_x_to_A (T* A, int ha, int wa, int lda, const T* x, int incx) {
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
	const int j = blockDim.y * blockIdx.y + threadIdx.y;

  const int index = i + j * lda;
  T a, b;

  if (i < ha && j < wa) {
    a = A[index];
    b = x[j * incx];
  }

  __syncthreads();

  if (i < ha && j < wa) {
    A[index] = OP()(a, b);
  }
}


template <typename T, typename OP>
__global__ void gpu_x_op_A_rows_to_A (const T* x , int incx, T* A, int ha, int wa, int lda) {
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
	const int j = blockDim.y * blockIdx.y + threadIdx.y;

  const int index = i + j * lda;
  T a, b;

  if (i < ha && j < wa) {
    a = x[j * incx];
    b = A[index];
  }

  __syncthreads();

  if (i < ha && j < wa) {
    A[index] = OP()(a, b);
  }
}


template <typename T, typename OP>
__global__ void gpu_A_cols_op_x_to_A (T* A, int ha, int wa, int lda, const T* x, int incx) {
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
	const int j = blockDim.y * blockIdx.y + threadIdx.y;

  const int index = i + j * lda;
  T a, b;

  if (i < ha && j < wa) {
    a = A[index];
    b = x[i * incx];
  }

  __syncthreads();

  if (i < ha && j < wa) {
    A[index] = OP()(a, b);
  }
}


template <typename T, typename OP>
__global__ void gpu_x_op_A_cols_to_A (const T* x, int incx, T* A, int ha, int wa, int lda) {
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
	const int j = blockDim.y * blockIdx.y + threadIdx.y;

  const int index = i + j * lda;
  T a, b;

  if (i < ha && j < wa) {
    a = x[i * incx];
    b = A[index];
  }

  __syncthreads();

  if (i < ha && j < wa) {
    A[index] = OP()(a, b);
  }
}


}

#endif // GPU_HPP
