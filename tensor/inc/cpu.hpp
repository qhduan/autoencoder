#ifndef CPU_HPP
#define CPU_HPP
/*!
 * \file cpu.h
 * \brief some CPU operation funcions
 * 		2D convolution
 * 		set a row of matrix
 * 		matrix transpose
 * \author qhduan.com
 * \date 2014-08-08
 */

#include <iostream>
#include <algorithm>

#include "../inc/device.hpp"

namespace tensor {


template <typename OP>
void cpu_vector_call (OP op) {
  for (int i = 0; i < op.n; i++) {
    op(i);
  }
}


template <typename OP>
void cpu_matrix_call (OP op) {
  for (int i = 0; i < op.m; i++) {
    for (int j = 0; j < op.n; j++) {
      op(i, j);
    }
  }
}


template <typename T, typename OP>
void cpu_x_reduce (int n, const T* x, int incx, T* result) {
  for (int i = 0; i < n; i++) {
    if (i == 0) {
      *result = OP().pre(x[i * incx]);
    } else {
      *result = OP()(*result, OP().pre(x[i * incx]));
    }
  }
}


template <typename T, typename OP>
void cpu_A_each_row_reduce_to_x (int m, int n, const T* A, int lda, T* x, int incx) {
  for (int i = 0; i < m; i++) {
    T ret = 0.0;
    for (int j = 0; j < n; j++) {
      if (j == 0) {
        ret = OP().pre(A[i + j * lda]);
      } else {
        ret = OP()(ret, OP().pre(A[i + j * lda]));
      }
    }
    x[i * incx] = ret;
  }
}


template <typename T, typename OP>
void cpu_A_each_col_reduce_to_x (int m, int n, const T* A, int lda, T* x, int incx) {
  for (int j = 0; j < n; j++) {
    T ret = 0.0;
    for (int i = 0; i < m; i++) {
      if (i == 0) {
        ret = OP().pre(A[i + j * lda]);
      } else {
        ret = OP()(ret, OP().pre(A[i + j * lda]));
      }
    }
    x[j * incx] = ret;
  }
}


template <typename T, typename OP>
void cpu_A_reduce (int m, int n, const T* A, int lda, T* result) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      if (i == 0 && j == 0) {
        *result = OP().pre(A[i + j * lda]);
      } else {
        *result = OP()(*result, OP().pre(A[i + j * lda]));
      }
    }
  }
}


template <typename T>
void cpu_A_conv2_B_to_C (const T* A, int ha, int wa, int lda,
  const T* B, int hb, int wb, int ldb, T* C, int hc, int wc, int ldc) {

	for (int i = 0; i < hc; i++) {

		for (int j = 0; j < wc; j++) {
			T value = 0;

			for (int x = 0; x < hb; x++){

				for (int y = 0; y < wb; y++) {
					int m = i + x;
					int n = j + y;
					value += A[m + lda * n] * B[x + ldb * y];
				}
			}
			C[i + ldc * j] = value;
		}
	}
}


template <typename T>
void cpu_A_tran_to_B (const T* A, int ha, int wa, int lda, T* B, int hb, int wb, int ldb) {

	for (int i = 0; i < ha; i++) {
		for (int j = 0; j < wa; j++) {
			B[i * ldb + j] = A[i + j * lda];
		}
	}
}


template <typename T, typename OP>
void cpu_A_rows_op_x_to_A (T* A, int ha, int wa, int lda, const T* x, int incx) {

  for (int i = 0; i < ha;i++) {
    for (int j = 0; j < wa; j++) {
      A[i + j * lda] = OP()(A[i + j * lda], x[j * incx]);
    }
  }
}


template <typename T, typename OP>
void cpu_x_op_A_rows_to_A (const T* x , int incx, T* A, int ha, int wa, int lda) {

  for (int i = 0; i < ha;i++) {

    for (int j = 0; j < wa; j++) {
      A[i + j * lda] = OP()(x[j * incx], A[i + j * lda]);
    }
  }
}


template <typename T, typename OP>
void cpu_A_cols_op_x_to_A (T* A, int ha, int wa, int lda, const T* x, int incx) {

  for (int i = 0; i < ha;i++) {

    for (int j = 0; j < wa; j++) {
      A[i + j * lda] = OP()(A[i + j * lda], x[i * incx]);
    }
  }
}


template <typename T, typename OP>
void cpu_x_op_A_cols_to_A (const T* x, int incx, T* A, int ha, int wa, int lda) {

  for (int i = 0; i < ha;i++) {

    for (int j = 0; j < wa; j++) {
      A[i + j * lda] = OP()(x[i * incx], A[i + j * lda]);
    }
  }
}


}

#endif // CPU_HPP
