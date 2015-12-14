#ifndef KERNEL_HPP
#define KERNEL_HPP
/*!
 * \file vector_kernel.h
 * \brief define device
 * \author qhduan.com
 * \date 2015-10-27
*/



namespace tensor {


/*
 *
 *
 * // vector kernel
 *
 *
*/



template <Device dev, typename T, typename OP>
struct x_op_y_to_z {
  int n; const T* x; int incx; const T* y; int incy; T* z; int incz;
  __device__ void operator () (int i) {
    if (i < n) {
      z[i * incz] = OP()(x[i * incx], y[i * incy]);
    }
  }
};



template <Device dev, typename T, typename OP>
struct x_op_v_to_y {
  int n; const T* x; int incx; T value; T* y; int incy;
  __device__ void operator () (int i) {
    if (i < n) {
      y[i * incy] = OP()(x[i * incx], value);
    }
  }
};



template <Device dev, typename T, typename OP>
struct v_op_x_to_y {
  int n; T value; const T* x; int incx; T* y; int incy;
  __device__ void operator () (int i) {
    if (i < n) {
      y[i * incy] = OP()(value, x[i * incx]);
    }
  }
};



template <Device dev, typename T>
struct v_to_x {
  int n; T value; T* x; int incx;
  __host__ __device__ void operator () (int i) {
    if (i < n) {
     x[i * incx] = value;
    }
  }
};



// some function
template <Device dev, typename T, typename OP>
struct fn_x_to_y {
  int n; const T* x; int incx; T* y; int incy;
  __host__ __device__ void operator () (int i) {
    if (i < n) {
     y[i * incy] = OP()(x[i * incx]);
    }
  }
};


/*
 *
 *
 * // matrix kernel
 *
 *
*/


template <Device dev, typename T, typename OP>
struct A_op_B_to_C {
  int m; int n; const T* A; int lda; const T* B; int ldb; T* C; int ldc;
  __host__ __device__ void operator () (int i, int j) {
    if (i < m && j < n) {
      C[i + j * ldc] = OP()(A[i + j * lda], B[i + j * ldb]);
    }
  }
};



template <Device dev, typename T, typename OP>
struct A_op_v_to_B {
  int m; int n; const T* A; int lda; T value; T* B; int ldb;
  __host__ __device__ void operator () (int i, int j) {
    if (i < m && j < n) {
      B[i + j * ldb] = OP()(A[i + j * lda], value);
    }
  }
};



template <Device dev, typename T, typename OP>
struct v_op_A_to_B {
  int m; int n; T value; const T* A; int lda; T* B; int ldb;
  __host__ __device__ void operator () (int i, int j) {
    if (i < m && j < n) {
      B[i + j * ldb] = OP()(value, A[i + j * lda]);
    }
  }
};



template <Device dev, typename T>
struct v_to_A {
  int m; int n; T value; T* A; int lda;
  __host__ __device__ void operator () (int i, int j) {
    if (i < m && j < n) {
     A[i + j * lda] = value;
    }
  }
};



// some function
template <Device dev, typename T, typename OP>
struct fn_A_to_B {
  int m; int n; const T* A; int lda; T* B; int ldb;
  __host__ __device__ void operator () (int i, int j) {
    if (i < m && j < n) {
      B[i + j * ldb] = OP()(A[i + j * lda]);
    }
  }
};



/*
 * B = A'
*/
template <Device dev, typename T>
struct trans_A_to_B {
  int m; int n; const T* A; int lda; T* B; int ldb;
  __host__ __device__ void operator () (int i, int j) {
    if (i < m && j < n) {
      B[j + i * ldb] = A[i + j * lda];
    }
  }
};



/*
 * B = A.each_row() op x
*/
template <Device dev, typename T, typename OP>
struct A_each_row_op_x_to_B {
  int m; int n; const T* A; int lda; const T* x; int incx; T* B; int ldb;
  __host__ __device__ void operator () (int i, int j) {
    if (i < m && j < n) {
      B[i + j * ldb] = OP()(A[i + j * lda], x[j * incx]);
    }
  }
};



/*
 * B = x op A.each_row()
*/
template <Device dev, typename T, typename OP>
struct x_op_A_each_row_to_B {
  int m; int n; const T* x; int incx; const T* A; int lda; T* B; int ldb;
  __host__ __device__ void operator () (int i, int j) {
    if (i < m && j < n) {
      B[i + j * ldb] = OP()(x[j * incx], A[i + j * lda]);
    }
  }
};



/*
 * B = A.each_col() op x
*/
template <Device dev, typename T, typename OP>
struct A_each_col_op_x_to_B {
  int m; int n; const T* A; int lda; const T* x; int incx; T* B; int ldb;
  __host__ __device__ void operator () (int i, int j) {
    if (i < m && j < n) {
      B[i + j * ldb] = OP()(A[i + j * lda], x[i * incx]);
    }
  }
};



/*
 * B = x op A.each_col()
*/
template <Device dev, typename T, typename OP>
struct x_op_A_each_col_to_B {
  int m; int n; const T* x; int incx; const T* A; int lda; T* B; int ldb;
  __host__ __device__ void operator () (int i, int j) {
    if (i < m && j < n) {
      B[i + j * ldb] = OP()(x[i * incx], A[i + j * lda]);
    }
  }
};


}


#endif // KERNEL_HPP
