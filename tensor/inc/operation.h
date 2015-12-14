#ifndef OPERATION_H
#define OPERATION_H
/*!
 * \file operation.h
 * \brief contain all the math operator for vector and matrix
 * \author qhduan.com
 * \date 2014-08-08
 */

#include <iostream>
#include <vector>

#include "../inc/device.hpp"
#include "../inc/blas.h"

namespace tensor {



/*!
 * \brief A static class
 * 		contain all math operation
 */
template<Device dev>
class TensorOperation{
private:

	/*! \brief Blas object for some operation */
	static BLAS<dev> blas_;
public:

  /*
   *
   * BLAS
   *
   */

	template<typename T>
	static T asum (int n, const T* x, int incx);

  template<typename T>
  static void copy (int n, const T* x, int incx, T* y, int incy);

	template<typename T>
	static T dot (int n, const T* x, int incx, const T* y, int incy);

	template<typename T>
	static T nrm2 (int n, const T* x, int incx);

  template<typename T>
  static void ger (int m, int n, T alpha, const T* x, int incx,
    const T* y, int incy, T *A, const int lda);

  template<typename T>
  static void gemv (bool transa, int m, int n, T alpha, const T* A, int lda,
    const T* x, int incx, T beta, T* y, int incy);

	template<typename T>
	static void gemm (bool transa, bool transb, int m, int n, int k, T alpha,
		const T *A, int lda, const T *B, int ldb, T beta, T *C, int ldc );


  /*
   *
   * Other
   *
   */


 	template <typename OP>
 	static void matrix_call (OP op);

	template <typename OP>
	static void vector_call (OP op);

	/*! \brief get an element from pointer */
	template<typename T>
	static T get (const T* x);

	/*! \brief set a value to a pointer */
	template<typename T>
	static void set (T* x, T val);

	/*! \brief x -> y */
	template<typename T>
	static void x_to_y (int n, const T* x, int incx, T* y, int incy);

	/*! \brief reduce vector */
	template <typename T, typename OP>
	static T x_reduce (int n, const T* x, int incx);

	/*! \brief reduce matrix */
	template <typename T, typename OP>
	static T A_reduce (int m, int n, const T* A, int lda);

	template <typename T, typename OP>
	static void A_each_row_reduce_to_x (int m, int n, const T* A, int lda, T* x, int incx);

	template <typename T, typename OP>
	static void A_each_col_reduce_to_x (int m, int n, const T* A, int lda, T* x, int incx);

	/*! \brief A -> B */
	template<typename T>
	static void A_to_B (int m, int n, const T* A, int lda, T* B, int ldb, Device dest = dev);

  /*! \brief compute (row of A) .op x -> A */
  template<typename T, typename OP>
  static void A_rows_op_x_to_A (T* A, int ha, int wa, int lda, const T* x, int incx);

  /*! \brief compute x .op (row of A) -> A */
  template<typename T, typename OP>
  static void x_op_A_rows_to_A (const T* x , int incx, T* A, int ha, int wa, int lda);

  /*! \brief compute (col of A) .op x -> A */
  template<typename T, typename OP>
  static void A_cols_op_x_to_A (T* A, int ha, int wa, int lda, const T* x, int incx);

  /*! \brief compute x .op (col of A) -> A */
  template<typename T, typename OP>
  static void x_op_A_cols_to_A (const T* x, int incx, T* A, int ha, int wa, int lda);


};


}

#endif // OPERATION_H
