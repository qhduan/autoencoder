#ifndef BLAS_H
#define BLAS_H
/*!
 * \file blas.h
 * \brief contain some BLAS functions
 * 		asum: sum a vector
 * 		dot: dot two vector
 * 		nrm2: normal^2 (L2) of a vector
 * 		gemm: matrix product, C = a*A*B + b*C
 *    I keep the functions of BLAS are non-statics because I want create
 *    cuBlas handle and destroy it
 * \author qhduan.com
 * \date 2014-05-13
 */

#include <cublas_v2.h>
#include "../inc/device.hpp"

namespace tensor {

/*!
 * \brief BLAS structure
 */
template<Device dev>
class BLAS {
private:
	/*! \brief new cuBlas need a handle */
	cublasHandle_t cublas_handle_;
  static BLAS* instance_;
public:
	/*!
	 * \brief create a BLAS class, GPU version need init a handle
	 */
	BLAS ();
	/*!
	 * \brief create a BLAS class, GPU version need destroy a handle
	 */
	~BLAS ();

  static BLAS& instance () {
    if (instance_ == NULL) {
      instance_ = new BLAS();
    }
    return *instance_;
  }

	/*!
	 * \brief sum a vector
	 * \param n size of vector
	 * \param x pointer to vector
	 * \param incx BLAS argument
	 * \return sum value
	 */
	template<typename T>
	T asum (int n, const T* x, int incx);

  /*!
   * \brief copy from x to y
   */
  template<typename T>
  void copy (int n, const T* x, int incx, T* y, int incy);

	/*!
	 * \brief dot product of two vector
	 * \param n size of vector
	 * \param x pointer to vector x
	 * \param incx BLAS argument
	 * \param y pointer to vector y
	 * \param incy BLAS argument
	 * \return dot value
	 */
	template<typename T>
	T dot (int n, const T* x, int incx, const T* y, int incy);

	/*!
	 * \brief normal2 of a vector
	 * \param n size of vector
	 * \param x pointer to vector
	 * \param incx BLAS argument
	 * \return normal2 value
	 */
	template<typename T>
	T nrm2 (int n, const T* x, int incx);

  /*!
   * \brief alpha*x*y + A -> A
   */
  template<typename T>
  void ger (int m, int n, T alpha, const T* x, int incx,
            const T* y, int incy, T *A, const int lda);

  /*!
   * \brief alpha*A*x + beta*y ->y
   */
  template<typename T>
  void gemv (bool transa, int m, int n, T alpha, const T* A, int lda,
    const T* x, int incx, T beta, T* y, int incy);

	/*!
	 * \brief C = alpha * A * B + beta * C
	 * \param transa tranpose A(true) or not(false)
	 * \param transb tranpose B(true) or not(false)
	 * \param m height of C, height of A
	 * \param n width of C, width of B
	 * \param k width of A, height of B
	 * \param alpha alpha value
	 * \param A pointer to matrix A
	 * \param lda lead A
	 * \param B pointer to matrix B
	 * \param ldb lead B
	 * \param beta beta value
	 * \param C pointer to matrix C
	 * \param ldc lead C
	 */
	template<typename T>
	void gemm (bool transa, bool transb, int m, int n, int k, T alpha,
		const T *A, int lda, const T *B, int ldb, T beta, T *C, int ldc );
};

}

#endif // BLAS_H
