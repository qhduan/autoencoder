#include "../inc/blas.h"

#include <stdexcept>
#include <iostream>
#include <string>

#include <cblas.h>

namespace tensor {

template<>
BLAS<CPU>* BLAS<CPU>::instance_ = NULL;

template<>
BLAS<GPU>* BLAS<GPU>::instance_ = NULL;

inline CBLAS_TRANSPOSE GetCblasTrans(bool t) {
	return t ? CblasTrans : CblasNoTrans;
}

inline cublasOperation_t GetCudablasTrans(bool t) {
	return t ? CUBLAS_OP_T : CUBLAS_OP_N;
}

// CPU

template<>
BLAS<CPU>::BLAS () {
}
template<>
BLAS<CPU>::~BLAS () {
}

// CPU float

template<> template<>
float BLAS<CPU>::asum<float> (int n, const float* x, int incx) {
	return cblas_sasum(n, x, incx);
}

template<> template<>
void BLAS<CPU>::copy<float> (int n, const float* x, int incx, float* y, int incy) {
	cblas_scopy(n, x, incx, y, incy);
}

template<> template<>
float BLAS<CPU>::dot<float> (int n, const float* x, int incx, const float* y, int incy) {
	return cblas_sdot(n, x, incx, y, incy);
}

template<> template<>
float BLAS<CPU>::nrm2<float> (int n, const float* x, int incx) {
	return cblas_snrm2(n, x, incx);
}

template<> template<>
void BLAS<CPU>::ger<float> (int m, int n, float alpha, const float* x, int incx,
  const float* y, int incy, float *A, const int lda) {
	cblas_sger(CblasColMajor, m, n, alpha, x, incx, y, incy, A, lda);
}

template<> template<>
void BLAS<CPU>::gemv<float> (bool transa, int m, int n, float alpha, const float* A, int lda,
  const float* x, int incx, float beta, float* y, int incy) {
  cblas_sgemv(CblasColMajor, GetCblasTrans(transa), m, n,
    alpha, A, lda, x, incx, beta, y ,incy);
}

template<> template<>
void BLAS<CPU>::gemm<float> (bool transa, bool transb, int m, int n, int k, float alpha,
	const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc ) {
	cblas_sgemm(CblasColMajor, GetCblasTrans(transa), GetCblasTrans(transb), m, n, k, alpha,
		A, lda, B, ldb, beta,C, ldc);
}

// CPU double

template<> template<>
double BLAS<CPU>::asum<double> (int n, const double* x, int incx) {
	return cblas_dasum(n, x, incx);
}

template<> template<>
void BLAS<CPU>::copy<double> (int n, const double* x, int incx, double* y, int incy) {
	cblas_dcopy(n, x, incx, y, incy);
}

template<> template<>
double BLAS<CPU>::dot<double> (int n, const double* x, int incx, const double* y, int incy) {
	return cblas_ddot(n, x, incx, y, incy);
}

template<> template<>
double BLAS<CPU>::nrm2<double> (int n, const double* x, int incx) {
	return cblas_dnrm2(n, x, incx);
}

template<> template<>
void BLAS<CPU>::ger<double> (int m, int n, double alpha, const double* x, int incx,
  const double* y, int incy, double *A, const int lda) {
	cblas_dger(CblasColMajor, m, n, alpha, x, incx, y, incy, A, lda);
}

template<> template<>
void BLAS<CPU>::gemv<double> (bool transa, int m, int n, double alpha, const double* A, int lda,
  const double* x, int incx, double beta, double* y, int incy) {
  cblas_dgemv(CblasColMajor, GetCblasTrans(transa), m, n,
    alpha, A, lda, x, incx, beta, y ,incy);
}

template<> template<>
void BLAS<CPU>::gemm<double> (bool transa, bool transb, int m, int n, int k, double alpha, \
	const double *A, int lda, const double *B, int ldb, double beta, double *C, int ldc ) {
	cblas_dgemm(CblasColMajor, GetCblasTrans(transa), GetCblasTrans(transb), m, n, k, alpha,\
		A, lda, B, ldb, beta, C, ldc);
}

// GPU

const char* cublasGetErrorString (cublasStatus_t err) {
  switch (err) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
		case CUBLAS_STATUS_NOT_SUPPORTED:
			return "CUBLAS_STATUS_NOT_SUPPORTED";
		case CUBLAS_STATUS_LICENSE_ERROR:
			return "CUBLAS_STATUS_LICENSE_ERROR";
  }
  return "UNKNOWN";
}

#define CUBLAS_CALL(str, x) cublasStatus_t err = (x); if (err != CUBLAS_STATUS_SUCCESS) { std::cerr<<"error: "<<cublasGetErrorString(err)<<std::endl; throw std::runtime_error(str); }

template<>
BLAS<GPU>::BLAS () {
	CUBLAS_CALL("cublasCreate", cublasCreate(&cublas_handle_));
}
template<>
BLAS<GPU>::~BLAS () {
	cublasDestroy(cublas_handle_);
}

// GPU float

template<> template<>
float BLAS<GPU>::asum<float> (int n, const float* x, int incx) {
	float result;
	CUBLAS_CALL("cublasSasum", cublasSasum(cublas_handle_, n, x, incx, &result));
  cudaDeviceSynchronize();
	return result;
}

template<> template<>
void BLAS<GPU>::copy<float> (int n, const float* x, int incx, float* y, int incy) {
	CUBLAS_CALL("cublasScopy", cublasScopy(cublas_handle_, n, x, incx, y, incy));
  cudaDeviceSynchronize();
}

template<> template<>
float BLAS<GPU>::dot<float> (int n, const float* x, int incx, const float* y, int incy) {
	float result;
	CUBLAS_CALL("cublasSdot", cublasSdot(cublas_handle_, n, x, incx, y, incy, &result));
  cudaDeviceSynchronize();
	return result;
}

template<> template<>
float BLAS<GPU>::nrm2<float> (int n, const float* x, int incx) {
	float result;
	CUBLAS_CALL("cublasSnrm2", cublasSnrm2(cublas_handle_, n, x, incx, &result));
  cudaDeviceSynchronize();
	return result;
}

template<> template<>
void BLAS<GPU>::ger<float> (int m, int n, float alpha, const float* x, int incx,
  const float* y, int incy, float *A, const int lda) {
	CUBLAS_CALL("cublasSger", cublasSger(cublas_handle_, m, n, &alpha, x, incx, y, incy, A, lda));
  cudaDeviceSynchronize();
}

template<> template<>
void BLAS<GPU>::gemv<float> (bool transa, int m, int n, float alpha, const float* A, int lda,
  const float* x, int incx, float beta, float* y, int incy) {
  CUBLAS_CALL("cublasSgemv", cublasSgemv(cublas_handle_, GetCudablasTrans(transa), m, n, &alpha, A, lda, x, incx, &beta, y ,incy));
  cudaDeviceSynchronize();
}

template<> template<>
void BLAS<GPU>::gemm<float> (bool transa, bool transb, int m, int n, int k, float alpha,
	const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc ) {
	CUBLAS_CALL("cublasSgemm", cublasSgemm(cublas_handle_, GetCudablasTrans(transa), GetCudablasTrans(transb), m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc));
  cudaDeviceSynchronize();
}

// GPU double

template<> template<>
double BLAS<GPU>::asum<double> (int n, const double* x, int incx) {
	double result;
	CUBLAS_CALL("cublasDasum", cublasDasum(cublas_handle_, n, x, incx, &result));
  cudaDeviceSynchronize();
	return result;
}

template<> template<>
void BLAS<GPU>::copy<double> (int n, const double* x, int incx, double* y, int incy) {
	CUBLAS_CALL("cublasDcopy", cublasDcopy(cublas_handle_, n, x, incx, y, incy));
  cudaDeviceSynchronize();
}

template<> template<>
double BLAS<GPU>::dot<double> (int n, const double* x, int incx, const double* y, int incy) {
	double result;
	CUBLAS_CALL("cublasDdot", cublasDdot(cublas_handle_, n, x, incx, y, incy, &result));
  cudaDeviceSynchronize();
	return result;
}

template<> template<>
double BLAS<GPU>::nrm2<double> (int n, const double* x, int incx) {
	double result;
	CUBLAS_CALL("cublasDnrm2", cublasDnrm2(cublas_handle_, n, x, incx, &result));
  cudaDeviceSynchronize();
	return result;
}

template<> template<>
void BLAS<GPU>::ger<double> (int m, int n, double alpha, const double* x, int incx,
  const double* y, int incy, double *A, const int lda) {
	CUBLAS_CALL("cublasDger", cublasDger(cublas_handle_, m, n, &alpha, x, incx, y, incy, A, lda));
  cudaDeviceSynchronize();
}

template<> template<>
void BLAS<GPU>::gemv<double> (bool transa, int m, int n, double alpha, const double* A, int lda,
  const double* x, int incx, double beta, double* y, int incy) {
  CUBLAS_CALL("cublasDgemv", cublasDgemv(cublas_handle_, GetCudablasTrans(transa), m, n, &alpha, A, lda, x, incx, &beta, y ,incy));
  cudaDeviceSynchronize();
}

template<> template<>
void BLAS<GPU>::gemm<double> (bool transa, bool transb, int m, int n, int k, double alpha,
	const double *A, int lda, const double *B, int ldb, double beta, double *C, int ldc ) {
	CUBLAS_CALL("cublasDgemm", cublasDgemm(cublas_handle_, GetCudablasTrans(transa), GetCudablasTrans(transb), m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc));
  cudaDeviceSynchronize();
}


}
