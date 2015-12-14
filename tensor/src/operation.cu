#include "../inc/operation.h"

#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <fstream>
#include <cstring>


#define BLOCK_SIZE 32
#define BLOCK_NUM_THREADS 1024

#include "../inc/cpu.hpp"
#include "../inc/gpu.hpp"

namespace tensor {

template<Device dev>
BLAS<dev> TensorOperation<dev>::blas_;

/*
 *
 * CPU DEVICE
 *
*/


template <> template <typename OP>
void TensorOperation<CPU>::vector_call (OP op) {
  cpu_vector_call(op);
}

template void TensorOperation<CPU>::vector_call<>(v_to_x<CPU, float>);
template void TensorOperation<CPU>::vector_call<>(v_to_x<CPU, double>);


template void TensorOperation<CPU>::vector_call<>(x_op_y_to_z<CPU, float, functor::Plus<CPU, float> >);
template void TensorOperation<CPU>::vector_call<>(x_op_y_to_z<CPU, double, functor::Plus<CPU, double> >);
template void TensorOperation<CPU>::vector_call<>(x_op_y_to_z<CPU, float, functor::Minus<CPU, float> >);
template void TensorOperation<CPU>::vector_call<>(x_op_y_to_z<CPU, double, functor::Minus<CPU, double> >);
template void TensorOperation<CPU>::vector_call<>(x_op_y_to_z<CPU, float, functor::Multiplies<CPU, float> >);
template void TensorOperation<CPU>::vector_call<>(x_op_y_to_z<CPU, double, functor::Multiplies<CPU, double> >);
template void TensorOperation<CPU>::vector_call<>(x_op_y_to_z<CPU, float, functor::Divides<CPU, float> >);
template void TensorOperation<CPU>::vector_call<>(x_op_y_to_z<CPU, double, functor::Divides<CPU, double> >);
template void TensorOperation<CPU>::vector_call<>(x_op_y_to_z<CPU, float, functor::Pow<CPU, float> >);
template void TensorOperation<CPU>::vector_call<>(x_op_y_to_z<CPU, double, functor::Pow<CPU, double> >);


template void TensorOperation<CPU>::vector_call<>(x_op_v_to_y<CPU, float, functor::Plus<CPU, float> >);
template void TensorOperation<CPU>::vector_call<>(x_op_v_to_y<CPU, double, functor::Plus<CPU, double> >);
template void TensorOperation<CPU>::vector_call<>(x_op_v_to_y<CPU, float, functor::Minus<CPU, float> >);
template void TensorOperation<CPU>::vector_call<>(x_op_v_to_y<CPU, double, functor::Minus<CPU, double> >);
template void TensorOperation<CPU>::vector_call<>(x_op_v_to_y<CPU, float, functor::Multiplies<CPU, float> >);
template void TensorOperation<CPU>::vector_call<>(x_op_v_to_y<CPU, double, functor::Multiplies<CPU, double> >);
template void TensorOperation<CPU>::vector_call<>(x_op_v_to_y<CPU, float, functor::Divides<CPU, float> >);
template void TensorOperation<CPU>::vector_call<>(x_op_v_to_y<CPU, double, functor::Divides<CPU, double> >);
template void TensorOperation<CPU>::vector_call<>(x_op_v_to_y<CPU, float, functor::Pow<CPU, float> >);
template void TensorOperation<CPU>::vector_call<>(x_op_v_to_y<CPU, double, functor::Pow<CPU, double> >);


template void TensorOperation<CPU>::vector_call<>(v_op_x_to_y<CPU, float, functor::Plus<CPU, float> >);
template void TensorOperation<CPU>::vector_call<>(v_op_x_to_y<CPU, double, functor::Plus<CPU, double> >);
template void TensorOperation<CPU>::vector_call<>(v_op_x_to_y<CPU, float, functor::Minus<CPU, float> >);
template void TensorOperation<CPU>::vector_call<>(v_op_x_to_y<CPU, double, functor::Minus<CPU, double> >);
template void TensorOperation<CPU>::vector_call<>(v_op_x_to_y<CPU, float, functor::Multiplies<CPU, float> >);
template void TensorOperation<CPU>::vector_call<>(v_op_x_to_y<CPU, double, functor::Multiplies<CPU, double> >);
template void TensorOperation<CPU>::vector_call<>(v_op_x_to_y<CPU, float, functor::Divides<CPU, float> >);
template void TensorOperation<CPU>::vector_call<>(v_op_x_to_y<CPU, double, functor::Divides<CPU, double> >);
template void TensorOperation<CPU>::vector_call<>(v_op_x_to_y<CPU, float, functor::Pow<CPU, float> >);
template void TensorOperation<CPU>::vector_call<>(v_op_x_to_y<CPU, double, functor::Pow<CPU, double> >);


template void TensorOperation<CPU>::vector_call<>(fn_x_to_y<CPU, float, functor::Log<CPU, float> >);
template void TensorOperation<CPU>::vector_call<>(fn_x_to_y<CPU, double, functor::Log<CPU, double> >);
template void TensorOperation<CPU>::vector_call<>(fn_x_to_y<CPU, float, functor::Exp<CPU, float> >);
template void TensorOperation<CPU>::vector_call<>(fn_x_to_y<CPU, double, functor::Exp<CPU, double> >);
template void TensorOperation<CPU>::vector_call<>(fn_x_to_y<CPU, float, functor::Abs<CPU, float> >);
template void TensorOperation<CPU>::vector_call<>(fn_x_to_y<CPU, double, functor::Abs<CPU, double> >);
template void TensorOperation<CPU>::vector_call<>(fn_x_to_y<CPU, float, functor::Sign<CPU, float> >);
template void TensorOperation<CPU>::vector_call<>(fn_x_to_y<CPU, double, functor::Sign<CPU, double> >);
template void TensorOperation<CPU>::vector_call<>(fn_x_to_y<CPU, float, functor::Sigmoid<CPU, float> >);
template void TensorOperation<CPU>::vector_call<>(fn_x_to_y<CPU, double, functor::Sigmoid<CPU, double> >);
template void TensorOperation<CPU>::vector_call<>(fn_x_to_y<CPU, float, functor::Dsigmoid<CPU, float> >);
template void TensorOperation<CPU>::vector_call<>(fn_x_to_y<CPU, double, functor::Dsigmoid<CPU, double> >);
template void TensorOperation<CPU>::vector_call<>(fn_x_to_y<CPU, float, functor::Tanh<CPU, float> >);
template void TensorOperation<CPU>::vector_call<>(fn_x_to_y<CPU, double, functor::Tanh<CPU, double> >);
template void TensorOperation<CPU>::vector_call<>(fn_x_to_y<CPU, float, functor::Dtanh<CPU, float> >);
template void TensorOperation<CPU>::vector_call<>(fn_x_to_y<CPU, double, functor::Dtanh<CPU, double> >);




template <> template <typename OP>
void TensorOperation<CPU>::matrix_call (OP op) {
  cpu_matrix_call(op);
}

template void TensorOperation<CPU>::matrix_call<>(v_to_A<CPU, float>);
template void TensorOperation<CPU>::matrix_call<>(v_to_A<CPU, double>);


template void TensorOperation<CPU>::matrix_call<>(A_op_B_to_C<CPU, float, functor::Plus<CPU, float> >);
template void TensorOperation<CPU>::matrix_call<>(A_op_B_to_C<CPU, double, functor::Plus<CPU, double> >);
template void TensorOperation<CPU>::matrix_call<>(A_op_B_to_C<CPU, float, functor::Minus<CPU, float> >);
template void TensorOperation<CPU>::matrix_call<>(A_op_B_to_C<CPU, double, functor::Minus<CPU, double> >);
template void TensorOperation<CPU>::matrix_call<>(A_op_B_to_C<CPU, float, functor::Multiplies<CPU, float> >);
template void TensorOperation<CPU>::matrix_call<>(A_op_B_to_C<CPU, double, functor::Multiplies<CPU, double> >);
template void TensorOperation<CPU>::matrix_call<>(A_op_B_to_C<CPU, float, functor::Divides<CPU, float> >);
template void TensorOperation<CPU>::matrix_call<>(A_op_B_to_C<CPU, double, functor::Divides<CPU, double> >);
template void TensorOperation<CPU>::matrix_call<>(A_op_B_to_C<CPU, float, functor::Pow<CPU, float> >);
template void TensorOperation<CPU>::matrix_call<>(A_op_B_to_C<CPU, double, functor::Pow<CPU, double> >);


template void TensorOperation<CPU>::matrix_call<>(A_op_v_to_B<CPU, float, functor::Plus<CPU, float> >);
template void TensorOperation<CPU>::matrix_call<>(A_op_v_to_B<CPU, double, functor::Plus<CPU, double> >);
template void TensorOperation<CPU>::matrix_call<>(A_op_v_to_B<CPU, float, functor::Minus<CPU, float> >);
template void TensorOperation<CPU>::matrix_call<>(A_op_v_to_B<CPU, double, functor::Minus<CPU, double> >);
template void TensorOperation<CPU>::matrix_call<>(A_op_v_to_B<CPU, float, functor::Multiplies<CPU, float> >);
template void TensorOperation<CPU>::matrix_call<>(A_op_v_to_B<CPU, double, functor::Multiplies<CPU, double> >);
template void TensorOperation<CPU>::matrix_call<>(A_op_v_to_B<CPU, float, functor::Divides<CPU, float> >);
template void TensorOperation<CPU>::matrix_call<>(A_op_v_to_B<CPU, double, functor::Divides<CPU, double> >);
template void TensorOperation<CPU>::matrix_call<>(A_op_v_to_B<CPU, float, functor::Pow<CPU, float> >);
template void TensorOperation<CPU>::matrix_call<>(A_op_v_to_B<CPU, double, functor::Pow<CPU, double> >);


template void TensorOperation<CPU>::matrix_call<>(v_op_A_to_B<CPU, float, functor::Plus<CPU, float> >);
template void TensorOperation<CPU>::matrix_call<>(v_op_A_to_B<CPU, double, functor::Plus<CPU, double> >);
template void TensorOperation<CPU>::matrix_call<>(v_op_A_to_B<CPU, float, functor::Minus<CPU, float> >);
template void TensorOperation<CPU>::matrix_call<>(v_op_A_to_B<CPU, double, functor::Minus<CPU, double> >);
template void TensorOperation<CPU>::matrix_call<>(v_op_A_to_B<CPU, float, functor::Multiplies<CPU, float> >);
template void TensorOperation<CPU>::matrix_call<>(v_op_A_to_B<CPU, double, functor::Multiplies<CPU, double> >);
template void TensorOperation<CPU>::matrix_call<>(v_op_A_to_B<CPU, float, functor::Divides<CPU, float> >);
template void TensorOperation<CPU>::matrix_call<>(v_op_A_to_B<CPU, double, functor::Divides<CPU, double> >);
template void TensorOperation<CPU>::matrix_call<>(v_op_A_to_B<CPU, float, functor::Pow<CPU, float> >);
template void TensorOperation<CPU>::matrix_call<>(v_op_A_to_B<CPU, double, functor::Pow<CPU, double> >);


template void TensorOperation<CPU>::matrix_call<>(A_each_row_op_x_to_B<CPU, float, functor::Plus<CPU, float> >);
template void TensorOperation<CPU>::matrix_call<>(A_each_row_op_x_to_B<CPU, double, functor::Plus<CPU, double> >);
template void TensorOperation<CPU>::matrix_call<>(A_each_row_op_x_to_B<CPU, float, functor::Minus<CPU, float> >);
template void TensorOperation<CPU>::matrix_call<>(A_each_row_op_x_to_B<CPU, double, functor::Minus<CPU, double> >);
template void TensorOperation<CPU>::matrix_call<>(A_each_row_op_x_to_B<CPU, float, functor::Multiplies<CPU, float> >);
template void TensorOperation<CPU>::matrix_call<>(A_each_row_op_x_to_B<CPU, double, functor::Multiplies<CPU, double> >);
template void TensorOperation<CPU>::matrix_call<>(A_each_row_op_x_to_B<CPU, float, functor::Divides<CPU, float> >);
template void TensorOperation<CPU>::matrix_call<>(A_each_row_op_x_to_B<CPU, double, functor::Divides<CPU, double> >);
template void TensorOperation<CPU>::matrix_call<>(A_each_row_op_x_to_B<CPU, float, functor::Pow<CPU, float> >);
template void TensorOperation<CPU>::matrix_call<>(A_each_row_op_x_to_B<CPU, double, functor::Pow<CPU, double> >);
template void TensorOperation<CPU>::matrix_call<>(A_each_row_op_x_to_B<CPU, float, functor::Left<CPU, float> >);
template void TensorOperation<CPU>::matrix_call<>(A_each_row_op_x_to_B<CPU, double, functor::Left<CPU, double> >);
template void TensorOperation<CPU>::matrix_call<>(A_each_row_op_x_to_B<CPU, float, functor::Right<CPU, float> >);
template void TensorOperation<CPU>::matrix_call<>(A_each_row_op_x_to_B<CPU, double, functor::Right<CPU, double> >);


template void TensorOperation<CPU>::matrix_call<>(x_op_A_each_row_to_B<CPU, float, functor::Plus<CPU, float> >);
template void TensorOperation<CPU>::matrix_call<>(x_op_A_each_row_to_B<CPU, double, functor::Plus<CPU, double> >);
template void TensorOperation<CPU>::matrix_call<>(x_op_A_each_row_to_B<CPU, float, functor::Minus<CPU, float> >);
template void TensorOperation<CPU>::matrix_call<>(x_op_A_each_row_to_B<CPU, double, functor::Minus<CPU, double> >);
template void TensorOperation<CPU>::matrix_call<>(x_op_A_each_row_to_B<CPU, float, functor::Multiplies<CPU, float> >);
template void TensorOperation<CPU>::matrix_call<>(x_op_A_each_row_to_B<CPU, double, functor::Multiplies<CPU, double> >);
template void TensorOperation<CPU>::matrix_call<>(x_op_A_each_row_to_B<CPU, float, functor::Divides<CPU, float> >);
template void TensorOperation<CPU>::matrix_call<>(x_op_A_each_row_to_B<CPU, double, functor::Divides<CPU, double> >);
template void TensorOperation<CPU>::matrix_call<>(x_op_A_each_row_to_B<CPU, float, functor::Pow<CPU, float> >);
template void TensorOperation<CPU>::matrix_call<>(x_op_A_each_row_to_B<CPU, double, functor::Pow<CPU, double> >);
template void TensorOperation<CPU>::matrix_call<>(x_op_A_each_row_to_B<CPU, float, functor::Left<CPU, float> >);
template void TensorOperation<CPU>::matrix_call<>(x_op_A_each_row_to_B<CPU, double, functor::Left<CPU, double> >);
template void TensorOperation<CPU>::matrix_call<>(x_op_A_each_row_to_B<CPU, float, functor::Right<CPU, float> >);
template void TensorOperation<CPU>::matrix_call<>(x_op_A_each_row_to_B<CPU, double, functor::Right<CPU, double> >);


template void TensorOperation<CPU>::matrix_call<>(A_each_col_op_x_to_B<CPU, float, functor::Plus<CPU, float> >);
template void TensorOperation<CPU>::matrix_call<>(A_each_col_op_x_to_B<CPU, double, functor::Plus<CPU, double> >);
template void TensorOperation<CPU>::matrix_call<>(A_each_col_op_x_to_B<CPU, float, functor::Minus<CPU, float> >);
template void TensorOperation<CPU>::matrix_call<>(A_each_col_op_x_to_B<CPU, double, functor::Minus<CPU, double> >);
template void TensorOperation<CPU>::matrix_call<>(A_each_col_op_x_to_B<CPU, float, functor::Multiplies<CPU, float> >);
template void TensorOperation<CPU>::matrix_call<>(A_each_col_op_x_to_B<CPU, double, functor::Multiplies<CPU, double> >);
template void TensorOperation<CPU>::matrix_call<>(A_each_col_op_x_to_B<CPU, float, functor::Divides<CPU, float> >);
template void TensorOperation<CPU>::matrix_call<>(A_each_col_op_x_to_B<CPU, double, functor::Divides<CPU, double> >);
template void TensorOperation<CPU>::matrix_call<>(A_each_col_op_x_to_B<CPU, float, functor::Pow<CPU, float> >);
template void TensorOperation<CPU>::matrix_call<>(A_each_col_op_x_to_B<CPU, double, functor::Pow<CPU, double> >);
template void TensorOperation<CPU>::matrix_call<>(A_each_col_op_x_to_B<CPU, float, functor::Left<CPU, float> >);
template void TensorOperation<CPU>::matrix_call<>(A_each_col_op_x_to_B<CPU, double, functor::Left<CPU, double> >);
template void TensorOperation<CPU>::matrix_call<>(A_each_col_op_x_to_B<CPU, float, functor::Right<CPU, float> >);
template void TensorOperation<CPU>::matrix_call<>(A_each_col_op_x_to_B<CPU, double, functor::Right<CPU, double> >);


template void TensorOperation<CPU>::matrix_call<>(x_op_A_each_col_to_B<CPU, float, functor::Plus<CPU, float> >);
template void TensorOperation<CPU>::matrix_call<>(x_op_A_each_col_to_B<CPU, double, functor::Plus<CPU, double> >);
template void TensorOperation<CPU>::matrix_call<>(x_op_A_each_col_to_B<CPU, float, functor::Minus<CPU, float> >);
template void TensorOperation<CPU>::matrix_call<>(x_op_A_each_col_to_B<CPU, double, functor::Minus<CPU, double> >);
template void TensorOperation<CPU>::matrix_call<>(x_op_A_each_col_to_B<CPU, float, functor::Multiplies<CPU, float> >);
template void TensorOperation<CPU>::matrix_call<>(x_op_A_each_col_to_B<CPU, double, functor::Multiplies<CPU, double> >);
template void TensorOperation<CPU>::matrix_call<>(x_op_A_each_col_to_B<CPU, float, functor::Divides<CPU, float> >);
template void TensorOperation<CPU>::matrix_call<>(x_op_A_each_col_to_B<CPU, double, functor::Divides<CPU, double> >);
template void TensorOperation<CPU>::matrix_call<>(x_op_A_each_col_to_B<CPU, float, functor::Pow<CPU, float> >);
template void TensorOperation<CPU>::matrix_call<>(x_op_A_each_col_to_B<CPU, double, functor::Pow<CPU, double> >);
template void TensorOperation<CPU>::matrix_call<>(x_op_A_each_col_to_B<CPU, float, functor::Left<CPU, float> >);
template void TensorOperation<CPU>::matrix_call<>(x_op_A_each_col_to_B<CPU, double, functor::Left<CPU, double> >);
template void TensorOperation<CPU>::matrix_call<>(x_op_A_each_col_to_B<CPU, float, functor::Right<CPU, float> >);
template void TensorOperation<CPU>::matrix_call<>(x_op_A_each_col_to_B<CPU, double, functor::Right<CPU, double> >);


template void TensorOperation<CPU>::matrix_call<>(fn_A_to_B<CPU, float, functor::Log<CPU, float> >);
template void TensorOperation<CPU>::matrix_call<>(fn_A_to_B<CPU, double, functor::Log<CPU, double> >);
template void TensorOperation<CPU>::matrix_call<>(fn_A_to_B<CPU, float, functor::Exp<CPU, float> >);
template void TensorOperation<CPU>::matrix_call<>(fn_A_to_B<CPU, double, functor::Exp<CPU, double> >);
template void TensorOperation<CPU>::matrix_call<>(fn_A_to_B<CPU, float, functor::Abs<CPU, float> >);
template void TensorOperation<CPU>::matrix_call<>(fn_A_to_B<CPU, double, functor::Abs<CPU, double> >);
template void TensorOperation<CPU>::matrix_call<>(fn_A_to_B<CPU, float, functor::Sign<CPU, float> >);
template void TensorOperation<CPU>::matrix_call<>(fn_A_to_B<CPU, double, functor::Sign<CPU, double> >);
template void TensorOperation<CPU>::matrix_call<>(fn_A_to_B<CPU, float, functor::Sigmoid<CPU, float> >);
template void TensorOperation<CPU>::matrix_call<>(fn_A_to_B<CPU, double, functor::Sigmoid<CPU, double> >);
template void TensorOperation<CPU>::matrix_call<>(fn_A_to_B<CPU, float, functor::Dsigmoid<CPU, float> >);
template void TensorOperation<CPU>::matrix_call<>(fn_A_to_B<CPU, double, functor::Dsigmoid<CPU, double> >);
template void TensorOperation<CPU>::matrix_call<>(fn_A_to_B<CPU, float, functor::Tanh<CPU, float> >);
template void TensorOperation<CPU>::matrix_call<>(fn_A_to_B<CPU, double, functor::Tanh<CPU, double> >);
template void TensorOperation<CPU>::matrix_call<>(fn_A_to_B<CPU, float, functor::Dtanh<CPU, float> >);
template void TensorOperation<CPU>::matrix_call<>(fn_A_to_B<CPU, double, functor::Dtanh<CPU, double> >);


template void TensorOperation<CPU>::matrix_call<>(trans_A_to_B<CPU, float>);
template void TensorOperation<CPU>::matrix_call<>(trans_A_to_B<CPU, double>);




template <> template <typename T, typename OP>
T TensorOperation<CPU>::x_reduce (int n, const T* x, int incx) {
  T ret = 0.0;
  cpu_x_reduce<T, OP>(n, x, incx, &ret);
  return ret;
}

template float TensorOperation<CPU>::x_reduce<float, functor::reduce_sum<CPU,float> >(int,const float*,int);
template double TensorOperation<CPU>::x_reduce<double, functor::reduce_sum<CPU,double> >(int,const double*,int);

template float TensorOperation<CPU>::x_reduce<float, functor::reduce_max<CPU,float> >(int,const float*,int);
template double TensorOperation<CPU>::x_reduce<double, functor::reduce_max<CPU,double> >(int,const double*,int);

template float TensorOperation<CPU>::x_reduce<float, functor::reduce_min<CPU,float> >(int,const float*,int);
template double TensorOperation<CPU>::x_reduce<double, functor::reduce_min<CPU,double> >(int,const double*,int);

template float TensorOperation<CPU>::x_reduce<float, functor::reduce_sum_of_abs<CPU,float> >(int,const float*,int);
template double TensorOperation<CPU>::x_reduce<double, functor::reduce_sum_of_abs<CPU,double> >(int,const double*,int);

template float TensorOperation<CPU>::x_reduce<float, functor::reduce_sum_of_squared<CPU,float> >(int,const float*,int);
template double TensorOperation<CPU>::x_reduce<double, functor::reduce_sum_of_squared<CPU,double> >(int,const double*,int);




template <> template <typename T, typename OP>
T TensorOperation<CPU>::A_reduce (int m, int n, const T* A, int lda) {
  T ret = 0.0;
  cpu_A_reduce<T, OP>(m, n, A, lda, &ret);
  return ret;
}

template float TensorOperation<CPU>::A_reduce<float, functor::reduce_sum<CPU,float> >(int,int,const float*,int);
template double TensorOperation<CPU>::A_reduce<double, functor::reduce_sum<CPU,double> >(int,int,const double*,int);

template float TensorOperation<CPU>::A_reduce<float, functor::reduce_max<CPU,float> >(int,int,const float*,int);
template double TensorOperation<CPU>::A_reduce<double, functor::reduce_max<CPU,double> >(int,int,const double*,int);

template float TensorOperation<CPU>::A_reduce<float, functor::reduce_min<CPU,float> >(int,int,const float*,int);
template double TensorOperation<CPU>::A_reduce<double, functor::reduce_min<CPU,double> >(int,int,const double*,int);

template float TensorOperation<CPU>::A_reduce<float, functor::reduce_sum_of_abs<CPU,float> >(int,int,const float*,int);
template double TensorOperation<CPU>::A_reduce<double, functor::reduce_sum_of_abs<CPU,double> >(int,int,const double*,int);

template float TensorOperation<CPU>::A_reduce<float, functor::reduce_sum_of_squared<CPU,float> >(int,int,const float*,int);
template double TensorOperation<CPU>::A_reduce<double, functor::reduce_sum_of_squared<CPU,double> >(int,int,const double*,int);




template <> template <typename T, typename OP>
void TensorOperation<CPU>::A_each_row_reduce_to_x (int m, int n, const T* A, int lda, T* x, int incx) {
  cpu_A_each_row_reduce_to_x<T, OP>(m, n, A, lda, x, incx);
}

template void TensorOperation<CPU>::A_each_row_reduce_to_x<float, functor::reduce_sum<CPU,float> >(int,int,const float*,int,float*,int);
template void TensorOperation<CPU>::A_each_row_reduce_to_x<double, functor::reduce_sum<CPU,double> >(int,int,const double*,int,double*,int);

template void TensorOperation<CPU>::A_each_row_reduce_to_x<float, functor::reduce_max<CPU,float> >(int,int,const float*,int,float*,int);
template void TensorOperation<CPU>::A_each_row_reduce_to_x<double, functor::reduce_max<CPU,double> >(int,int,const double*,int,double*,int);

template void TensorOperation<CPU>::A_each_row_reduce_to_x<float, functor::reduce_min<CPU,float> >(int,int,const float*,int,float*,int);
template void TensorOperation<CPU>::A_each_row_reduce_to_x<double, functor::reduce_min<CPU,double> >(int,int,const double*,int,double*,int);

template void TensorOperation<CPU>::A_each_row_reduce_to_x<float, functor::reduce_sum_of_abs<CPU,float> >(int,int,const float*,int,float*,int);
template void TensorOperation<CPU>::A_each_row_reduce_to_x<double, functor::reduce_sum_of_abs<CPU,double> >(int,int,const double*,int,double*,int);

template void TensorOperation<CPU>::A_each_row_reduce_to_x<float, functor::reduce_sum_of_squared<CPU,float> >(int,int,const float*,int,float*,int);
template void TensorOperation<CPU>::A_each_row_reduce_to_x<double, functor::reduce_sum_of_squared<CPU,double> >(int,int,const double*,int,double*,int);




template <> template <typename T, typename OP>
void TensorOperation<CPU>::A_each_col_reduce_to_x (int m, int n, const T* A, int lda, T* x, int incx) {
  cpu_A_each_col_reduce_to_x<T, OP>(m, n, A, lda, x, incx);
}

template void TensorOperation<CPU>::A_each_col_reduce_to_x<float, functor::reduce_sum<CPU,float> >(int,int,const float*,int,float*,int);
template void TensorOperation<CPU>::A_each_col_reduce_to_x<double, functor::reduce_sum<CPU,double> >(int,int,const double*,int,double*,int);

template void TensorOperation<CPU>::A_each_col_reduce_to_x<float, functor::reduce_max<CPU,float> >(int,int,const float*,int,float*,int);
template void TensorOperation<CPU>::A_each_col_reduce_to_x<double, functor::reduce_max<CPU,double> >(int,int,const double*,int,double*,int);

template void TensorOperation<CPU>::A_each_col_reduce_to_x<float, functor::reduce_min<CPU,float> >(int,int,const float*,int,float*,int);
template void TensorOperation<CPU>::A_each_col_reduce_to_x<double, functor::reduce_min<CPU,double> >(int,int,const double*,int,double*,int);

template void TensorOperation<CPU>::A_each_col_reduce_to_x<float, functor::reduce_sum_of_abs<CPU,float> >(int,int,const float*,int,float*,int);
template void TensorOperation<CPU>::A_each_col_reduce_to_x<double, functor::reduce_sum_of_abs<CPU,double> >(int,int,const double*,int,double*,int);

template void TensorOperation<CPU>::A_each_col_reduce_to_x<float, functor::reduce_sum_of_squared<CPU,float> >(int,int,const float*,int,float*,int);
template void TensorOperation<CPU>::A_each_col_reduce_to_x<double, functor::reduce_sum_of_squared<CPU,double> >(int,int,const double*,int,double*,int);




template <> template <typename T>
T TensorOperation<CPU>::asum (int n, const T* x, int incx) {
  return BLAS<CPU>::instance().asum(n, x, incx);
}
template float TensorOperation<CPU>::asum(int,const float*,int);
template double TensorOperation<CPU>::asum(int,const double*,int);

template <> template <typename T>
void TensorOperation<CPU>::copy (int n, const T* x, int incx, T* y, int incy) {
  BLAS<CPU>::instance().copy(n, x, incx, y, incy);
}
template void TensorOperation<CPU>::copy(int,const float*,int,float*,int);
template void TensorOperation<CPU>::copy(int,const double*,int,double*,int);

template <> template <typename T>
T TensorOperation<CPU>::dot (int n, const T* x, int incx, const T* y, int incy) {
  return BLAS<CPU>::instance().dot(n, x, incx, y, incy);
}
template float TensorOperation<CPU>::dot(int,const float*,int,const float*,int);
template double TensorOperation<CPU>::dot(int,const double*,int,const double*,int);

template <> template <typename T>
T TensorOperation<CPU>::nrm2 (int n, const T* x, int incx) {
  return BLAS<CPU>::instance().nrm2(n, x, incx);
}
template float TensorOperation<CPU>::nrm2(int,const float*,int);
template double TensorOperation<CPU>::nrm2(int,const double*,int);

template <> template <typename T>
void TensorOperation<CPU>::ger (int m, int n, T alpha, const T* x, int incx,
  const T* y, int incy, T *A, const int lda) {
  BLAS<CPU>::instance().ger(m, n, alpha, x, incx, y, incy, A, lda);
}
template void TensorOperation<CPU>::ger(int,int,float,const float*,int,const float*,int,float*,int);
template void TensorOperation<CPU>::ger(int,int,double,const double*,int,const double*,int,double*,int);

template <> template <typename T>
void TensorOperation<CPU>::gemv (bool transa, int m, int n, T alpha, const T* A, int lda,
  const T* x, int incx, T beta, T* y, int incy) {
    BLAS<CPU>::instance().gemv(transa, m, n, alpha, A, lda, x, incx, beta, y, incy);
}
template void TensorOperation<CPU>::gemv(bool,int,int,float,const float*,int,const float*,int,float,float*,int);
template void TensorOperation<CPU>::gemv(bool,int,int,double,const double*,int,const double*,int,double,double*,int);

template <> template <typename T>
void TensorOperation<CPU>::gemm (bool transa, bool transb, int m, int n, int k, T alpha,
  const T *A, int lda, const T *B, int ldb, T beta, T *C, int ldc ) {
  BLAS<CPU>::instance().gemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
template void TensorOperation<CPU>::gemm(bool,bool,int,int,int,float,const float*,int,const float*,int,float,float*,int);
template void TensorOperation<CPU>::gemm(bool,bool,int,int,int,double,const double*,int,const double*,int,double,double*,int);




template <> template <typename T>
T TensorOperation<CPU>::get(const T* x) {
  return *x;
}

template float TensorOperation<CPU>::get(const float*);
template double TensorOperation<CPU>::get(const double*);




template <> template <typename T>
void TensorOperation<CPU>::set(T* x, T val) {
  *x = val;
}

template void TensorOperation<CPU>::set(float*, float);
template void TensorOperation<CPU>::set(double*, double);




template <> template <typename T>
void TensorOperation<CPU>::x_to_y (int n, const T* x, int incx, T* y, int incy) {
  TensorOperation<CPU>::template copy<T> (n, x, incx, y, incy);
}

template void TensorOperation<CPU>::x_to_y(int, const float*, int, float*, int);
template void TensorOperation<CPU>::x_to_y(int, const double*, int, double*, int);

template <> template <typename T>
void TensorOperation<CPU>::A_to_B (int m, int n, const T* A, int lda, T* B, int ldb, Device dest) {
  // cpu_A_to_B(m, n, A, lda, B, ldb);
  if (CPU == dest) {
    cudaMemcpy2D(B, ldb*sizeof(T), A, lda*sizeof(T), m*sizeof(T), n, cudaMemcpyHostToHost);
  } else {
    cudaMemcpy2D(B, ldb*sizeof(T), A, lda*sizeof(T), m*sizeof(T), n, cudaMemcpyHostToDevice);
  }
}

template void TensorOperation<CPU>::A_to_B(int, int, const float*, int, float*, int, Device);
template void TensorOperation<CPU>::A_to_B(int, int, const double*, int, double*, int, Device);




template <> template <typename T, typename OP>
void TensorOperation<CPU>::A_rows_op_x_to_A (T* A, int ha, int wa, int lda, const T* x, int incx) {
  cpu_A_rows_op_x_to_A<T, OP>(A, ha, wa, lda, x, incx);
}

template void TensorOperation<CPU>::A_rows_op_x_to_A<float, functor::Plus<CPU,float> >(float*,int,int,int,const float*,int);
template void TensorOperation<CPU>::A_rows_op_x_to_A<double, functor::Plus<CPU,double> >(double*,int,int,int,const double*,int);

template void TensorOperation<CPU>::A_rows_op_x_to_A<float, functor::Minus<CPU,float> >(float*,int,int,int,const float*,int);
template void TensorOperation<CPU>::A_rows_op_x_to_A<double, functor::Minus<CPU,double> >(double*,int,int,int,const double*,int);

template void TensorOperation<CPU>::A_rows_op_x_to_A<float, functor::Multiplies<CPU,float> >(float*,int,int,int,const float*,int);
template void TensorOperation<CPU>::A_rows_op_x_to_A<double, functor::Multiplies<CPU,double> >(double*,int,int,int,const double*,int);

template void TensorOperation<CPU>::A_rows_op_x_to_A<float, functor::Divides<CPU,float> >(float*,int,int,int,const float*,int);
template void TensorOperation<CPU>::A_rows_op_x_to_A<double, functor::Divides<CPU,double> >(double*,int,int,int,const double*,int);

template void TensorOperation<CPU>::A_rows_op_x_to_A<float, functor::Pow<CPU,float> >(float*,int,int,int,const float*,int);
template void TensorOperation<CPU>::A_rows_op_x_to_A<double, functor::Pow<CPU,double> >(double*,int,int,int,const double*,int);




template <> template <typename T, typename OP>
void TensorOperation<CPU>::x_op_A_rows_to_A (const T* x , int incx, T* A, int ha, int wa, int lda) {
  cpu_x_op_A_rows_to_A<T, OP>(x, incx, A, ha, wa, lda);
}

template void TensorOperation<CPU>::x_op_A_rows_to_A<float, functor::Plus<CPU,float> >(const float*,int,float*,int,int,int);
template void TensorOperation<CPU>::x_op_A_rows_to_A<double, functor::Plus<CPU,double> >(const double*,int,double*,int,int,int);

template void TensorOperation<CPU>::x_op_A_rows_to_A<float, functor::Minus<CPU,float> >(const float*,int,float*,int,int,int);
template void TensorOperation<CPU>::x_op_A_rows_to_A<double, functor::Minus<CPU,double> >(const double*,int,double*,int,int,int);

template void TensorOperation<CPU>::x_op_A_rows_to_A<float, functor::Multiplies<CPU,float> >(const float*,int,float*,int,int,int);
template void TensorOperation<CPU>::x_op_A_rows_to_A<double, functor::Multiplies<CPU,double> >(const double*,int,double*,int,int,int);

template void TensorOperation<CPU>::x_op_A_rows_to_A<float, functor::Divides<CPU,float> >(const float*,int,float*,int,int,int);
template void TensorOperation<CPU>::x_op_A_rows_to_A<double, functor::Divides<CPU,double> >(const double*,int,double*,int,int,int);

template void TensorOperation<CPU>::x_op_A_rows_to_A<float, functor::Pow<CPU,float> >(const float*,int,float*,int,int,int);
template void TensorOperation<CPU>::x_op_A_rows_to_A<double, functor::Pow<CPU,double> >(const double*,int,double*,int,int,int);




template <> template <typename T, typename OP>
void TensorOperation<CPU>::A_cols_op_x_to_A (T* A, int ha, int wa, int lda, const T* x, int incx) {
  cpu_A_cols_op_x_to_A<T, OP>(A, ha, wa, lda, x, incx);
}

template void TensorOperation<CPU>::A_cols_op_x_to_A<float, functor::Plus<CPU,float> >(float*,int,int,int,const float*,int);
template void TensorOperation<CPU>::A_cols_op_x_to_A<double, functor::Plus<CPU,double> >(double*,int,int,int,const double*,int);

template void TensorOperation<CPU>::A_cols_op_x_to_A<float, functor::Minus<CPU,float> >(float*,int,int,int,const float*,int);
template void TensorOperation<CPU>::A_cols_op_x_to_A<double, functor::Minus<CPU,double> >(double*,int,int,int,const double*,int);

template void TensorOperation<CPU>::A_cols_op_x_to_A<float, functor::Multiplies<CPU,float> >(float*,int,int,int,const float*,int);
template void TensorOperation<CPU>::A_cols_op_x_to_A<double, functor::Multiplies<CPU,double> >(double*,int,int,int,const double*,int);

template void TensorOperation<CPU>::A_cols_op_x_to_A<float, functor::Divides<CPU,float> >(float*,int,int,int,const float*,int);
template void TensorOperation<CPU>::A_cols_op_x_to_A<double, functor::Divides<CPU,double> >(double*,int,int,int,const double*,int);

template void TensorOperation<CPU>::A_cols_op_x_to_A<float, functor::Pow<CPU,float> >(float*,int,int,int,const float*,int);
template void TensorOperation<CPU>::A_cols_op_x_to_A<double, functor::Pow<CPU,double> >(double*,int,int,int,const double*,int);




template <> template <typename T, typename OP>
void TensorOperation<CPU>::x_op_A_cols_to_A (const T* x, int incx, T* A, int ha, int wa, int lda) {
  cpu_x_op_A_cols_to_A<T, OP>(x, incx, A, ha, wa, lda);
}

template void TensorOperation<CPU>::x_op_A_cols_to_A<float, functor::Plus<CPU,float> >(const float*,int,float*,int,int,int);
template void TensorOperation<CPU>::x_op_A_cols_to_A<double, functor::Plus<CPU,double> >(const double*,int,double*,int,int,int);

template void TensorOperation<CPU>::x_op_A_cols_to_A<float, functor::Minus<CPU,float> >(const float*,int,float*,int,int,int);
template void TensorOperation<CPU>::x_op_A_cols_to_A<double, functor::Minus<CPU,double> >(const double*,int,double*,int,int,int);

template void TensorOperation<CPU>::x_op_A_cols_to_A<float, functor::Multiplies<CPU,float> >(const float*,int,float*,int,int,int);
template void TensorOperation<CPU>::x_op_A_cols_to_A<double, functor::Multiplies<CPU,double> >(const double*,int,double*,int,int,int);

template void TensorOperation<CPU>::x_op_A_cols_to_A<float, functor::Divides<CPU,float> >(const float*,int,float*,int,int,int);
template void TensorOperation<CPU>::x_op_A_cols_to_A<double, functor::Divides<CPU,double> >(const double*,int,double*,int,int,int);

template void TensorOperation<CPU>::x_op_A_cols_to_A<float, functor::Pow<CPU,float> >(const float*,int,float*,int,int,int);
template void TensorOperation<CPU>::x_op_A_cols_to_A<double, functor::Pow<CPU,double> >(const double*,int,double*,int,int,int);




/*
 *
 * GPU DEVICE
 *
 */




template <> template <typename T>
T TensorOperation<GPU>::asum (int n, const T* x, int incx) {
  return BLAS<GPU>::instance().asum(n, x, incx);
}
template float TensorOperation<GPU>::asum(int,const float*,int);
template double TensorOperation<GPU>::asum(int,const double*,int);

template <> template <typename T>
void TensorOperation<GPU>::copy (int n, const T* x, int incx, T* y, int incy) {
  BLAS<GPU>::instance().copy(n, x, incx, y, incy);
}
template void TensorOperation<GPU>::copy(int,const float*,int,float*,int);
template void TensorOperation<GPU>::copy(int,const double*,int,double*,int);

template <> template <typename T>
T TensorOperation<GPU>::dot (int n, const T* x, int incx, const T* y, int incy) {
  return BLAS<GPU>::instance().dot(n, x, incx, y, incy);
}
template float TensorOperation<GPU>::dot(int,const float*,int,const float*,int);
template double TensorOperation<GPU>::dot(int,const double*,int,const double*,int);

template <> template <typename T>
T TensorOperation<GPU>::nrm2 (int n, const T* x, int incx) {
  return BLAS<GPU>::instance().nrm2(n, x, incx);
}
template float TensorOperation<GPU>::nrm2(int,const float*,int);
template double TensorOperation<GPU>::nrm2(int,const double*,int);

template <> template <typename T>
void TensorOperation<GPU>::ger (int m, int n, T alpha, const T* x, int incx,
  const T* y, int incy, T *A, const int lda) {
  BLAS<GPU>::instance().ger(m, n, alpha, x, incx, y, incy, A, lda);
}
template void TensorOperation<GPU>::ger(int,int,float,const float*,int,const float*,int,float*,int);
template void TensorOperation<GPU>::ger(int,int,double,const double*,int,const double*,int,double*,int);

template <> template <typename T>
void TensorOperation<GPU>::gemv (bool transa, int m, int n, T alpha, const T* A, int lda,
  const T* x, int incx, T beta, T* y, int incy) {
    BLAS<GPU>::instance().gemv(transa, m, n, alpha, A, lda, x, incx, beta, y, incy);
}
template void TensorOperation<GPU>::gemv(bool,int,int,float,const float*,int,const float*,int,float,float*,int);
template void TensorOperation<GPU>::gemv(bool,int,int,double,const double*,int,const double*,int,double,double*,int);

template <> template <typename T>
void TensorOperation<GPU>::gemm (bool transa, bool transb, int m, int n, int k, T alpha,
  const T *A, int lda, const T *B, int ldb, T beta, T *C, int ldc ) {
  BLAS<GPU>::instance().gemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
template void TensorOperation<GPU>::gemm(bool,bool,int,int,int,float,const float*,int,const float*,int,float,float*,int);
template void TensorOperation<GPU>::gemm(bool,bool,int,int,int,double,const double*,int,const double*,int,double,double*,int);




#define GLOBAL_MATRIX(h,w) dim3((((h)-1)/BLOCK_SIZE+1), (((w)-1)/BLOCK_SIZE+1))

#define BLOCK_MATRIX(h,w) dim3(BLOCK_SIZE,BLOCK_SIZE)

#define GLOBAL_VECTOR(len) dim3(((len)-1)/BLOCK_NUM_THREADS+1)

#define BLOCK_VECTOR(len) dim3(BLOCK_NUM_THREADS)




template <> template <typename OP>
void TensorOperation<GPU>::vector_call (OP op) {
  gpu_vector_call<<<GLOBAL_VECTOR(op.n), BLOCK_VECTOR(op.n)>>>(op);
  cudaDeviceSynchronize();
}

template void TensorOperation<GPU>::vector_call<>(v_to_x<GPU, float>);
template void TensorOperation<GPU>::vector_call<>(v_to_x<GPU, double>);


template void TensorOperation<GPU>::vector_call<>(x_op_y_to_z<GPU, float, functor::Plus<GPU, float> >);
template void TensorOperation<GPU>::vector_call<>(x_op_y_to_z<GPU, double, functor::Plus<GPU, double> >);
template void TensorOperation<GPU>::vector_call<>(x_op_y_to_z<GPU, float, functor::Minus<GPU, float> >);
template void TensorOperation<GPU>::vector_call<>(x_op_y_to_z<GPU, double, functor::Minus<GPU, double> >);
template void TensorOperation<GPU>::vector_call<>(x_op_y_to_z<GPU, float, functor::Multiplies<GPU, float> >);
template void TensorOperation<GPU>::vector_call<>(x_op_y_to_z<GPU, double, functor::Multiplies<GPU, double> >);
template void TensorOperation<GPU>::vector_call<>(x_op_y_to_z<GPU, float, functor::Divides<GPU, float> >);
template void TensorOperation<GPU>::vector_call<>(x_op_y_to_z<GPU, double, functor::Divides<GPU, double> >);
template void TensorOperation<GPU>::vector_call<>(x_op_y_to_z<GPU, float, functor::Pow<GPU, float> >);
template void TensorOperation<GPU>::vector_call<>(x_op_y_to_z<GPU, double, functor::Pow<GPU, double> >);


template void TensorOperation<GPU>::vector_call<>(x_op_v_to_y<GPU, float, functor::Plus<GPU, float> >);
template void TensorOperation<GPU>::vector_call<>(x_op_v_to_y<GPU, double, functor::Plus<GPU, double> >);
template void TensorOperation<GPU>::vector_call<>(x_op_v_to_y<GPU, float, functor::Minus<GPU, float> >);
template void TensorOperation<GPU>::vector_call<>(x_op_v_to_y<GPU, double, functor::Minus<GPU, double> >);
template void TensorOperation<GPU>::vector_call<>(x_op_v_to_y<GPU, float, functor::Multiplies<GPU, float> >);
template void TensorOperation<GPU>::vector_call<>(x_op_v_to_y<GPU, double, functor::Multiplies<GPU, double> >);
template void TensorOperation<GPU>::vector_call<>(x_op_v_to_y<GPU, float, functor::Divides<GPU, float> >);
template void TensorOperation<GPU>::vector_call<>(x_op_v_to_y<GPU, double, functor::Divides<GPU, double> >);
template void TensorOperation<GPU>::vector_call<>(x_op_v_to_y<GPU, float, functor::Pow<GPU, float> >);
template void TensorOperation<GPU>::vector_call<>(x_op_v_to_y<GPU, double, functor::Pow<GPU, double> >);


template void TensorOperation<GPU>::vector_call<>(v_op_x_to_y<GPU, float, functor::Plus<GPU, float> >);
template void TensorOperation<GPU>::vector_call<>(v_op_x_to_y<GPU, double, functor::Plus<GPU, double> >);
template void TensorOperation<GPU>::vector_call<>(v_op_x_to_y<GPU, float, functor::Minus<GPU, float> >);
template void TensorOperation<GPU>::vector_call<>(v_op_x_to_y<GPU, double, functor::Minus<GPU, double> >);
template void TensorOperation<GPU>::vector_call<>(v_op_x_to_y<GPU, float, functor::Multiplies<GPU, float> >);
template void TensorOperation<GPU>::vector_call<>(v_op_x_to_y<GPU, double, functor::Multiplies<GPU, double> >);
template void TensorOperation<GPU>::vector_call<>(v_op_x_to_y<GPU, float, functor::Divides<GPU, float> >);
template void TensorOperation<GPU>::vector_call<>(v_op_x_to_y<GPU, double, functor::Divides<GPU, double> >);
template void TensorOperation<GPU>::vector_call<>(v_op_x_to_y<GPU, float, functor::Pow<GPU, float> >);
template void TensorOperation<GPU>::vector_call<>(v_op_x_to_y<GPU, double, functor::Pow<GPU, double> >);


template void TensorOperation<GPU>::vector_call<>(fn_x_to_y<GPU, float, functor::Log<GPU, float> >);
template void TensorOperation<GPU>::vector_call<>(fn_x_to_y<GPU, double, functor::Log<GPU, double> >);
template void TensorOperation<GPU>::vector_call<>(fn_x_to_y<GPU, float, functor::Exp<GPU, float> >);
template void TensorOperation<GPU>::vector_call<>(fn_x_to_y<GPU, double, functor::Exp<GPU, double> >);
template void TensorOperation<GPU>::vector_call<>(fn_x_to_y<GPU, float, functor::Abs<GPU, float> >);
template void TensorOperation<GPU>::vector_call<>(fn_x_to_y<GPU, double, functor::Abs<GPU, double> >);
template void TensorOperation<GPU>::vector_call<>(fn_x_to_y<GPU, float, functor::Sign<GPU, float> >);
template void TensorOperation<GPU>::vector_call<>(fn_x_to_y<GPU, double, functor::Sign<GPU, double> >);
template void TensorOperation<GPU>::vector_call<>(fn_x_to_y<GPU, float, functor::Sigmoid<GPU, float> >);
template void TensorOperation<GPU>::vector_call<>(fn_x_to_y<GPU, double, functor::Sigmoid<GPU, double> >);
template void TensorOperation<GPU>::vector_call<>(fn_x_to_y<GPU, float, functor::Dsigmoid<GPU, float> >);
template void TensorOperation<GPU>::vector_call<>(fn_x_to_y<GPU, double, functor::Dsigmoid<GPU, double> >);
template void TensorOperation<GPU>::vector_call<>(fn_x_to_y<GPU, float, functor::Tanh<GPU, float> >);
template void TensorOperation<GPU>::vector_call<>(fn_x_to_y<GPU, double, functor::Tanh<GPU, double> >);
template void TensorOperation<GPU>::vector_call<>(fn_x_to_y<GPU, float, functor::Dtanh<GPU, float> >);
template void TensorOperation<GPU>::vector_call<>(fn_x_to_y<GPU, double, functor::Dtanh<GPU, double> >);




template <> template <typename OP>
void TensorOperation<GPU>::matrix_call (OP op) {
  gpu_matrix_call<<<GLOBAL_MATRIX(op.m, op.n), BLOCK_MATRIX(op.m, op.n)>>>(op);
  cudaDeviceSynchronize();
}

template void TensorOperation<GPU>::matrix_call<>(v_to_A<GPU, float>);
template void TensorOperation<GPU>::matrix_call<>(v_to_A<GPU, double>);


template void TensorOperation<GPU>::matrix_call<>(A_op_B_to_C<GPU, float, functor::Plus<GPU, float> >);
template void TensorOperation<GPU>::matrix_call<>(A_op_B_to_C<GPU, double, functor::Plus<GPU, double> >);
template void TensorOperation<GPU>::matrix_call<>(A_op_B_to_C<GPU, float, functor::Minus<GPU, float> >);
template void TensorOperation<GPU>::matrix_call<>(A_op_B_to_C<GPU, double, functor::Minus<GPU, double> >);
template void TensorOperation<GPU>::matrix_call<>(A_op_B_to_C<GPU, float, functor::Multiplies<GPU, float> >);
template void TensorOperation<GPU>::matrix_call<>(A_op_B_to_C<GPU, double, functor::Multiplies<GPU, double> >);
template void TensorOperation<GPU>::matrix_call<>(A_op_B_to_C<GPU, float, functor::Divides<GPU, float> >);
template void TensorOperation<GPU>::matrix_call<>(A_op_B_to_C<GPU, double, functor::Divides<GPU, double> >);
template void TensorOperation<GPU>::matrix_call<>(A_op_B_to_C<GPU, float, functor::Pow<GPU, float> >);
template void TensorOperation<GPU>::matrix_call<>(A_op_B_to_C<GPU, double, functor::Pow<GPU, double> >);


template void TensorOperation<GPU>::matrix_call<>(A_op_v_to_B<GPU, float, functor::Plus<GPU, float> >);
template void TensorOperation<GPU>::matrix_call<>(A_op_v_to_B<GPU, double, functor::Plus<GPU, double> >);
template void TensorOperation<GPU>::matrix_call<>(A_op_v_to_B<GPU, float, functor::Minus<GPU, float> >);
template void TensorOperation<GPU>::matrix_call<>(A_op_v_to_B<GPU, double, functor::Minus<GPU, double> >);
template void TensorOperation<GPU>::matrix_call<>(A_op_v_to_B<GPU, float, functor::Multiplies<GPU, float> >);
template void TensorOperation<GPU>::matrix_call<>(A_op_v_to_B<GPU, double, functor::Multiplies<GPU, double> >);
template void TensorOperation<GPU>::matrix_call<>(A_op_v_to_B<GPU, float, functor::Divides<GPU, float> >);
template void TensorOperation<GPU>::matrix_call<>(A_op_v_to_B<GPU, double, functor::Divides<GPU, double> >);
template void TensorOperation<GPU>::matrix_call<>(A_op_v_to_B<GPU, float, functor::Pow<GPU, float> >);
template void TensorOperation<GPU>::matrix_call<>(A_op_v_to_B<GPU, double, functor::Pow<GPU, double> >);


template void TensorOperation<GPU>::matrix_call<>(v_op_A_to_B<GPU, float, functor::Plus<GPU, float> >);
template void TensorOperation<GPU>::matrix_call<>(v_op_A_to_B<GPU, double, functor::Plus<GPU, double> >);
template void TensorOperation<GPU>::matrix_call<>(v_op_A_to_B<GPU, float, functor::Minus<GPU, float> >);
template void TensorOperation<GPU>::matrix_call<>(v_op_A_to_B<GPU, double, functor::Minus<GPU, double> >);
template void TensorOperation<GPU>::matrix_call<>(v_op_A_to_B<GPU, float, functor::Multiplies<GPU, float> >);
template void TensorOperation<GPU>::matrix_call<>(v_op_A_to_B<GPU, double, functor::Multiplies<GPU, double> >);
template void TensorOperation<GPU>::matrix_call<>(v_op_A_to_B<GPU, float, functor::Divides<GPU, float> >);
template void TensorOperation<GPU>::matrix_call<>(v_op_A_to_B<GPU, double, functor::Divides<GPU, double> >);
template void TensorOperation<GPU>::matrix_call<>(v_op_A_to_B<GPU, float, functor::Pow<GPU, float> >);
template void TensorOperation<GPU>::matrix_call<>(v_op_A_to_B<GPU, double, functor::Pow<GPU, double> >);


template void TensorOperation<GPU>::matrix_call<>(A_each_row_op_x_to_B<GPU, float, functor::Plus<GPU, float> >);
template void TensorOperation<GPU>::matrix_call<>(A_each_row_op_x_to_B<GPU, double, functor::Plus<GPU, double> >);
template void TensorOperation<GPU>::matrix_call<>(A_each_row_op_x_to_B<GPU, float, functor::Minus<GPU, float> >);
template void TensorOperation<GPU>::matrix_call<>(A_each_row_op_x_to_B<GPU, double, functor::Minus<GPU, double> >);
template void TensorOperation<GPU>::matrix_call<>(A_each_row_op_x_to_B<GPU, float, functor::Multiplies<GPU, float> >);
template void TensorOperation<GPU>::matrix_call<>(A_each_row_op_x_to_B<GPU, double, functor::Multiplies<GPU, double> >);
template void TensorOperation<GPU>::matrix_call<>(A_each_row_op_x_to_B<GPU, float, functor::Divides<GPU, float> >);
template void TensorOperation<GPU>::matrix_call<>(A_each_row_op_x_to_B<GPU, double, functor::Divides<GPU, double> >);
template void TensorOperation<GPU>::matrix_call<>(A_each_row_op_x_to_B<GPU, float, functor::Pow<GPU, float> >);
template void TensorOperation<GPU>::matrix_call<>(A_each_row_op_x_to_B<GPU, double, functor::Pow<GPU, double> >);
template void TensorOperation<GPU>::matrix_call<>(A_each_row_op_x_to_B<GPU, float, functor::Left<GPU, float> >);
template void TensorOperation<GPU>::matrix_call<>(A_each_row_op_x_to_B<GPU, double, functor::Left<GPU, double> >);
template void TensorOperation<GPU>::matrix_call<>(A_each_row_op_x_to_B<GPU, float, functor::Right<GPU, float> >);
template void TensorOperation<GPU>::matrix_call<>(A_each_row_op_x_to_B<GPU, double, functor::Right<GPU, double> >);


template void TensorOperation<GPU>::matrix_call<>(x_op_A_each_row_to_B<GPU, float, functor::Plus<GPU, float> >);
template void TensorOperation<GPU>::matrix_call<>(x_op_A_each_row_to_B<GPU, double, functor::Plus<GPU, double> >);
template void TensorOperation<GPU>::matrix_call<>(x_op_A_each_row_to_B<GPU, float, functor::Minus<GPU, float> >);
template void TensorOperation<GPU>::matrix_call<>(x_op_A_each_row_to_B<GPU, double, functor::Minus<GPU, double> >);
template void TensorOperation<GPU>::matrix_call<>(x_op_A_each_row_to_B<GPU, float, functor::Multiplies<GPU, float> >);
template void TensorOperation<GPU>::matrix_call<>(x_op_A_each_row_to_B<GPU, double, functor::Multiplies<GPU, double> >);
template void TensorOperation<GPU>::matrix_call<>(x_op_A_each_row_to_B<GPU, float, functor::Divides<GPU, float> >);
template void TensorOperation<GPU>::matrix_call<>(x_op_A_each_row_to_B<GPU, double, functor::Divides<GPU, double> >);
template void TensorOperation<GPU>::matrix_call<>(x_op_A_each_row_to_B<GPU, float, functor::Pow<GPU, float> >);
template void TensorOperation<GPU>::matrix_call<>(x_op_A_each_row_to_B<GPU, double, functor::Pow<GPU, double> >);
template void TensorOperation<GPU>::matrix_call<>(x_op_A_each_row_to_B<GPU, float, functor::Left<GPU, float> >);
template void TensorOperation<GPU>::matrix_call<>(x_op_A_each_row_to_B<GPU, double, functor::Left<GPU, double> >);
template void TensorOperation<GPU>::matrix_call<>(x_op_A_each_row_to_B<GPU, float, functor::Right<GPU, float> >);
template void TensorOperation<GPU>::matrix_call<>(x_op_A_each_row_to_B<GPU, double, functor::Right<GPU, double> >);


template void TensorOperation<GPU>::matrix_call<>(A_each_col_op_x_to_B<GPU, float, functor::Plus<GPU, float> >);
template void TensorOperation<GPU>::matrix_call<>(A_each_col_op_x_to_B<GPU, double, functor::Plus<GPU, double> >);
template void TensorOperation<GPU>::matrix_call<>(A_each_col_op_x_to_B<GPU, float, functor::Minus<GPU, float> >);
template void TensorOperation<GPU>::matrix_call<>(A_each_col_op_x_to_B<GPU, double, functor::Minus<GPU, double> >);
template void TensorOperation<GPU>::matrix_call<>(A_each_col_op_x_to_B<GPU, float, functor::Multiplies<GPU, float> >);
template void TensorOperation<GPU>::matrix_call<>(A_each_col_op_x_to_B<GPU, double, functor::Multiplies<GPU, double> >);
template void TensorOperation<GPU>::matrix_call<>(A_each_col_op_x_to_B<GPU, float, functor::Divides<GPU, float> >);
template void TensorOperation<GPU>::matrix_call<>(A_each_col_op_x_to_B<GPU, double, functor::Divides<GPU, double> >);
template void TensorOperation<GPU>::matrix_call<>(A_each_col_op_x_to_B<GPU, float, functor::Pow<GPU, float> >);
template void TensorOperation<GPU>::matrix_call<>(A_each_col_op_x_to_B<GPU, double, functor::Pow<GPU, double> >);
template void TensorOperation<GPU>::matrix_call<>(A_each_col_op_x_to_B<GPU, float, functor::Left<GPU, float> >);
template void TensorOperation<GPU>::matrix_call<>(A_each_col_op_x_to_B<GPU, double, functor::Left<GPU, double> >);
template void TensorOperation<GPU>::matrix_call<>(A_each_col_op_x_to_B<GPU, float, functor::Right<GPU, float> >);
template void TensorOperation<GPU>::matrix_call<>(A_each_col_op_x_to_B<GPU, double, functor::Right<GPU, double> >);


template void TensorOperation<GPU>::matrix_call<>(x_op_A_each_col_to_B<GPU, float, functor::Plus<GPU, float> >);
template void TensorOperation<GPU>::matrix_call<>(x_op_A_each_col_to_B<GPU, double, functor::Plus<GPU, double> >);
template void TensorOperation<GPU>::matrix_call<>(x_op_A_each_col_to_B<GPU, float, functor::Minus<GPU, float> >);
template void TensorOperation<GPU>::matrix_call<>(x_op_A_each_col_to_B<GPU, double, functor::Minus<GPU, double> >);
template void TensorOperation<GPU>::matrix_call<>(x_op_A_each_col_to_B<GPU, float, functor::Multiplies<GPU, float> >);
template void TensorOperation<GPU>::matrix_call<>(x_op_A_each_col_to_B<GPU, double, functor::Multiplies<GPU, double> >);
template void TensorOperation<GPU>::matrix_call<>(x_op_A_each_col_to_B<GPU, float, functor::Divides<GPU, float> >);
template void TensorOperation<GPU>::matrix_call<>(x_op_A_each_col_to_B<GPU, double, functor::Divides<GPU, double> >);
template void TensorOperation<GPU>::matrix_call<>(x_op_A_each_col_to_B<GPU, float, functor::Pow<GPU, float> >);
template void TensorOperation<GPU>::matrix_call<>(x_op_A_each_col_to_B<GPU, double, functor::Pow<GPU, double> >);
template void TensorOperation<GPU>::matrix_call<>(x_op_A_each_col_to_B<GPU, float, functor::Left<GPU, float> >);
template void TensorOperation<GPU>::matrix_call<>(x_op_A_each_col_to_B<GPU, double, functor::Left<GPU, double> >);
template void TensorOperation<GPU>::matrix_call<>(x_op_A_each_col_to_B<GPU, float, functor::Right<GPU, float> >);
template void TensorOperation<GPU>::matrix_call<>(x_op_A_each_col_to_B<GPU, double, functor::Right<GPU, double> >);


template void TensorOperation<GPU>::matrix_call<>(fn_A_to_B<GPU, float, functor::Log<GPU, float> >);
template void TensorOperation<GPU>::matrix_call<>(fn_A_to_B<GPU, double, functor::Log<GPU, double> >);
template void TensorOperation<GPU>::matrix_call<>(fn_A_to_B<GPU, float, functor::Exp<GPU, float> >);
template void TensorOperation<GPU>::matrix_call<>(fn_A_to_B<GPU, double, functor::Exp<GPU, double> >);
template void TensorOperation<GPU>::matrix_call<>(fn_A_to_B<GPU, float, functor::Abs<GPU, float> >);
template void TensorOperation<GPU>::matrix_call<>(fn_A_to_B<GPU, double, functor::Abs<GPU, double> >);
template void TensorOperation<GPU>::matrix_call<>(fn_A_to_B<GPU, float, functor::Sign<GPU, float> >);
template void TensorOperation<GPU>::matrix_call<>(fn_A_to_B<GPU, double, functor::Sign<GPU, double> >);
template void TensorOperation<GPU>::matrix_call<>(fn_A_to_B<GPU, float, functor::Sigmoid<GPU, float> >);
template void TensorOperation<GPU>::matrix_call<>(fn_A_to_B<GPU, double, functor::Sigmoid<GPU, double> >);
template void TensorOperation<GPU>::matrix_call<>(fn_A_to_B<GPU, float, functor::Dsigmoid<GPU, float> >);
template void TensorOperation<GPU>::matrix_call<>(fn_A_to_B<GPU, double, functor::Dsigmoid<GPU, double> >);
template void TensorOperation<GPU>::matrix_call<>(fn_A_to_B<GPU, float, functor::Tanh<GPU, float> >);
template void TensorOperation<GPU>::matrix_call<>(fn_A_to_B<GPU, double, functor::Tanh<GPU, double> >);
template void TensorOperation<GPU>::matrix_call<>(fn_A_to_B<GPU, float, functor::Dtanh<GPU, float> >);
template void TensorOperation<GPU>::matrix_call<>(fn_A_to_B<GPU, double, functor::Dtanh<GPU, double> >);


template void TensorOperation<GPU>::matrix_call<>(trans_A_to_B<GPU, float>);
template void TensorOperation<GPU>::matrix_call<>(trans_A_to_B<GPU, double>);




template <> template <typename T, typename OP>
T TensorOperation<GPU>::x_reduce (int n, const T* x, int incx) {
  T* result;
  cudaMalloc((void**)&result, sizeof(T));
  gpu_x_reduce<T, OP><<<1, BLOCK_NUM_THREADS>>>(n, x, incx, result);
  cudaDeviceSynchronize();
  T ret;
  cudaMemcpy(&ret, result, sizeof(T), cudaMemcpyDeviceToHost);
  cudaFree(result);
  return ret;
}

template float TensorOperation<GPU>::x_reduce<float, functor::reduce_sum<GPU,float> >(int,const float*,int);
template double TensorOperation<GPU>::x_reduce<double, functor::reduce_sum<GPU,double> >(int,const double*,int);

template float TensorOperation<GPU>::x_reduce<float, functor::reduce_max<GPU,float> >(int,const float*,int);
template double TensorOperation<GPU>::x_reduce<double, functor::reduce_max<GPU,double> >(int,const double*,int);

template float TensorOperation<GPU>::x_reduce<float, functor::reduce_min<GPU,float> >(int,const float*,int);
template double TensorOperation<GPU>::x_reduce<double, functor::reduce_min<GPU,double> >(int,const double*,int);

template float TensorOperation<GPU>::x_reduce<float, functor::reduce_sum_of_abs<GPU,float> >(int,const float*,int);
template double TensorOperation<GPU>::x_reduce<double, functor::reduce_sum_of_abs<GPU,double> >(int,const double*,int);

template float TensorOperation<GPU>::x_reduce<float, functor::reduce_sum_of_squared<GPU,float> >(int,const float*,int);
template double TensorOperation<GPU>::x_reduce<double, functor::reduce_sum_of_squared<GPU,double> >(int,const double*,int);




template <> template <typename T, typename OP>
T TensorOperation<GPU>::A_reduce (int m, int n, const T* A, int lda) {
  T* result;
  cudaMalloc((void**)&result, sizeof(T));
  gpu_A_reduce<T, OP><<<1, BLOCK_NUM_THREADS>>>(m, n, A, lda, result);
  cudaDeviceSynchronize();
  T ret;
  cudaMemcpy(&ret, result, sizeof(T), cudaMemcpyDeviceToHost);
  cudaFree(result);
  return ret;
}

template float TensorOperation<GPU>::A_reduce<float, functor::reduce_sum<GPU,float> >(int,int,const float*,int);
template double TensorOperation<GPU>::A_reduce<double, functor::reduce_sum<GPU,double> >(int,int,const double*,int);

template float TensorOperation<GPU>::A_reduce<float, functor::reduce_max<GPU,float> >(int,int,const float*,int);
template double TensorOperation<GPU>::A_reduce<double, functor::reduce_max<GPU,double> >(int,int,const double*,int);

template float TensorOperation<GPU>::A_reduce<float, functor::reduce_min<GPU,float> >(int,int,const float*,int);
template double TensorOperation<GPU>::A_reduce<double, functor::reduce_min<GPU,double> >(int,int,const double*,int);

template float TensorOperation<GPU>::A_reduce<float, functor::reduce_sum_of_abs<GPU,float> >(int,int,const float*,int);
template double TensorOperation<GPU>::A_reduce<double, functor::reduce_sum_of_abs<GPU,double> >(int,int,const double*,int);

template float TensorOperation<GPU>::A_reduce<float, functor::reduce_sum_of_squared<GPU,float> >(int,int,const float*,int);
template double TensorOperation<GPU>::A_reduce<double, functor::reduce_sum_of_squared<GPU,double> >(int,int,const double*,int);




template <> template <typename T, typename OP>
void TensorOperation<GPU>::A_each_row_reduce_to_x (int m, int n, const T* A, int lda, T* x, int incx) {
  gpu_A_each_row_reduce_to_x<T, OP><<<m, BLOCK_NUM_THREADS>>>(m, n, A, lda, x, incx);
  cudaDeviceSynchronize();
}

template void TensorOperation<GPU>::A_each_row_reduce_to_x<float, functor::reduce_sum<GPU,float> >(int,int,const float*,int,float*,int);
template void TensorOperation<GPU>::A_each_row_reduce_to_x<double, functor::reduce_sum<GPU,double> >(int,int,const double*,int,double*,int);

template void TensorOperation<GPU>::A_each_row_reduce_to_x<float, functor::reduce_max<GPU,float> >(int,int,const float*,int,float*,int);
template void TensorOperation<GPU>::A_each_row_reduce_to_x<double, functor::reduce_max<GPU,double> >(int,int,const double*,int,double*,int);

template void TensorOperation<GPU>::A_each_row_reduce_to_x<float, functor::reduce_min<GPU,float> >(int,int,const float*,int,float*,int);
template void TensorOperation<GPU>::A_each_row_reduce_to_x<double, functor::reduce_min<GPU,double> >(int,int,const double*,int,double*,int);

template void TensorOperation<GPU>::A_each_row_reduce_to_x<float, functor::reduce_sum_of_abs<GPU,float> >(int,int,const float*,int,float*,int);
template void TensorOperation<GPU>::A_each_row_reduce_to_x<double, functor::reduce_sum_of_abs<GPU,double> >(int,int,const double*,int,double*,int);

template void TensorOperation<GPU>::A_each_row_reduce_to_x<float, functor::reduce_sum_of_squared<GPU,float> >(int,int,const float*,int,float*,int);
template void TensorOperation<GPU>::A_each_row_reduce_to_x<double, functor::reduce_sum_of_squared<GPU,double> >(int,int,const double*,int,double*,int);




template <> template <typename T, typename OP>
void TensorOperation<GPU>::A_each_col_reduce_to_x (int m, int n, const T* A, int lda, T* x, int incx) {
  gpu_A_each_col_reduce_to_x<T, OP><<<n, BLOCK_NUM_THREADS>>>(m, n, A, lda, x, incx);
  cudaDeviceSynchronize();
}

template void TensorOperation<GPU>::A_each_col_reduce_to_x<float, functor::reduce_sum<GPU,float> >(int,int,const float*,int,float*,int);
template void TensorOperation<GPU>::A_each_col_reduce_to_x<double, functor::reduce_sum<GPU,double> >(int,int,const double*,int,double*,int);

template void TensorOperation<GPU>::A_each_col_reduce_to_x<float, functor::reduce_max<GPU,float> >(int,int,const float*,int,float*,int);
template void TensorOperation<GPU>::A_each_col_reduce_to_x<double, functor::reduce_max<GPU,double> >(int,int,const double*,int,double*,int);

template void TensorOperation<GPU>::A_each_col_reduce_to_x<float, functor::reduce_min<GPU,float> >(int,int,const float*,int,float*,int);
template void TensorOperation<GPU>::A_each_col_reduce_to_x<double, functor::reduce_min<GPU,double> >(int,int,const double*,int,double*,int);

template void TensorOperation<GPU>::A_each_col_reduce_to_x<float, functor::reduce_sum_of_abs<GPU,float> >(int,int,const float*,int,float*,int);
template void TensorOperation<GPU>::A_each_col_reduce_to_x<double, functor::reduce_sum_of_abs<GPU,double> >(int,int,const double*,int,double*,int);

template void TensorOperation<GPU>::A_each_col_reduce_to_x<float, functor::reduce_sum_of_squared<GPU,float> >(int,int,const float*,int,float*,int);
template void TensorOperation<GPU>::A_each_col_reduce_to_x<double, functor::reduce_sum_of_squared<GPU,double> >(int,int,const double*,int,double*,int);



template <> template <typename T>
T TensorOperation<GPU>::get(const T* x) {
  T r;
  cudaMemcpy(&r, x, sizeof(T), cudaMemcpyDeviceToHost );
  return r;
}

template float TensorOperation<GPU>::get(const float*);
template double TensorOperation<GPU>::get(const double*);




template <> template <typename T>
void TensorOperation<GPU>::set(T* x, T val) {
  cudaMemcpy(x, &val, sizeof(T), cudaMemcpyHostToDevice );
}

template void TensorOperation<GPU>::set(float*, float);
template void TensorOperation<GPU>::set(double*, double);




template <> template <typename T>
void TensorOperation<GPU>::x_to_y (int n, const T* x, int incx, T* y, int incy) {
  TensorOperation<GPU>::template copy<T> (n, x, incx, y, incy);
  cudaDeviceSynchronize();
}

template void TensorOperation<GPU>::x_to_y(int, const float*, int, float*, int);
template void TensorOperation<GPU>::x_to_y(int, const double*, int, double*, int);

template <> template <typename T>
void TensorOperation<GPU>::A_to_B (int m, int n, const T* A, int lda, T* B, int ldb, Device dest) {
  // gpu_A_to_B<<<GLOBAL_MATRIX(m, n), BLOCK_MATRIX(m, n)>>>(m, n, A, lda, B, ldb);
  if (GPU == dest) {
    cudaMemcpy2D(B, ldb*sizeof(T), A, lda*sizeof(T), m*sizeof(T), n, cudaMemcpyDeviceToDevice);
  } else {
    cudaMemcpy2D(B, ldb*sizeof(T), A, lda*sizeof(T), m*sizeof(T), n, cudaMemcpyDeviceToHost);
  }
}

template void TensorOperation<GPU>::A_to_B(int, int, const float*, int, float*, int, Device);
template void TensorOperation<GPU>::A_to_B(int, int, const double*, int, double*, int, Device);




template <> template <typename T, typename OP>
void TensorOperation<GPU>::A_rows_op_x_to_A (T* A, int ha, int wa, int lda, const T* x, int incx) {
  gpu_A_rows_op_x_to_A<T, OP><<<GLOBAL_MATRIX(ha, wa), BLOCK_MATRIX(ha, wa)>>>(A, ha, wa, lda, x, incx);
  cudaDeviceSynchronize();
}

template void TensorOperation<GPU>::A_rows_op_x_to_A<float, functor::Plus<GPU,float> >(float*,int,int,int,const float*,int);
template void TensorOperation<GPU>::A_rows_op_x_to_A<double, functor::Plus<GPU,double> >(double*,int,int,int,const double*,int);

template void TensorOperation<GPU>::A_rows_op_x_to_A<float, functor::Minus<GPU,float> >(float*,int,int,int,const float*,int);
template void TensorOperation<GPU>::A_rows_op_x_to_A<double, functor::Minus<GPU,double> >(double*,int,int,int,const double*,int);

template void TensorOperation<GPU>::A_rows_op_x_to_A<float, functor::Multiplies<GPU,float> >(float*,int,int,int,const float*,int);
template void TensorOperation<GPU>::A_rows_op_x_to_A<double, functor::Multiplies<GPU,double> >(double*,int,int,int,const double*,int);

template void TensorOperation<GPU>::A_rows_op_x_to_A<float, functor::Divides<GPU,float> >(float*,int,int,int,const float*,int);
template void TensorOperation<GPU>::A_rows_op_x_to_A<double, functor::Divides<GPU,double> >(double*,int,int,int,const double*,int);

template void TensorOperation<GPU>::A_rows_op_x_to_A<float, functor::Pow<GPU,float> >(float*,int,int,int,const float*,int);
template void TensorOperation<GPU>::A_rows_op_x_to_A<double, functor::Pow<GPU,double> >(double*,int,int,int,const double*,int);




template <> template <typename T, typename OP>
void TensorOperation<GPU>::x_op_A_rows_to_A (const T* x , int incx, T* A, int ha, int wa,int lda) {
  gpu_x_op_A_rows_to_A<T, OP><<<GLOBAL_MATRIX(ha, wa), BLOCK_MATRIX(ha, wa)>>>(x, incx, A, ha, wa,lda);
  cudaDeviceSynchronize();
}

template void TensorOperation<GPU>::x_op_A_rows_to_A<float, functor::Plus<GPU,float> >(const float*,int,float*,int,int,int);
template void TensorOperation<GPU>::x_op_A_rows_to_A<double, functor::Plus<GPU,double> >(const double*,int,double*,int,int,int);

template void TensorOperation<GPU>::x_op_A_rows_to_A<float, functor::Minus<GPU,float> >(const float*,int,float*,int,int,int);
template void TensorOperation<GPU>::x_op_A_rows_to_A<double, functor::Minus<GPU,double> >(const double*,int,double*,int,int,int);

template void TensorOperation<GPU>::x_op_A_rows_to_A<float, functor::Multiplies<GPU,float> >(const float*,int,float*,int,int,int);
template void TensorOperation<GPU>::x_op_A_rows_to_A<double, functor::Multiplies<GPU,double> >(const double*,int,double*,int,int,int);

template void TensorOperation<GPU>::x_op_A_rows_to_A<float, functor::Divides<GPU,float> >(const float*,int,float*,int,int,int);
template void TensorOperation<GPU>::x_op_A_rows_to_A<double, functor::Divides<GPU,double> >(const double*,int,double*,int,int,int);

template void TensorOperation<GPU>::x_op_A_rows_to_A<float, functor::Pow<GPU,float> >(const float*,int,float*,int,int,int);
template void TensorOperation<GPU>::x_op_A_rows_to_A<double, functor::Pow<GPU,double> >(const double*,int,double*,int,int,int);




template <> template <typename T, typename OP>
void TensorOperation<GPU>::A_cols_op_x_to_A (T* A, int ha, int wa, int lda, const T* x, int incx) {
  gpu_A_cols_op_x_to_A<T, OP><<<GLOBAL_MATRIX(ha, wa), BLOCK_MATRIX(ha, wa)>>>(A, ha, wa, lda, x, incx);
  cudaDeviceSynchronize();
}

template void TensorOperation<GPU>::A_cols_op_x_to_A<float, functor::Plus<GPU,float> >(float*,int,int,int,const float*,int);
template void TensorOperation<GPU>::A_cols_op_x_to_A<double, functor::Plus<GPU,double> >(double*,int,int,int,const double*,int);

template void TensorOperation<GPU>::A_cols_op_x_to_A<float, functor::Minus<GPU,float> >(float*,int,int,int,const float*,int);
template void TensorOperation<GPU>::A_cols_op_x_to_A<double, functor::Minus<GPU,double> >(double*,int,int,int,const double*,int);

template void TensorOperation<GPU>::A_cols_op_x_to_A<float, functor::Multiplies<GPU,float> >(float*,int,int,int,const float*,int);
template void TensorOperation<GPU>::A_cols_op_x_to_A<double, functor::Multiplies<GPU,double> >(double*,int,int,int,const double*,int);

template void TensorOperation<GPU>::A_cols_op_x_to_A<float, functor::Divides<GPU,float> >(float*,int,int,int,const float*,int);
template void TensorOperation<GPU>::A_cols_op_x_to_A<double, functor::Divides<GPU,double> >(double*,int,int,int,const double*,int);

template void TensorOperation<GPU>::A_cols_op_x_to_A<float, functor::Pow<GPU,float> >(float*,int,int,int,const float*,int);
template void TensorOperation<GPU>::A_cols_op_x_to_A<double, functor::Pow<GPU,double> >(double*,int,int,int,const double*,int);




template <> template <typename T, typename OP>
void TensorOperation<GPU>::x_op_A_cols_to_A (const T* x, int incx, T* A, int ha, int wa, int lda) {
  gpu_x_op_A_cols_to_A<T, OP><<<GLOBAL_MATRIX(ha, wa), BLOCK_MATRIX(ha, wa)>>>(x, incx, A, ha, wa, lda);
  cudaDeviceSynchronize();
}

template void TensorOperation<GPU>::x_op_A_cols_to_A<float, functor::Plus<GPU,float> >(const float*,int,float*,int,int,int);
template void TensorOperation<GPU>::x_op_A_cols_to_A<double, functor::Plus<GPU,double> >(const double*,int,double*,int,int,int);

template void TensorOperation<GPU>::x_op_A_cols_to_A<float, functor::Minus<GPU,float> >(const float*,int,float*,int,int,int);
template void TensorOperation<GPU>::x_op_A_cols_to_A<double, functor::Minus<GPU,double> >(const double*,int,double*,int,int,int);

template void TensorOperation<GPU>::x_op_A_cols_to_A<float, functor::Multiplies<GPU,float> >(const float*,int,float*,int,int,int);
template void TensorOperation<GPU>::x_op_A_cols_to_A<double, functor::Multiplies<GPU,double> >(const double*,int,double*,int,int,int);

template void TensorOperation<GPU>::x_op_A_cols_to_A<float, functor::Divides<GPU,float> >(const float*,int,float*,int,int,int);
template void TensorOperation<GPU>::x_op_A_cols_to_A<double, functor::Divides<GPU,double> >(const double*,int,double*,int,int,int);

template void TensorOperation<GPU>::x_op_A_cols_to_A<float, functor::Pow<GPU,float> >(const float*,int,float*,int,int,int);
template void TensorOperation<GPU>::x_op_A_cols_to_A<double, functor::Pow<GPU,double> >(const double*,int,double*,int,int,int);




}
