#ifndef DEVICE_HPP
#define DEVICE_HPP
/*!
 * \file device.h
 * \brief define device
 * \author ner.center
 * \date 2014-06-21
 */

#include <cublas_v2.h>


namespace tensor {

#ifndef DEVICE_DEFINE
#define DEVICE_DEFINE
/*!
 * \enum Device
 * \brief device type, CPU or GPU
 */
enum Device {CPU, GPU};
#endif // DEVICE_DEFINE


namespace functor {


  // operator(x, y)

  template <Device dev, typename T>
  struct Plus {
    __host__ __device__ T operator() (const T& x, const T& y) { return x + y; }
  };

  template <Device dev, typename T>
  struct Minus {
    __host__ __device__ T operator() (const T& x, const T& y) { return x - y; }
  };

  template <Device dev, typename T>
  struct Multiplies {
    __host__ __device__ T operator() (const T& x, const T& y) { return x * y; }
  };

  template <Device dev, typename T>
  struct Divides {
    __host__ __device__ T operator() (const T& x, const T& y) { return x / y; }
  };

  template <Device dev, typename T>
  struct Pow {
    __host__ __device__ T operator() (const T& x, const T& y) { return pow(x, y); }
  };

  template <Device dev, typename T>
  struct Left {
    __host__ __device__ T operator() (const T& x, const T&) { return x; }
  };

  template <Device dev, typename T>
  struct Right {
    __host__ __device__ T operator() (const T&, const T& y) { return y; }
  };


  // operator(x)

  template <Device dev, typename T>
  struct Log {
    __host__ __device__ T operator() (const T& x) { return log(x); }
  };

  template <Device dev, typename T>
  struct Exp {
    __host__ __device__ T operator() (const T& x) { return exp(x); }
  };

  template <Device dev, typename T>
  struct Abs {
    __host__ __device__ T operator() (const T& x) { return fabs(x); }
  };

  template <Device dev, typename T>
  struct Sign {
    __host__ __device__ T operator() (const T& x) {
      return x == 0.0 ? 0.0 : ( (x > 0.0) ? 1.0 : -1.0 );
    }
  };

  template <Device dev, typename T>
  struct Sigmoid {
    __host__ __device__ T operator() (const T& x) {
      return 1.0 / (1.0 + exp(-x));
    }
  };

  template <Device dev, typename T>
  struct Dsigmoid {
    __host__ __device__ T operator() (const T& x) {
      T s = Sigmoid<dev, T>()(x);
      return s * (1 - s);
    }
  };

  template <Device dev, typename T>
  struct Tanh {
    __host__ __device__ T operator() (const T& x) {
      // same as octave, avoid nan
      if (x > 10) {
        return 1;
      } else if (x < -10) {
        return -1;
      } else {
        T a = exp(x);
        T b = 1.0 / a;
        return (a - b) / (a + b);
      }
    }
  };

  template <Device dev, typename T>
  struct Dtanh {
    __host__ __device__ T operator() (const T& x) {
      T t = Tanh<dev, T>()(x);
      return 1 - t * t;
    }
  };


  // x = pre(x)
  // y = pre(y)
  // operator(x, y)

  template <Device dev, typename T>
  struct reduce_sum {
    __host__ __device__ T pre (const T& x) { return x; }
    __host__ __device__ T operator() (const T& x, const T& y) { return x + y; }
  };

  template <Device dev, typename T>
  struct reduce_max {
    __host__ __device__ T pre (const T& x) { return x; }
    __host__ __device__ T operator() (const T& x, const T& y) { return fmax(x, y); }
  };

  template <Device dev, typename T>
  struct reduce_min {
    __host__ __device__ T pre (const T& x) { return x; }
    __host__ __device__ T operator() (const T& x, const T& y) { return fmin(x, y); }
  };

  template <Device dev, typename T>
  struct reduce_sum_of_abs {
    __host__ __device__ T pre (const T& x) { return fabs(x); }
    __host__ __device__ T operator() (const T& x, const T& y) { return x + y; }
  };

  template <Device dev, typename T>
  struct reduce_sum_of_squared {
    __host__ __device__ T pre (const T& x) { return x * x; }
    __host__ __device__ T operator() (const T& x, const T& y) { return x + y; }
  };


}


}


#include "../inc/kernel.hpp"


#endif // DEVICE_HPP
