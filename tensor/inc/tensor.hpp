#ifndef TENSOR_HPP
#define TENSOR_HPP
/*!
 * \file tensor.hpp
 * \brief
 * \author qhduan.com
 * \date 2015-10-16
 */

#include <vector>
#include <initializer_list>
#include <cstdarg>
#include <cmath>
#include <functional>
#include <random>
#include <algorithm>
#include <string>
#include <sstream>
#include <iomanip>

#include "../inc/operation.h"
#include "../inc/mempool.h"
#include "../inc/tool.h"

namespace tensor {

  template <Device dev, typename T>
  class MatTranspose;

  template <Device dev, typename T>
  class MatEachRow;

  template <Device dev, typename T>
  class MatEachCol;

  template <Device dev, typename T>
  class MatSub;

  /* \brief dense matrix class */
  template <Device dev, typename T>
  class Mat;

  /* \brief base class for Row and Col */
  template <Device dev, typename T>
  class Vec;

  /* \brief dense row vector class */
  template <Device dev, typename T>
  class Row;

  /* \brief dense column vector class */
  template <Device dev, typename T>
  class Col;


  namespace fill {
    enum type {
      zeros,
      ones,
      eye,
      randu,
      randn,
      none
    };
  };

  /*!
   * \brief ValueProxy class
   *    make Mat(m, n) could be assigned
   *    eg. Mat a(3, 3); a(1, 1) = 4.0;
   */
  template <Device dev, typename T>
  class ValueProxy {
  public:
    /*!
     * \brief init ValueProxy with a pointer (host or device), and Device type
     */
    ValueProxy (T* const pointer) {
      this->pointer_ = pointer;
      this->value_ = TensorOperation<dev>::template get<T>(pointer);
    }

    ValueProxy& operator= (const T& val) {
      this->value_ = val;
      TensorOperation<dev>::template set<T>(this->pointer_, val);
      return *this;
    }

    operator T () {
      return this->value_;
    }
  private:
    T value_;
    T* pointer_;
  };

} // namespace tensor

#include "../inc/vec.hpp"
#include "../inc/mateach.hpp"
#include "../inc/matsub.hpp"
#include "../inc/mat.hpp"
#include "../inc/col.hpp"
#include "../inc/row.hpp"
#include "../inc/util.hpp"

#include "../inc/tool.h"

#endif //TENSOR_HPP
