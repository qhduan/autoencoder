#ifndef MATSUB_HPP
#define MATSUB_HPP
/*!
 * \file mateach.hpp
 * \brief
 * \author qhduan.com
 * \date 2015-10-16
*/

namespace tensor {

  template <Device dev, typename T>
  class MatSub {
  public:

    MatSub (T* elem, int rows, int cols, int pitch)
    : m_(elem, rows, cols, pitch) {}

    MatSub (const MatSub& ms)
    : MatSub(ms.m_.elem_, ms.m_.n_rows, ms.m_.n_cols, ms.m_.pitch_) {}

    operator const Mat<dev, T> () const {
      return this->m_;
    }

    MatSub& operator= (const Mat<dev, T>& m) {
      this->m_ = m;
      return *this;
    }

    MatSub& operator= (const MatSub& ms) {
      this->m_ = ms.m_;
      return *this;
    }

    friend bool operator== (const MatSub& a, const MatSub& b) {
      return a.m_ == b.m_;
    }

    ~MatSub () {
      this->m_.elem_ = NULL;
    }

  private:

    Mat<dev, T> m_;
  };

}

#endif // MATSUB_HPP
