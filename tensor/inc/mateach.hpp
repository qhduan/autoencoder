#ifndef MATEACH_HPP
#define MATEACH_HPP
/*!
 * \file mateach.hpp
 * \brief
 * \author qhduan.com
 * \date 2015-10-16
*/

namespace tensor {

  template <Device dev, typename T>
  class MatTranspose {
  public:
    MatTranspose (const Mat<dev, T>& m)
    : m_(m) {}

    operator Mat<dev, T> () {
      Mat<dev, T> m(this->m_.n_cols, this->m_.n_rows);
      TensorOperation<dev>::matrix_call(trans_A_to_B<dev, T>({
        this->m_.n_rows, this->m_.n_cols, this->m_.memptr(), this->m_.pitch(),
        m.memptr(), m.pitch()
      }));
      /*
      TensorOperation<dev>::template A_tran_to_B<T>(
        this->m_.memptr(), this->m_.n_rows, this->m_.n_cols, this->m_.pitch(),
        m.memptr(), m.n_rows, m.n_cols, m.pitch()
      );
      */
      return m;
    }

    friend Mat<dev, T> operator * (const MatTranspose<dev, T>& mt, const Mat<dev, T>& m1) {
      const Mat<dev, T>& m0 = mt.m_;
      Mat<dev, T> m(m0.n_cols, m1.n_cols);
      TensorOperation<dev>::template gemm<T>(
        true, false, m.n_rows, m.n_cols, m0.n_rows,
        1, m0.memptr(), m0.pitch(), m1.memptr(), m1.pitch(), 0, m.memptr(), m.pitch()
      );
      return m;
    }

    friend Mat<dev, T> operator * (const Mat<dev, T>& m0, const MatTranspose<dev, T>& mt) {
      const Mat<dev, T>& m1 = mt.m_;
      Mat<dev, T> m(m0.n_rows, m1.n_rows);
      TensorOperation<dev>::template gemm<T>(
        false, true, m.n_rows, m.n_cols, m0.n_cols,
        1, m0.memptr(), m0.pitch(), m1.memptr(), m1.pitch(), 0, m.memptr(), m.pitch()
      );
      return m;
    }

  private:
    const Mat<dev, T>& m_;
  };

  template <Device dev, typename T>
  class MatEachRow {
  public:
    MatEachRow (Mat<dev, T>& m)
    : m_(m) {}

    MatEachRow& operator += (const Mat<dev, T>& m) {
      if (m.n_cols == this->m_.n_cols && m.n_rows == 1) {
        TensorOperation<dev>::matrix_call(A_each_row_op_x_to_B<dev, T, functor::Plus<dev, T> >({
          this->m_.n_rows, this->m_.n_cols, this->m_.memptr(), this->m_.pitch(),
          m.memptr(), m.pitch(), this->m_.memptr(), this->m_.pitch()
        }));
      } else {
        std::cerr << this->m_.n_rows << "x" << this->m_.n_cols << "\n"
                  << m.n_rows << "x" << m.n_cols << (m.n_cols == this->m_.n_cols) << (m.n_rows == 1) << std::endl;
        throw std::runtime_error("Mat::each_row() += invalid mat");
      }
      return *this;
    }

    MatEachRow& operator += (const Row<dev, T>& r) {
      if (r.n_cols == this->m_.n_cols && r.n_rows == 1) {
        TensorOperation<dev>::matrix_call(A_each_row_op_x_to_B<dev, T, functor::Plus<dev, T> >({
          this->m_.n_rows, this->m_.n_cols, this->m_.memptr(), this->m_.pitch(),
          r.memptr(), r.inc(), this->m_.memptr(), this->m_.pitch()
        }));
      } else {
        std::cerr << this->m_.n_rows << "x" << this->m_.n_cols << "\n"
                  << r.n_rows << "x" << r.n_cols << std::endl;
        throw std::runtime_error("Mat::each_row() += invalid row");
      }
      return *this;
    }

    MatEachRow& operator -= (const Mat<dev, T>& m) {
      if (m.n_cols == this->m_.n_cols && m.n_rows == 1) {
        TensorOperation<dev>::matrix_call(A_each_row_op_x_to_B<dev, T, functor::Minus<dev, T> >({
          this->m_.n_rows, this->m_.n_cols, this->m_.memptr(), this->m_.pitch(),
          m.memptr(), m.pitch(), this->m_.memptr(), this->m_.pitch()
        }));
      } else {
        std::cerr << this->m_.n_rows << "x" << this->m_.n_cols << "\n"
                  << m.n_rows << "x" << m.n_cols << std::endl;
        throw std::runtime_error("Mat::each_row() -= invalid mat");
      }
      return *this;
    }

    MatEachRow& operator -= (const Row<dev, T>& r) {
      if (r.n_cols == this->m_.n_cols && r.n_rows == 1) {
        TensorOperation<dev>::matrix_call(A_each_row_op_x_to_B<dev, T, functor::Minus<dev, T> >({
          this->m_.n_rows, this->m_.n_cols, this->m_.memptr(), this->m_.pitch(),
          r.memptr(), r.inc(), this->m_.memptr(), this->m_.pitch()
        }));
      } else {
        std::cerr << this->m_.n_rows << "x" << this->m_.n_cols << "\n"
                  << r.n_rows << "x" << r.n_cols << std::endl;
        throw std::runtime_error("Mat::each_row() -= invalid row");
      }
      return *this;
    }

    MatEachRow& operator %= (const Mat<dev, T>& m) {
      if (m.n_cols == this->m_.n_cols && m.n_rows == 1) {
        TensorOperation<dev>::matrix_call(A_each_row_op_x_to_B<dev, T, functor::Multiplies<dev, T> >({
          this->m_.n_rows, this->m_.n_cols, this->m_.memptr(), this->m_.pitch(),
          m.memptr(), m.pitch(), this->m_.memptr(), this->m_.pitch()
        }));
      } else {
        std::cerr << this->m_.n_rows << "x" << this->m_.n_cols << "\n"
                  << m.n_rows << "x" << m.n_cols << std::endl;
        throw std::runtime_error("Mat::each_row() %= invalid mat");
      }
      return *this;
    }

    MatEachRow& operator %= (const Row<dev, T>& r) {
      if (r.n_cols == this->m_.n_cols && r.n_rows == 1) {
        TensorOperation<dev>::matrix_call(A_each_row_op_x_to_B<dev, T, functor::Multiplies<dev, T> >({
          this->m_.n_rows, this->m_.n_cols, this->m_.memptr(), this->m_.pitch(),
          r.memptr(), r.inc(), this->m_.memptr(), this->m_.pitch()
        }));
      } else {
        std::cerr << this->m_.n_rows << "x" << this->m_.n_cols << "\n"
                  << r.n_rows << "x" << r.n_cols << std::endl;
        throw std::runtime_error("Mat::each_row() %= invalid row");
      }
      return *this;
    }

    MatEachRow& operator /= (const Mat<dev, T>& m) {
      if (m.n_cols == this->m_.n_cols && m.n_rows == 1) {
        TensorOperation<dev>::matrix_call(A_each_row_op_x_to_B<dev, T, functor::Divides<dev, T> >({
          this->m_.n_rows, this->m_.n_cols, this->m_.memptr(), this->m_.pitch(),
          m.memptr(), m.pitch(), this->m_.memptr(), this->m_.pitch()
        }));
      } else {
        std::cerr << this->m_.n_rows << "x" << this->m_.n_cols << "\n"
                  << m.n_rows << "x" << m.n_cols << std::endl;
        throw std::runtime_error("Mat::each_row() /= invalid mat");
      }
      return *this;
    }

    MatEachRow& operator /= (const Row<dev, T>& r) {
      if (r.n_cols == this->m_.n_cols && r.n_rows == 1) {
        TensorOperation<dev>::matrix_call(A_each_row_op_x_to_B<dev, T, functor::Divides<dev, T> >({
          this->m_.n_rows, this->m_.n_cols, this->m_.memptr(), this->m_.pitch(),
          r.memptr(), r.inc(), this->m_.memptr(), this->m_.pitch()
        }));
      } else {
        std::cerr << this->m_.n_rows << "x" << this->m_.n_cols << "\n"
                  << r.n_rows << "x" << r.n_cols << std::endl;
        throw std::runtime_error("Mat::each_row() /= invalid row");
      }
      return *this;
    }

    MatEachRow& operator = (const Mat<dev, T>& m) {
      if (m.n_cols == this->m_.n_cols && m.n_rows == 1) {
        TensorOperation<dev>::matrix_call(A_each_row_op_x_to_B<dev, T, functor::Right<dev, T> >({
          this->m_.n_rows, this->m_.n_cols, this->m_.memptr(), this->m_.pitch(),
          m.memptr(), m.pitch(), this->m_.memptr(), this->m_.pitch()
        }));
      } else {
        std::cerr << this->m_.n_rows << "x" << this->m_.n_cols << "\n"
                  << m.n_rows << "x" << m.n_cols << (m.n_cols == this->m_.n_cols) << (m.n_rows == 1) << std::endl;
        throw std::runtime_error("Mat::each_row() = invalid mat");
      }
      return *this;
    }

    MatEachRow& operator = (const Row<dev, T>& r) {
      if (r.n_cols == this->m_.n_cols && r.n_rows == 1) {
        TensorOperation<dev>::matrix_call(A_each_row_op_x_to_B<dev, T, functor::Right<dev, T> >({
          this->m_.n_rows, this->m_.n_cols, this->m_.memptr(), this->m_.pitch(),
          r.memptr(), r.inc(), this->m_.memptr(), this->m_.pitch()
        }));
      } else {
        std::cerr << this->m_.n_rows << "x" << this->m_.n_cols << "\n"
                  << r.n_rows << "x" << r.n_cols << std::endl;
        throw std::runtime_error("Mat::each_row() = invalid row");
      }
      return *this;
    }

  private:
    Mat<dev, T>& m_;
  };

  template <Device dev, typename T>
  class MatEachCol {
  public:
    MatEachCol (Mat<dev, T>& m)
    : m_(m) {}

    MatEachCol& operator += (const Mat<dev, T>& m) {
      if (m.n_cols == 1 && m.n_rows == this->m_.n_rows) {
        TensorOperation<dev>::matrix_call(A_each_col_op_x_to_B<dev, T, functor::Plus<dev, T> >({
          this->m_.n_rows, this->m_.n_cols, this->m_.memptr(), this->m_.pitch(),
          m.memptr(), 1, this->m_.memptr(), this->m_.pitch()
        }));
      } else {
        std::cerr << this->m_.n_rows << "x" << this->m_.n_cols << "\n"
                  << m.n_rows << "x" << m.n_cols << std::endl;
        throw std::runtime_error("Mat::each_col() += invalid mat");
      }
      return *this;
    }

    MatEachCol& operator += (const Col<dev, T>& c) {
      if (c.n_cols == 1 && c.n_rows == this->m_.n_rows) {
        TensorOperation<dev>::matrix_call(A_each_col_op_x_to_B<dev, T, functor::Plus<dev, T> >({
          this->m_.n_rows, this->m_.n_cols, this->m_.memptr(), this->m_.pitch(),
          c.memptr(), 1, this->m_.memptr(), this->m_.pitch()
        }));
      } else {
        std::cerr << this->m_.n_rows << "x" << this->m_.n_cols << "\n"
                  << c.n_rows << "x" << c.n_cols << std::endl;
        throw std::runtime_error("Mat::each_col() += invalid col");
      }
      return *this;
    }

    MatEachCol& operator -= (const Mat<dev, T>& m) {
      if (m.n_cols == 1 && m.n_rows == this->m_.n_rows) {
        TensorOperation<dev>::matrix_call(A_each_col_op_x_to_B<dev, T, functor::Minus<dev, T> >({
          this->m_.n_rows, this->m_.n_cols, this->m_.memptr(), this->m_.pitch(),
          m.memptr(), 1, this->m_.memptr(), this->m_.pitch()
        }));
      } else {
        std::cerr << this->m_.n_rows << "x" << this->m_.n_cols << "\n"
                  << m.n_rows << "x" << m.n_cols << std::endl;
        throw std::runtime_error("Mat::each_col() -= invalid mat");
      }
      return *this;
    }

    MatEachCol& operator -= (const Col<dev, T>& c) {
      if (c.n_cols == 1 && c.n_rows == this->m_.n_rows) {
        TensorOperation<dev>::matrix_call(A_each_col_op_x_to_B<dev, T, functor::Minus<dev, T> >({
          this->m_.n_rows, this->m_.n_cols, this->m_.memptr(), this->m_.pitch(),
          c.memptr(), 1, this->m_.memptr(), this->m_.pitch()
        }));
      } else {
        std::cerr << this->m_.n_rows << "x" << this->m_.n_cols << "\n"
                  << c.n_rows << "x" << c.n_cols << std::endl;
        throw std::runtime_error("Mat::each_col() -= invalid col");
      }
      return *this;
    }

    MatEachCol& operator %= (const Mat<dev, T>& m) {
      if (m.n_cols == 1 && m.n_rows == this->m_.n_rows) {
        TensorOperation<dev>::matrix_call(A_each_col_op_x_to_B<dev, T, functor::Multiplies<dev, T> >({
          this->m_.n_rows, this->m_.n_cols, this->m_.memptr(), this->m_.pitch(),
          m.memptr(), 1, this->m_.memptr(), this->m_.pitch()
        }));
      } else {
        std::cerr << this->m_.n_rows << "x" << this->m_.n_cols << "\n"
                  << m.n_rows << "x" << m.n_cols << std::endl;
        throw std::runtime_error("Mat::each_col() %= invalid mat");
      }
      return *this;
    }

    MatEachCol& operator %= (const Col<dev, T>& c) {
      if (c.n_cols == 1 && c.n_rows == this->m_.n_rows) {
        TensorOperation<dev>::matrix_call(A_each_col_op_x_to_B<dev, T, functor::Multiplies<dev, T> >({
          this->m_.n_rows, this->m_.n_cols, this->m_.memptr(), this->m_.pitch(),
          c.memptr(), 1, this->m_.memptr(), this->m_.pitch()
        }));
      } else {
        std::cerr << this->m_.n_rows << "x" << this->m_.n_cols << "\n"
                  << c.n_rows << "x" << c.n_cols << std::endl;
        throw std::runtime_error("Mat::each_col() %= invalid col");
      }
      return *this;
    }

    MatEachCol& operator /= (const Mat<dev, T>& m) {
      if (m.n_cols == 1 && m.n_rows == this->m_.n_rows) {
        TensorOperation<dev>::matrix_call(A_each_col_op_x_to_B<dev, T, functor::Divides<dev, T> >({
          this->m_.n_rows, this->m_.n_cols, this->m_.memptr(), this->m_.pitch(),
          m.memptr(), 1, this->m_.memptr(), this->m_.pitch()
        }));
      } else {
        std::cerr << this->m_.n_rows << "x" << this->m_.n_cols << "\n"
                  << m.n_rows << "x" << m.n_cols << std::endl;
        throw std::runtime_error("Mat::each_col() /= invalid mat");
      }
      return *this;
    }

    MatEachCol& operator /= (const Col<dev, T>& c) {
      if (c.n_cols == 1 && c.n_rows == this->m_.n_rows) {
        TensorOperation<dev>::matrix_call(A_each_col_op_x_to_B<dev, T, functor::Divides<dev, T> >({
          this->m_.n_rows, this->m_.n_cols, this->m_.memptr(), this->m_.pitch(),
          c.memptr(), 1, this->m_.memptr(), this->m_.pitch()
        }));
      } else {
        std::cerr << this->m_.n_rows << "x" << this->m_.n_cols << "\n"
                  << c.n_rows << "x" << c.n_cols << std::endl;
        throw std::runtime_error("Mat::each_col() /= invalid col");
      }
      return *this;
    }

    MatEachCol& operator = (const Mat<dev, T>& m) {
      if (m.n_cols == 1 && m.n_rows == this->m_.n_rows) {
        TensorOperation<dev>::matrix_call(A_each_col_op_x_to_B<dev, T, functor::Right<dev, T> >({
          this->m_.n_rows, this->m_.n_cols, this->m_.memptr(), this->m_.pitch(),
          m.memptr(), 1, this->m_.memptr(), this->m_.pitch()
        }));
      } else {
        std::cerr << this->m_.n_rows << "x" << this->m_.n_cols << "\n"
                  << m.n_rows << "x" << m.n_cols << std::endl;
        throw std::runtime_error("Mat::each_col() = invalid mat");
      }
      return *this;
    }

    MatEachCol& operator = (const Col<dev, T>& c) {
      if (c.n_cols == 1 && c.n_rows == this->m_.n_rows) {
        TensorOperation<dev>::matrix_call(A_each_col_op_x_to_B<dev, T, functor::Right<dev, T> >({
          this->m_.n_rows, this->m_.n_cols, this->m_.memptr(), this->m_.pitch(),
          c.memptr(), 1, this->m_.memptr(), this->m_.pitch()
        }));
      } else {
        std::cerr << this->m_.n_rows << "x" << this->m_.n_cols << "\n"
                  << c.n_rows << "x" << c.n_cols << std::endl;
        throw std::runtime_error("Mat::each_col() = invalid col");
      }
      return *this;
    }

  private:
    Mat<dev, T>& m_;
  };

} // namespace tensor

#endif //MATEACH_HPP
