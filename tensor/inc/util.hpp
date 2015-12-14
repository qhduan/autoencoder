#ifndef UTIL_HPP
#define UTIL_HPP
/*!
 * \file util.hpp
 * \brief
 * \author qhduan.com
 * \date 2015-10-16
 */

namespace tensor {


  /*!
   * \brief Row * Col = value
   */
  template <Device dev, typename T>
  T operator * (const Row<dev, T>& r, const Col<dev, T>& c) {
    if (r.n_elem != c.n_elem || r.n_elem <= 0 || c.n_elem <= 0) {
      std::cerr << "incompatible dimensions " << r.n_rows << "x" << r.n_cols
                << " * " << c.n_rows << "x" << c.n_cols << std::endl;
      throw std::runtime_error("Row * Col incompatible dimensions");
    }
    return TensorOperation<dev>::template dot<T>(
      r.n_elem, r.memptr(), r.inc(), c.memptr(), c.inc()
    );
  }


  /*!
   * \brief Col * Row = Mat
   */
  template <Device dev, typename T>
  Mat<dev, T> operator * (const Col<dev, T>& c, const Row<dev, T>& r) {
    Mat<dev, T> m(c.size_, r.size_);

    if (c.n_elem <= 0 || r.n_elem <= 0) {
      std::cerr << "incompatible dimensions " << c.n_rows << "x" << c.n_cols
                << " * " << r.n_rows << "x" << r.n_cols << std::endl;
      throw std::runtime_error("Col * row incompatible dimensions");
    }
    /*
      The 8th param c.n_rows is lda for BLAS
      we use column-major, so lda, in fact, not work
      but it have to be bigger then m.n_rows, otherwise BLAS will throw error
    */
    TensorOperation<dev>::template gemm<T>(
      false, false, m.n_rows, m.n_cols, 1,
      1, c.memptr(), c.n_rows, r.memptr(), r.inc(), 0, m.memptr(), m.pitch()
    );
    return m;
  }


  /*!
   * \brief Mat * Col = Col
   */
  template <Device dev, typename T>
  Col<dev, T> operator * (const Mat<dev, T>& m, const Col<dev, T>& c) {
    if (m.n_cols != c.n_elem || c.n_elem == 0) {
      std::cerr << "incompatible dimensions " << m.n_rows << "x" << m.n_cols
                << " * " << c.n_rows << "x" << c.n_cols << std::endl;
      throw std::runtime_error("Mat * Col incompatible dimensions");
    }
    Col<dev, T> ret(m.n_rows);
    /*
      The 10th param c.size_ is ldb for BLAS
      The 13th param ret.size_ is ldc for BLAS
      Both of them not work, but have to set a big value
    */
    /*
    TensorOperation<dev>::template gemm<T>(
      false, false, ret.n_rows, ret.n_cols, m.n_cols,
      1, m.memptr(), m.pitch(), c.memptr(), c.n_elem, 0, ret.memptr(), ret.n_elem
    );
    */
    TensorOperation<dev>::template gemv<T>(
      false, m.n_rows, m.n_cols, 1, m.memptr(), m.pitch(),
      c.memptr(), c.inc(), 0, ret.memptr(), 1
    );
    return ret;
  }


  /*!
   * \brief Row * Mat = Row
   */
  template <Device dev, typename T>
  Row<dev, T> operator * (const Row<dev, T>& r, const Mat<dev, T>& m) {
    if (r.n_elem != m.n_rows || r.n_elem == 0) {
      std::cerr << "incompatible dimensions " << r.n_rows << "x" << r.n_cols
                << " * " << m.n_rows << "x" << m.n_cols << std::endl;
      throw std::runtime_error("Row * Mat incompatible dimensions");
    }
    Row<dev, T> ret(m.n_cols);
    /*
      The 13th param ret.size_ is ldc for BLAS
      It makes ret is a fake matrix
    */
    TensorOperation<dev>::template gemm<T>(
      false, false, ret.n_rows, ret.n_cols, m.n_rows,
      1, r.memptr(), 1, m.memptr(), m.pitch(), 0, ret.memptr(), ret.n_elem
    );
    return ret;
  }


  /* \brief Matrix multiplication of two matrices */
  template <Device dev, typename T>
  Mat<dev, T> operator * (const Mat<dev, T>& m0, const Mat<dev, T>& m1) {
    if (m0.n_cols != m1.n_rows || m0.n_elem == 0 || m1.n_elem == 0) {
      std::cerr << "incompatible dimensions " << m0.n_rows << "x" << m0.n_cols
                << " * " << m1.n_rows << "x" << m1.n_cols << std::endl;
      throw std::runtime_error("Mat * Mat incompatible dimensions");
    }

    Mat<dev, T> m(m0.n_rows, m1.n_cols);

    TensorOperation<dev>::template gemm<T>(
      false, false, m0.n_rows, m1.n_cols, m0.n_cols,
      1, m0.memptr(), m0.pitch(), m1.memptr(), m1.pitch(), 0, m.memptr(), m.pitch()
    );

    return m;
  }

  template <Device dev, typename T>
  Mat<dev, T> pow (const Mat<dev, T>& m, const float& p) {
    return Mat<dev, T>(m).pow(static_cast<T>(p));
  }

  template <Device dev, typename T>
  Mat<dev, T> pow (const Mat<dev, T>& m, const double& p) {
    return Mat<dev, T>(m).pow(static_cast<T>(p));
  }


  /*!
   * \brief
   *   For vector, return the sum of all elements
   *   For matrix, dim == 0 return the sum of each column (return a Row like)
   *   dim == 1 return the sum of each row (return a Col like).
   *   To get a sum of all the elements, please use accu()
  */
  template <Device dev, typename T>
  Mat<dev, T> sum (const Mat<dev, T>& m, int dim = 0) {
    if (dim == 0) { // return row
      Mat<dev, T> row(1, m.n_cols);
      TensorOperation<dev>::template A_each_col_reduce_to_x<T, functor::reduce_sum<dev, T> >(
        m.n_rows, m.n_cols, m.memptr(), m.pitch(), row.memptr(), 1
      );
      return row;
    } else if (dim == 1) { // return column
      Mat<dev, T> col(m.n_rows, 1);
      TensorOperation<dev>::template A_each_row_reduce_to_x<T, functor::reduce_sum<dev, T> >(
        m.n_rows, m.n_cols, m.memptr(), m.pitch(), col.memptr(), 1
      );
      return col;
    } else {
      throw std::runtime_error("Mat sum() invalid dim");
    }
  }
  template <Device dev, typename T>
  T sum (const Vec<dev, T>& v) {
    return accu(v);
  }


  /*
   * \brief Return the sum of all the elements
   */
  template <Device dev, typename T>
  T accu (const Mat<dev, T>& m) {
    return TensorOperation<dev>::template A_reduce<float, functor::reduce_sum<dev, float> >(
      m.n_rows, m.n_cols, m.memptr(), m.pitch()
    );
  }
  template <Device dev, typename T>
  T accu (const Vec<dev, T>& v) {
    return TensorOperation<dev>::template A_reduce<float, functor::reduce_sum<dev, float> >(
      v.n_elem, v.memptr(), v.inc()
    );
  }


  /*!
   * \brief
   *   For vector, return the sum of all elements
   *   For matrix, dim == 0 return the sum of each column (return a Row like)
   *   dim == 1 return the sum of each row (return a Col like).
   *   To get a sum of all the elements, please use accu()
  */
  template <Device dev, typename T>
  Mat<dev, T> max (const Mat<dev, T>& m, int dim) {
    if (dim == 0) { // return row
      Mat<dev, T> row(1, m.n_cols);
      TensorOperation<dev>::template A_each_col_reduce_to_x<T, functor::reduce_max<dev, T> >(
        m.n_rows, m.n_cols, m.memptr(), m.pitch(), row.memptr(), 1
      );
      return row;
    } else if (dim == 1) { // return column
      Mat<dev, T> col(m.n_rows, 1);
      TensorOperation<dev>::template A_each_row_reduce_to_x<T, functor::reduce_max<dev, T> >(
        m.n_rows, m.n_cols, m.memptr(), m.pitch(), col.memptr(), 1
      );
      return col;
    } else {
      throw std::runtime_error("Mat max(Mat, dim) invalid dim");
    }
  }
  template <Device dev, typename T>
  T max (const Mat<dev, T>& m) {
   return TensorOperation<dev>::template A_reduce<float, functor::reduce_max<dev, float> >(
     m.n_rows, m.n_cols, m.memptr(), m.pitch()
   );
  }
  template <Device dev, typename T>
  T max (const Vec<dev, T>& v) {
   return TensorOperation<dev>::template A_reduce<float, functor::reduce_max<dev, float> >(
     v.n_elem, v.memptr(), v.inc()
   );
  }




  /*!
   * \brief
   *   For vector, return the sum of all elements
   *   For matrix, dim == 0 return the sum of each column (return a Row like)
   *   dim == 1 return the sum of each row (return a Col like).
   *   To get a sum of all the elements, please use accu()
  */
  template <Device dev, typename T>
  Mat<dev, T> min (const Mat<dev, T>& m, int dim) {
    if (dim == 0) { // return row
      Mat<dev, T> row(1, m.n_cols);
      TensorOperation<dev>::template A_each_col_reduce_to_x<T, functor::reduce_min<dev, T> >(
        m.n_rows, m.n_cols, m.memptr(), m.pitch(), row.memptr(), 1
      );
      return row;
    } else if (dim == 1) { // return column
      Mat<dev, T> col(m.n_rows, 1);
      TensorOperation<dev>::template A_each_row_reduce_to_x<T, functor::reduce_min<dev, T> >(
        m.n_rows, m.n_cols, m.memptr(), m.pitch(), col.memptr(), 1
      );
      return col;
    } else {
      throw std::runtime_error("Mat min(Mat, dim) invalid dim");
    }
  }
  template <Device dev, typename T>
  T min (const Mat<dev, T>& m) {
   return TensorOperation<dev>::template A_reduce<float, functor::reduce_min<dev, float> >(
     m.n_rows, m.n_cols, m.memptr(), m.pitch()
   );
  }
  template <Device dev, typename T>
  T min (const Vec<dev, T>& v) {
   return TensorOperation<dev>::template A_reduce<float, functor::reduce_min<dev, float> >(
     v.n_elem, v.memptr(), v.inc()
   );
  }


  template <Device dev, typename T>
  T norm (const Mat<dev, T>& m) {
    return std::sqrt(TensorOperation<dev>::template A_reduce<float, functor::reduce_sum_of_squared<dev, float> >(
      m.n_rows, m.n_cols, m.memptr(), m.pitch()
    ));
  }
  template <Device dev, typename T>
  T norm (const Vec<dev, T>& v) {
    return std::sqrt(TensorOperation<dev>::template A_reduce<float, functor::reduce_sum_of_squared<dev, float> >(
      v.n_elem, v.memptr(), v.inc()
    ));
  }


  template <Device dev, typename T>
  Mat<dev, T> sigmoid (const Mat<dev, T>& m) {
    return Mat<dev, T>(m).sigmoid();
  }


  template <Device dev, typename T>
  Mat<dev, T> dsigmoid (const Mat<dev, T>& m) {
    return Mat<dev, T>(m).dsigmoid();
  }


  template <Device dev, typename T>
  Mat<dev, T> abs (const Mat<dev, T>& m) {
    return Mat<dev, T>(m).abs();
  }
  template <Device dev, typename T>
  Col<dev, T> abs (const Col<dev, T>& c) {
    return Col<dev, T>(c).abs();
  }
  template <Device dev, typename T>
  Row<dev, T> abs (const Row<dev, T>& r) {
    return Row<dev, T>(r).abs();
  }


  template <Device dev, typename T>
  Mat<dev, T> log (const Mat<dev, T>& m) {
    return Mat<dev, T>(m).log();
  }
  template <Device dev, typename T>
  Col<dev, T> log (const Col<dev, T>& c) {
    return Col<dev, T>(c).log();
  }
  template <Device dev, typename T>
  Row<dev, T> log (const Row<dev, T>& r) {
    return Row<dev, T>(r).log();
  }


  template <Device dev, typename T>
  Mat<dev, T> exp (const Mat<dev, T>& m) {
    return Mat<dev, T>(m).exp();
  }
  template <Device dev, typename T>
  Col<dev, T> exp (const Col<dev, T>& c) {
    return Col<dev, T>(c).exp();
  }
  template <Device dev, typename T>
  Row<dev, T> exp (const Row<dev, T>& r) {
    return Row<dev, T>(r).exp();
  }


  template <Device dev, typename T>
  Mat<dev, T> reshape (const Mat<dev, T>& m, int rows, int cols) {
    if (m.n_rows == rows && m.n_cols == cols) {
      return m;
    }
    Mat<dev, T> t(m);
    t.reshape(rows, cols);
    return t;
  }


  template <typename T>
  T randu (int rows, int cols) {
    T t(rows, cols);
    t.randu();
    return t;
  }
  template <typename T>
  T randn (int rows, int cols) {
    T t(rows, cols);
    t.randn();
    return t;
  }


  template <typename T>
  T ones (int rows, int cols) {
    T t(rows, cols);
    t.ones();
    return t;
  }


  template <typename T>
  T zeros (int rows, int cols) {
    T t(rows, cols);
    t.zeros();
    return t;
  }


  template <typename T>
  T eye (int rows, int cols) {
    T t(rows, cols);
    t.eye();
    return t;
  }


} // namespace tensor

#endif //UTIL_HPP
