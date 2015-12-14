#ifndef COL_HPP
#define COL_HPP
/*!
 * \file tensor.hpp
 * \brief
 * \author qhduan.com
 * \date 2015-10-16
 */

namespace tensor {

  /* \brief dense column vector class */
  template <Device dev, typename T>
  class Col: public Vec<dev, T> {
  public:

    Col () : Vec<dev, T>() {}

    Col (int size)
    : Vec<dev, T>(size) {
      this->rows_ = size;
      this->cols_ = 1;
    }

    Col (const std::vector<T>& vec)
    : Vec<dev, T>(vec) {
      this->rows_ = vec.size();
      this->cols_ = 1;
    }

    Col (std::initializer_list<T> args)
    : Col(std::vector<T>(args)) {}

    Col (const char* str)
    : Col(std::string(str)) {}

    Col (const std::string& str)
    : Vec<dev, T>(str) {
      this->rows_ = this->n_elem;
      this->cols_ = 1;
    }

    Col (const Mat<dev, T>& m)
    : Col() {
      if (m.n_cols != 1) {
        throw std::runtime_error("Col(Mat) invalid argument, mat should have only one column");
      }
      this->rows_ = m.n_rows;
      this->cols_ = 1;
      this->size_ = this->n_rows * this->n_cols;
      this->inc_ = 1;
      this->elem_ = Mempool<dev>::instance().template malloc<T>(this->n_elem);
      TensorOperation<dev>::template A_to_B<T>(
        this->n_rows, this->n_cols,
        m.memptr(), m.pitch(), this->n_elem, this->inc()
      );
    }

    Col (Col&& c)
    : Col() {
      this->swap(c);
    }

    Col (const Col& c)
    : Col() {
      this->rows_ = c.n_rows;
      this->cols_ = c.n_cols;
      this->size_ = c.n_elem;
      this->inc_ = 1;
      this->elem_ = Mempool<dev>::instance().template malloc<T>(this->n_elem);
      TensorOperation<dev>::template A_to_B<T>(
        this->n_rows, this->n_cols,
        c.memptr(), c.inc(), this->n_elem, this->inc()
      );
    }

    ~Col () {}

    Col& operator= (Col&& c) {
      this->swap(c);
      return *this;
    }

    Col& operator= (const Col& c) {
      if (this->size_ == c.size_) {
        TensorOperation<dev>::template A_to_B<T>(
          this->n_rows, this->n_cols,
          c.memptr(), c.inc(), this->memptr(), this->inc()
        );
      } else {
        this->release();
        this->rows_ = c.n_rows;
        this->cols_ = c.n_cols;
        this->size_ = c.n_elem;
        this->inc_ = 1;
        this->proxy_ = false;
        this->elem_ = Mempool<dev>::instance().template malloc<T>(this->n_elem);
        TensorOperation<dev>::template A_to_B<T>(
          this->n_rows, this->n_cols,
          c.memptr(), c.inc(), this->memptr(), this->inc()
        );
      }
      return *this;
    }

    friend std::ostream& operator << (std::ostream& os, const Col<dev, T>& r) {
      r.print(os);
      return os;
    }

    void print (std::ostream& os, const std::string& header) const {
      if (header.length() > 0) {
        std::cout << header << std::endl;
      }
      auto p = os.precision();
      auto f = os.flags();
      os.precision(4);
      os.setf(std::ios::fixed, std:: ios::floatfield);

      for (int i = 0; i < this->n_elem; i++) {
        T value = (*this)(i);
        if (value >= 0) {
          os << "   " << value;
        } else {
          os << "  " << value;
        }
        if (i != (this->n_elem - 1)) {
          os << std::endl;
        }
      }

      os.precision(p);
      os.flags(f);
    }

    void print(const std::string& header) const {
      this->print(std::cout, header);
    }

    void print(std::ostream& os) const {
      this->print(os, std::string(""));
    }

    void print() const {
      this->print(std::cout, std::string(""));
    }

    friend Col operator + (const T& value, const Col& c0) {
      return c0 + value;
    }

    friend Col operator - (const T& value, const Col& c0) {
      Col c(c0.size_);
      TensorOperation<dev>::template val_op_x_to_y<T, functor::Minus<dev,T> >(
        c0.size_, value, c0.memptr(), c0.inc(), c.memptr(), c.inc()
      );
      return c;
    }

    friend Col operator % (const T& value, const Col& c0) {
      return c0 % value;
    }

    friend Col operator / (const T& value, const Col& c0) {
      Col c(c0.size_);
      TensorOperation<dev>::template val_op_x_to_y<T, functor::Divides<dev,T> >(
        c0.size_, value, c0.memptr(), c0.inc(), c.memptr(), c.inc()
      );
      return c;
    }


    friend Col operator + (const Col& c0, const T& value) {
      Col c(c0);
      c += value;
      return c;
    }

    friend Col operator + (const Col& c0, const Col& r1) {
      Col c(c0);
      c += r1;
      return c;
    }

    friend Col operator - (const Col& c0, const T& value) {
      Col c(c0);
      c -= value;
      return c;
    }

    friend Col operator - (const Col& c0, const Col& r1) {
      Col c(c0);
      c -= r1;
      return c;
    }

    friend Col operator % (const Col& c0, const T& value) {
      Col c(c0);
      c %= value;
      return c;
    }

    friend Col operator % (const Col& c0, const Col& r1) {
      Col c(c0);
      c %= r1;
      return c;
    }

    friend Col operator / (const Col& c0, const T& value) {
      Col c(c0);
      c /= value;
      return c;
    }

    friend Col operator / (const Col& c0, const Col& r1) {
      Col c(c0);
      c /= r1;
      return c;
    }

  };

} // namespace tensor

#endif //COL_HPP
