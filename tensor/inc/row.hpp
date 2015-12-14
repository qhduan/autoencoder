#ifndef ROW_HPP
#define ROW_HPP
/*!
 * \file tensor.hpp
 * \brief
 * \author qhduan.com
 * \date 2015-10-16
 */

namespace tensor {

  /* \brief dense row vector class */
  template <Device dev, typename T>
  class Row: public Vec<dev, T> {
  public:

    Row ()
    : Vec<dev, T>() {}

    Row (int size)
    : Vec<dev, T>(size) {
      this->rows_ = 1;
      this->cols_ = size;
    }

    Row (const std::vector<T>& vec)
    : Vec<dev, T>(vec) {
      this->rows_ = 1;
      this->cols_ = vec.size();
    }

    Row (std::initializer_list<T> args)
    : Row(std::vector<T>(args)) {}

    Row (const char* str)
    : Row(std::string(str)) {}

    Row (const std::string& str)
    : Vec<dev, T>(str) {
      this->rows_ = 1;
      this->cols_ = this->n_elem;
    }

    Row (const Mat<dev, T>& m)
    : Vec<dev, T>() {
      if (m.n_rows != 1) {
        throw std::runtime_error("Row(Mat) invalid argument, mat should have only one row");
      }
      this->rows_ = 1;
      this->cols_ = m.n_cols;
      this->size_ = this->n_rows * this->n_cols;
      this->inc_ = 1;
      this->elem_ = Mempool<dev>::instance().template malloc<T>(this->n_elem);
      TensorOperation<dev>::template A_to_B<T>(
        this->n_rows, this->n_cols,
        m.memptr(), m.pitch(), this->memptr(), this->inc()
      );
    }

    Row (Row&& r)
    : Row() {
      this->swap(r);
    }

    Row (const Row& r)
    : Row() {
      this->rows_ = r.n_rows;
      this->cols_ = r.n_cols;
      this->size_ = r.n_elem;
      this->inc_ = 1;
      this->elem_ = Mempool<dev>::instance().template malloc<T>(this->n_elem);
      TensorOperation<dev>::template A_to_B<T>(
        this->n_rows, this->n_cols,
        r.memptr(), r.inc(), this->memptr(), this->inc()
      );
    }

    ~Row () {}

    Row& operator= (Row&& r) {
      this->swap(r);
      return *this;
    }

    Row& operator= (const Row& r) {
      if (this->n_elem == r.n_elem) {
        TensorOperation<dev>::template A_to_B<T>(
          this->n_rows, this->n_cols,
          r.memptr(), r.inc(), this->memptr(), this->inc()
        );
      } else {
        this->release();
        this->rows_ = r.n_rows;
        this->cols_ = r.n_cols;
        this->size_ = r.n_elem;
        this->inc_ = 1;
        this->elem_ = Mempool<dev>::instance().template malloc<T>(this->n_elem);
        TensorOperation<dev>::template A_to_B<T>(
          this->n_rows, this->n_cols,
          r.memptr(), r.inc(), this->memptr(), this->inc()
        );
      }
      return *this;
    }

    friend std::ostream& operator << (std::ostream& os, const Row<dev, T>& r) {
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
      }
      os << std::endl;

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

    friend Row operator + (const T& value, const Row<dev, T>& r0) {
      return r0 + value;
    }

    friend Row operator - (const T& value, const Row<dev, T>& r0) {
      Row r(r0.size_);
      TensorOperation<dev>::template val_op_x_to_y<T, functor::Minus<dev,T> >(
        r0.size_, value, r0.memptr(), r0.inc(), r.memptr(), r.inc()
      );
      return r;
    }

    friend Row operator % (const T& value, const Row<dev, T>& r0) {
      return r0 % value;
    }

    friend Row operator / (const T& value, const Row<dev, T>& r0) {
      Row r(r0.size_);
      TensorOperation<dev>::template val_op_x_to_y<T, functor::Divides<dev,T> >(
        r0.size_, value, r0.memptr(), r0.inc(), r.memptr(), r.inc()
      );
      return r;
    }

    friend Row operator + (const Row<dev, T>& r0, const T& value) {
      Row r(r0);
      r += value;
      return r;
    }

    friend Row operator + (const Row<dev, T>& r0, const Row<dev, T>& r1) {
      Row r(r0);
      r += r1;
      return r;
    }

    friend Row operator - (const Row<dev, T>& r0, const T& value) {
      Row r(r0);
      r -= value;
      return r;
    }

    friend Row operator - (const Row<dev, T>& r0, const Row<dev, T>& r1) {
      Row r(r0);
      r -= r1;
      return r;
    }

    friend Row operator % (const Row<dev, T>& r0, const T& value) {
      Row r(r0);
      r %= value;
      return r;
    }

    friend Row operator % (const Row<dev, T>& r0, const Row<dev, T>& r1) {
      Row r(r0);
      r %= r1;
      return r;
    }

    friend Row operator / (const Row<dev, T>& r0, const T& value) {
      Row r(r0);
      r /= value;
      return r;
    }

    friend Row operator / (const Row<dev, T>& r0, const Row<dev, T>& r1) {
      Row r(r0);
      r /= r1;
      return r;
    }

  };

} // namespace tensor

#endif //ROW_HPP
