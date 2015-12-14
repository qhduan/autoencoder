#ifndef VEC_HPP
#define VEC_HPP
/*!
 * \file tensor.h
 * \brief
 * \author qhduan.com
 * \date 2014-05-15
 */

namespace tensor {

  template <Device dev, typename T>
  class Vec {
  public:

    Vec& operator += (const T& value) {
      TensorOperation<dev>::vector_call(x_op_v_to_y<dev, T, functor::Plus<dev,T> >({
        this->n_elem, this->memptr(), this->inc(),
        value, this->memptr(), this->inc()
      }));
      return *this;
    }

    Vec& operator += (const Vec<dev, T>& v) {
      if (this->size_ != v.size_) {
        std::cerr << "n_elem: " << this->n_elem << "\t"
                  << "n_elem: " << v.n_elem << "\t";
        throw std::runtime_error("Vec += invalid");
      }
      TensorOperation<dev>::vector_call(x_op_y_to_z<dev, T, functor::Plus<dev,T> >({
        this->n_elem, this->memptr(), this->inc(),
        v.memptr(), v.inc(), this->memptr(), this->inc()
      }));
      return *this;
    }

    Vec& operator -= (const T& value) {
      TensorOperation<dev>::vector_call(x_op_v_to_y<dev, T, functor::Minus<dev,T> >({
        this->n_elem, this->memptr(), this->inc(),
        value, this->memptr(), this->inc()
      }));
      return *this;
    }

    Vec& operator -= (const Vec<dev, T>& v) {
      if (this->size_ != v.size_) {
        std::cerr << "n_elem: " << this->n_elem << "\t"
                  << "n_elem: " << v.n_elem << "\t";
        throw std::runtime_error("Vec -= invalid");
      }
      TensorOperation<dev>::vector_call(x_op_y_to_z<dev, T, functor::Minus<dev,T> >({
        this->n_elem, this->memptr(), this->inc(),
        v.memptr(), v.inc(), this->memptr(), this->inc()
      }));
      return *this;
    }

    Vec& operator %= (const T& value) {
      TensorOperation<dev>::vector_call(x_op_v_to_y<dev, T, functor::Multiplies<dev,T> >({
        this->n_elem, this->memptr(), this->inc(),
        value, this->memptr(), this->inc()
      }));
      return *this;
    }

    Vec& operator %= (const Vec<dev, T>& v) {
      if (this->size_ != v.size_) {
        std::cerr << "n_elem: " << this->n_elem << "\t"
                  << "n_elem: " << v.n_elem << "\t";
        throw std::runtime_error("Vec += invalid");
      }
      TensorOperation<dev>::vector_call(x_op_y_to_z<dev, T, functor::Multiplies<dev,T> >({
        this->n_elem, this->memptr(), this->inc(),
        v.memptr(), v.inc(), this->memptr(), this->inc()
      }));
      return *this;
    }

    Vec& operator /= (const T& value) {
      TensorOperation<dev>::vector_call(x_op_v_to_y<dev, T, functor::Divides<dev,T> >({
        this->n_elem, this->memptr(), this->inc(),
        value, this->memptr(), this->inc()
      }));
      return *this;
    }

    Vec& operator /= (const Vec<dev, T>& v) {
      if (this->size_ != v.size_) {
        std::cerr << "n_elem: " << this->n_elem << "\t"
                  << "n_elem: " << v.n_elem << "\t";
        throw std::runtime_error("Vec += invalid");
      }
      TensorOperation<dev>::vector_call(x_op_y_to_z<dev, T, functor::Divides<dev,T> >({
        this->n_elem, this->memptr(), this->inc(),
        v.memptr(), v.inc(), this->memptr(), this->inc()
      }));
      return *this;
    }

    void fill (const T& value) {
      TensorOperation<dev>::vector_call(v_to_x<dev, T>({
        this->n_elem, value, this->memptr(), this->inc()
      }));
    }

    void fill (const std::vector<T>& vec) {
      int len = this->n_elem;
      int size = vec.size();
      if (size < len) {
        len = size;
      }
      for (int i = 0; i < len; i++) {
        (*this)(i) = vec[i];
      }
    }

    T operator () (int pos) const {
      if (pos < 0 || pos >= this->n_elem) {
        throw std::runtime_error("Vec::operator(int) invalid argument");
      }
      return TensorOperation<dev>::template get<T>(&this->memptr()[pos * this->inc()]);
    }

    T* memptr () {
      return this->elem_;
    }

    const T* memptr () const {
      return this->elem_;
    }

    int inc () const {
      return this->inc_;
    }

    ValueProxy<dev, T> operator () (int pos) {
      if (pos < 0 || pos >= this->size_) {
        throw std::runtime_error("Vec::operator(int) invalid argument");
      }
      return ValueProxy<dev, T>(&this->elem_[pos * this->inc_]);
    }

    /*! \brief Swap with other vector */
    Vec& swap (Vec&& v) {
      std::swap(this->elem_, v.elem_);
      std::swap(this->rows_, v.rows_);
      std::swap(this->cols_, v.cols_);
      std::swap(this->size_, v.size_);
      std::swap(this->inc_, v.inc_);
      return *this;
    }

    /*! \brief Swap with other vector */
    Vec& swap (Vec& v) {
      std::swap(this->elem_, v.elem_);
      std::swap(this->rows_, v.rows_);
      std::swap(this->cols_, v.cols_);
      std::swap(this->size_, v.size_);
      std::swap(this->inc_, v.inc_);
      return *this;
    }

    const int& n_rows;
    const int& n_cols;
    const int& n_elem;

  protected:

    Vec ()
    : n_rows(rows_)
    , n_cols(cols_)
    , n_elem(size_) {
      this->rows_ = 0;
      this->cols_ = 0;
      this->size_ = 0;
      this->inc_ = 1;
      this->elem_ = NULL;
    }

    Vec (int size)
    : Vec() {
      if (size <= 0) {
        throw std::runtime_error("Vec(int) size should be >= 1");
      }
      this->size_ = size;
      this->inc_ = 1;
      this->elem_ = Mempool<dev>::instance().template malloc<T>(this->size_);
    }

    Vec (const std::vector<T>& vec)
    : Vec() {
      int size = vec.size();
      if (size <= 0) {
        throw std::runtime_error("Vec(std::vector) size should be >= 1");
      }
      this->size_ = size;
      this->inc_ = 1;
      this->elem_ = Mempool<dev>::instance().template malloc<T>(this->size_);
      this->fill(vec);
    }

    Vec (const std::string& str)
    : Vec() {
      std::vector<T> vec;
      std::stringstream sss(str);
      while (true) {
        T num;
        if (sss >> num) {
          vec.push_back(num);
        } else {
          break;
        }
      }
      int size = vec.size();
      if (size <= 0) {
        throw std::runtime_error("Vec(std::string) size should be >= 1");
      }
      this->size_ = size;
      this->inc_ = 1;
      this->elem_ = Mempool<dev>::instance().template malloc<T>(this->size_);
      this->fill(vec);
    }

    Vec (std::initializer_list<T> args)
    : Vec(std::vector<T>(args)) {}

    void release () {
      if (this->elem_ != NULL) {
        Mempool<dev>::instance().template free<T>(this->elem_);
        this->elem_ = NULL;
      }
    }

    ~Vec () {
      this->release();
    }

    T* elem_;
    int rows_;
    int cols_;
    int size_;
    int inc_;
  };

} // namespace tensor

#endif //VEC_HPP
