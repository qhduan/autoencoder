#ifndef MAT_HPP
#define MAT_HPP

/*!
 * \file mat.hpp
 * \brief
 * \author qhduan.com
 * \date 2014-05-15
 */

namespace tensor {

  /*!
   * \brief Matrix class
   */
  template <Device dev, typename T>
  class Mat {
  public:

    friend MatSub<dev, T>;

    /*!
     * \brief Create an empty matrix
     */
    Mat ()
    : n_rows(rows_)
    , n_cols(cols_)
    , n_elem(size_)
    , elem_(NULL)
    , rows_(0)
    , cols_(0)
    , size_(0)
    , pitch_(0) {}

    /*!
     * \brief Create a matrix with rows and cols
     */
    Mat (int rows, int cols) : Mat() {
      if (rows <= 0 || cols <= 0) {
        std::cerr << "n_rows: " << rows << ", n_cols:" << cols << std::endl;
        throw std::runtime_error("Matrix invalid n_rows/n_cols");
      }
      this->rows_ = rows;
      this->cols_ = cols;
      this->pitch_ = this->n_rows;
      this->size_ = this->n_rows * this->n_cols;
      this->elem_ = Mempool<dev>::instance().template malloc<T>(this->n_elem);
    }

    /*!
     * \brief Create a matrix with rows and cols, but may assgin fill
     */
    Mat (int rows, int cols, fill::type t) : Mat(rows, cols) {
      switch (t) {
        case fill::zeros:
          this->zeros();
          break;
        case fill::ones:
          this->ones();
          break;
        case fill::eye:
          this->eye();
          break;
        case fill::randu:
          this->randu();
          break;
        case fill::randn:
          this->randn();
          break;
        case fill::none:
        default:
          break; //nothing
      }
    }

    /*!
     * \brief Create a matrix with an other rvalue reference matrix
     */

    Mat (Mat&& m) : Mat() {
      this->swap(m);
    }

    /*
     * \brief Create a matrix with an other matrix
     */
    Mat (const Mat<GPU, T>& m) : Mat() {
      this->constructor<GPU>(m);
    }

    Mat (const Mat<CPU, T>& m) : Mat() {
      this->constructor<CPU>(m);
    }

    /*!
     * \brief Create a matrix from Row
     */
    Mat (const Row<dev, T>& row) : Mat() {
      this->rows_ = row.n_rows;
      this->cols_ = row.n_cols;
      this->pitch_ = this->n_rows;
      this->size_ = this->n_rows * this->n_cols;
      this->elem_ = Mempool<dev>::instance().template malloc<T>(this->n_elem);
      TensorOperation<dev>::template x_to_y<T>(
        this->n_elem, row.memptr(), row.inc(), this->memptr(), 1
      );
    }

    /*!
     * \brief Create a matrix from Col
     */
    Mat (const Col<dev, T>& col) : Mat() {
      this->rows_ = col.n_rows;
      this->cols_ = col.n_cols;
      this->pitch_ = this->n_rows;
      this->size_ = this->n_rows * this->n_cols;
      this->elem_ = Mempool<dev>::instance().template malloc<T>(this->n_elem);
      TensorOperation<dev>::template x_to_y<T>(
        this->n_elem, col.memptr(), col.inc(), this->memptr(), 1
      );
    }

    /*!
     * \brief Create a matrix with initializer list
     * Mat m({1.0, 2.0});
     */
  	Mat (std::initializer_list<T> args) : Mat(std::vector<T>(args)) {}

    /*!
     * \brief Create a matrix with std::vector
     * Mat m(std::vector({1.0, 2.0}));
     */
    Mat (const std::vector<T>& vec) : Mat(vec.size(), 1) {
      for (int i = 0; i < this->size_; i++) {
        (*this)(i, 0) = vec[i];
      }
    }

    /*!
     * \brief Create a matrix with string
     * Mat m("1 2 3; 4 5 6");
     */
    Mat (const char* str) : Mat(std::string(str)) {}

    /*!
     * \brief Create a matrix with string
     * Mat m(std::string("1 2 3; 4 5 6"));
     */
    Mat (const std::string& str) : Mat() {
      std::vector< std::vector<T> > vec;

      std::stringstream ss(str);
      std::vector< std::string > elems;
      std::string item;
      while (std::getline(ss, item, ';')) {
        elems.push_back(item);
      }

      int size = elems.size();
      for (int i = 0; i < size; i++) {
        std::vector<T> t;
        std::stringstream sss(elems[i]);
        while (true) {
          T num;
          if (sss >> num) {
            t.push_back(num);
          } else {
            break;
          }
        }
        if (t.size() > 0) {
          vec.push_back(t);
        }
      }

      this->rows_ = vec.size();

      if (this->rows_ <= 0) {
        throw std::runtime_error("Mat(str) invalid rows");
      }

      this->cols_ = vec[0].size();

      if (this->cols_ <= 0) {
        throw std::runtime_error("Mat(str) invalid cols");
      }

      this->pitch_ = this->n_rows;
      this->size_ = this->n_rows * this->n_cols;
      this->elem_ = Mempool<dev>::instance().template malloc<T>(this->n_elem);

      for (int i = 0; i < this->n_rows; i++) {
        for (int j = 0; j < this->n_cols; j++) {
          (*this)(i, j) = vec[i][j];
        }
      }
    }

    /*! \brief Destruct matrix */
    ~Mat () {
      this->release();
    }

    /*! \brief Assgin a rvalue reference to matrix */

    Mat& operator= (Mat&& m) {
      this->swap(m);
      return *this;
    }

    /*! \brief Assgin an other matrix to matrix */
    Mat& operator= (const Mat<GPU, T>& m) {
      this->constructor<GPU>(m);
      return *this;
    }

    Mat& operator= (const Mat<CPU, T>& m) {
      this->constructor<CPU>(m);
      return *this;
    }

  	/*! \brief Mat m; m = std::vector({1, 2, 3}); // m is 3x1 */
  	Mat& operator= (const std::vector<T>& vec) {
  		Mat m(vec);
      this->swap(m);
  		return *this;
  	}

    /*! \brief Mat m; m = {1 2 3}; // m is 3x1 */
  	Mat& operator= (const std::initializer_list<T>& list) {
  		Mat m(list);
      this->swap(m);
  		return *this;
  	}

    /*! \brief Mat m; m = "1 2 3; 4 5 6; 7 8 9"; // m is 3x3 */
    Mat& operator= (const char* str) {
  		Mat m(str);
      this->swap(m);
      return *this;
    }

    /*! \brief Mat m; m = std::string("1 2 3; 4 5 6; 7 8 9"); // m is 3x3 */
  	Mat& operator= (const std::string& str) {
  		Mat m(str);
      this->swap(m);
  		return *this;
  	}

    const T operator [] (int index) const {
      if (index < 0 || index >= this->n_elem) {
        throw std::runtime_error("Mat::operator[int] invalid arguments");
      }
      return TensorOperation<dev>::template get<T>(&this->elem_[(index%this->n_rows) + (index/this->n_rows) * this->pitch_]);
    }

    ValueProxy<dev, T>  operator [] (int index) {
      if (index < 0 || index >= this->n_elem) {
        throw std::runtime_error("Mat::operator[int] invalid arguments");
      }
      return ValueProxy<dev, T>(&this->elem_[(index%this->n_rows) + (index/this->n_rows) * this->pitch_]);
    }

    const T operator () (int row, int col) const {
      if (row < 0 || row > this->n_rows || col < 0 || col > this->n_cols) {
        throw std::runtime_error("Mat::operator(int, int) invalid arguments");
      }
      return TensorOperation<dev>::template get<T>(&this->elem_[row + col * this->pitch_]);
    }

    ValueProxy<dev, T> operator () (int row, int col) {
      if (row < 0 || row > this->n_rows || col < 0 || col > this->n_cols) {
        throw std::runtime_error("Mat::operator(int, int) invalid arguments");
      }
      return ValueProxy<dev, T>(&this->elem_[row + col * this->pitch_]);
    }

    /*! \brief mat transpose */
    MatTranspose<dev, T> t () const {
      return MatTranspose<dev, T>(*this);
    }

    MatSub<dev, T> submat (int first_row, int last_row, int first_col, int last_col) {
      return MatSub<dev, T>(
        this->elem_ + (first_row + first_col * this->pitch()),
        last_row - first_row + 1,
        last_col - first_col + 1,
        this->pitch()
      );
    }

    const MatSub<dev, T> submat (int first_row, int last_row, int first_col, int last_col) const {
      return MatSub<dev, T>(
        this->elem_ + (first_row + first_col * this->pitch()),
        last_row - first_row + 1,
        last_col - first_col + 1,
        this->pitch()
      );
    }

    MatSub<dev, T> col (int col_number) {
      return this->submat(0, this->n_rows - 1, col_number, col_number);
    }

    const MatSub<dev, T> col (int col_number) const {
      return this->submat(0, this->n_rows - 1, col_number, col_number);
    }

    MatSub<dev, T> row (int row_number) {
      return this->submat(row_number, row_number, 0, this->n_cols - 1);
    }

    const MatSub<dev, T> row (int row_number) const {
      return this->submat(row_number, row_number, 0, this->n_cols - 1);
    }

    MatSub<dev, T> cols (int first_col, int last_col) {
      return this->submat(0, this->n_rows - 1, first_col, last_col);
    }

    const MatSub<dev, T> cols (int first_col, int last_col) const {
      return this->submat(0, this->n_rows - 1, first_col, last_col);
    }

    MatSub<dev, T> rows (int first_row, int last_row) {
      return this->submat(first_row, last_row, 0, this->n_cols - 1);
    }

    const MatSub<dev, T> rows (int first_row, int last_row) const {
      return this->submat(first_row, last_row, 0, this->n_cols - 1);
    }

    MatSub<dev, T> head_cols (int number_of_cols) {
      return this->submat(0, this->n_rows - 1, 0, number_of_cols - 1);
    }

    const MatSub<dev, T> head_cols (int number_of_cols) const {
      return this->submat(0, this->n_rows - 1, 0, number_of_cols - 1);
    }

    MatSub<dev, T> head_rows (int number_of_rows) {
      return this->submat(0, number_of_rows - 1, 0, this->n_cols - 1);
    }

    const MatSub<dev, T> head_rows (int number_of_rows) const {
      return this->submat(0, number_of_rows - 1, 0, this->n_cols - 1);
    }

    MatSub<dev, T> tail_cols (int number_of_cols) {
      return this->submat(0, this->n_rows - 1, this->cols_ - number_of_cols, this->n_cols - 1);
    }

    const MatSub<dev, T> tail_cols (int number_of_cols) const {
      return this->submat(0, this->n_rows - 1, this->cols_ - number_of_cols, this->n_cols - 1);
    }

    MatSub<dev, T> tail_rows (int number_of_rows) {
      return this->submat(this->rows_ - number_of_rows, this->n_rows - 1, 0, this->n_cols - 1);
    }

    const MatSub<dev, T> tail_rows (int number_of_rows) const {
      return this->submat(this->rows_ - number_of_rows, this->n_rows - 1, 0, this->n_cols - 1);
    }

    /*! \brief Fill matrix with certain number */
    Mat& fill (const T& value) {
      TensorOperation<dev>::matrix_call(v_to_A<dev, T>({
        this->n_rows, this->n_cols, value, this->memptr(), this->pitch()
      }));
      return *this;
    }

    Mat& reshape (int rows, int cols) {
      if (rows != this->n_rows || cols != this->n_cols) {

        T* data1 = Mempool<CPU>::instance().template malloc<T>(this->n_elem);
        TensorOperation<dev>::template A_to_B(
          this->n_rows, this->n_cols, this->memptr(), this->pitch(),
          data1, this->n_rows, CPU
        );

        int size = rows * cols;
        T* data2 = Mempool<CPU>::instance().template malloc<T>(size);

        for (int index = 0; index < size; index++) {
          if (index < this->n_elem) {
            int ni = index % rows;
            int nj = index / rows;
            int oi = index % this->n_rows;
            int oj = index / this->n_rows;
            data2[ni + nj * rows] = data1[oi + oj * this->n_rows];
          } else { // zero
            int ni = index % rows;
            int nj = index / rows;
            data2[ni + rows * nj] = 0.0;
          }
        }

        Mempool<CPU>::instance().template free<T>(data1);

        Mat m(rows, cols);
        m.fillData(data2, rows);

        Mempool<CPU>::instance().template free<T>(data2);

        this->swap(m);
      }
      return *this;
    }

    /*! \brief Fill matrix with 1 */
    Mat& ones () {
      return this->fill(static_cast<T>(1.0));
    }

    /*! \brief Fill matrix with 0 */
    Mat& zeros () {
      return this->fill(static_cast<T>(0.0));
    }

    Mat& eye () {
      this->fill(0);
      int len = this->rows_ < this->n_cols ? this->rows_ : this->cols_;
      for (int i = 0; i < len; i++) {
        (*this)(i, i) = 1;
      }
      return *this;
    }

    Mat& eye (int rows, int cols) {
      if (rows != this->rows && cols != this->cols) {
        Mat m(rows, cols);
        m.fill(0);
        m.eye();
        this->swap(m);
      } else {
        this->eye();
      }
      return *this;
    }

    MatEachCol<dev, T> each_col () {
      return MatEachCol<dev, T>(*this);
    }

    MatEachRow<dev, T> each_row () {
      return MatEachRow<dev, T>(*this);
    }

    /*! \brief Swap with other matrix */
    Mat& swap (Mat&& m) {
      std::swap(this->elem_, m.elem_);
      std::swap(this->rows_, m.rows_);
      std::swap(this->cols_, m.cols_);
      std::swap(this->size_, m.size_);
      std::swap(this->pitch_, m.pitch_);
      return *this;
    }

    /*! \brief Swap with other matrix */
    Mat& swap (Mat& m) {
      std::swap(this->elem_, m.elem_);
      std::swap(this->rows_, m.rows_);
      std::swap(this->cols_, m.cols_);
      std::swap(this->size_, m.size_);
      std::swap(this->pitch_, m.pitch_);
      return *this;
    }

    /*! \brief Swap row-1 and row-2 */
    Mat& swap_rows (int row1, int row2) {
      if (row1 < 0 || row1 >= this->rows_ || row2 < 0 || row2 >= this->rows_) {
        std::cerr << row1 << " " << row2 << std::endl;
        throw std::runtime_error("Mat::swap_rows invalid arguments");
      }
      if (row1 != row2) {
        Mat row_1 = std::move(this->row(row1));
        Mat row_2 = this->row(row2);
        this->row(row1) = row_2;
        this->row(row2) = row_1;
      }
      return *this;
    }

    /*! \brief Swap column-1 and column-2 */
    Mat& swap_cols (int col1, int col2) {
      if (col1 < 0 || col1 >= this->cols_ || col2 < 0 || col2 >= this->cols_) {
        std::cerr << col1 << " " << col2 << std::endl;
        throw std::runtime_error("Mat::swap_cols invalid arguments");
      }
      if (col1 != col2) {
        Mat col_1 = std::move(this->col(col1));
        Mat col_2 = this->col(col2);
        this->col(col1) = col_2;
        this->col(col2) = col_1;
      }
      return *this;
    }

    /*! \brief Element-wise rand uniform */
    Mat& randu (T a = 0.0, T b = 1.0) {
      std::random_device rd;
      std::default_random_engine e(rd());
      std::uniform_real_distribution<T> uniform_dist(a, b);

      T* data = Mempool<CPU>::instance().template malloc<T>(this->n_elem);
      for (int i = 0; i < this->n_elem; i++) {
        data[i] = uniform_dist(e);
      }

      this->fillData(data, this->n_rows);
      Mempool<CPU>::instance().template free<T>(data);
      return *this;
    }

    /*! \brief Element-wise rand normal */
    Mat& randn (T u = 0.0, T s = 1.0) {
      std::random_device rd;
      std::default_random_engine e(rd());
      std::normal_distribution<T> normal_dist(u, s);

      T* data = Mempool<CPU>::instance().template malloc<T>(this->n_elem);
      for (int i = 0; i < this->n_elem; i++) {
        data[i] = normal_dist(e);
      }

      this->fillData(data, this->n_rows);
      Mempool<CPU>::instance().template free<T>(data);
      return *this;
    }

    /*! \brief Element-wise exp */
    Mat& exp (void) {
      TensorOperation<dev>::matrix_call(fn_A_to_B<dev, T, functor::Exp<dev, T> >({
        this->n_rows, this->n_cols,
        this->memptr(), this->pitch(), this->memptr(), this->pitch()
      }));
      return *this;
    }

    /*! \brief Element-wise log */
    Mat& log (void) {
      TensorOperation<dev>::matrix_call(fn_A_to_B<dev, T, functor::Log<dev, T> >({
        this->n_rows, this->n_cols,
        this->memptr(), this->pitch(), this->memptr(), this->pitch()
      }));
      return *this;
    }

    /*! \brief Element-wise abs */
    Mat& abs (void) {
      TensorOperation<dev>::matrix_call(fn_A_to_B<dev, T, functor::Abs<dev, T> >({
        this->n_rows, this->n_cols,
        this->memptr(), this->pitch(), this->memptr(), this->pitch()
      }));
      return *this;
    }

    /*! \brief Element-wise sigmoid */
    Mat& sigmoid (void) {
      TensorOperation<dev>::matrix_call(fn_A_to_B<dev, T, functor::Sigmoid<dev, T> >({
        this->n_rows, this->n_cols,
        this->memptr(), this->pitch(), this->memptr(), this->pitch()
      }));
      return *this;
    }

    /*! \brief Element-wise d-sigmoid */
    Mat& dsigmoid (void) {
      TensorOperation<dev>::matrix_call(fn_A_to_B<dev, T, functor::Dsigmoid<dev, T> >({
        this->n_rows, this->n_cols,
        this->memptr(), this->pitch(), this->memptr(), this->pitch()
      }));
      return *this;
    }

    /*! \brief Element-wise tanh */
    Mat& tanh (void) {
      TensorOperation<dev>::matrix_call(fn_A_to_B<dev, T, functor::Tanh<dev, T> >({
        this->n_rows, this->n_cols,
        this->memptr(), this->pitch(), this->memptr(), this->pitch()
      }));
      return *this;
    }

    /*! \brief Element-wise d-tanh */
    Mat& dtanh (void) {
      TensorOperation<dev>::matrix_call(fn_A_to_B<dev, T, functor::Dtanh<dev, T> >({
        this->n_rows, this->n_cols,
        this->memptr(), this->pitch(), this->memptr(), this->pitch()
      }));
      return *this;
    }

    /*! \brief Element-wise pow(n) */
    Mat& pow (T n) {
      TensorOperation<dev>::matrix_call(A_op_v_to_B<dev, T, functor::Pow<dev, T> >({
        this->n_rows, this->n_cols, this->memptr(), this->pitch(),
        n, this->memptr(), this->pitch()
      }));
      return *this;
    }

    /*! \brief Element-wise sqrt */
    Mat& sqrt () {
      return this->pow(static_cast<T>(0.5));
    }

    /*! \brief Element-wise sign */
    Mat& sign () {
      TensorOperation<dev>::matrix_call(fn_A_to_B<dev, T, functor::Sign<dev, T> >({
        this->n_rows, this->n_cols,
        this->memptr(), this->pitch(), this->memptr(), this->pitch()
      }));
      return *this;
    }

    /*! \brief Element-wise floor */
    Mat& floor (void) {
      for (int i = 0; i < this->n_rows; i++) {
        for (int j = 0; j < this->n_cols; j++) {
          auto t = (*this)(i, j);
          (*this)(i, j) = std::floor(t);
        }
      }
      return *this;
    }

    /*! \brief Element-wise ceil */
    Mat& ceil (void) {
      for (int i = 0; i < this->n_rows; i++) {
        for (int j = 0; j < this->n_cols; j++) {
          auto t = (*this)(i, j);
          (*this)(i, j) = std::ceil(t);
        }
      }
      return *this;
    }

    /*! \brief Element-wise round */
    Mat& round (void) {
      for (int i = 0; i < this->n_rows; i++) {
        for (int j = 0; j < this->n_cols; j++) {
          auto t = (*this)(i, j);
          (*this)(i, j) = std::round(t);
        }
      }
      return *this;
    }

    friend std::ostream& operator << (std::ostream& os, const Mat<dev, T>& m) {
      m.print(os);
      return os;
    }

    /*! \brief Print matrix into ostream with header */
    void print (std::ostream& os, const std::string& header) const {
      if (header.length() > 0) {
        std::cout << header << std::endl;
      }
      auto p = os.precision();
      auto f = os.flags();
      os.precision(4);
      os.setf(std::ios::fixed, std:: ios::floatfield);

      for (int i = 0; i < this->n_rows; i++) {
        for (int j = 0; j < this->n_cols; j++) {
          T value = (*this)(i, j);
          if (value >= 0) {
            os << "   " << value;
          } else {
            os << "  " << value;
          }
        }
        os << std::endl;
      }
      os.precision(p);
      os.flags(f);
    }

    void print (const std::string& header) const {
      this->print(std::cout, header);
    }

    void print (std::ostream& os) const {
      this->print(os, std::string(""));
    }

    void print () const {
      this->print(std::cout, std::string(""));
    }

    T* memptr () {
      return this->elem_;
    }

    const T* memptr () const {
      return this->elem_;
    }

    int pitch () const {
      return this->pitch_;
    }

    /*! \brief in-place addition to value */
    Mat& operator += (const T& value) {
      TensorOperation<dev>::matrix_call(A_op_v_to_B<dev, T, functor::Plus<dev, T> >({
        this->n_rows, this->n_cols, this->memptr(), this->pitch(),
        value, this->memptr(), this->pitch()
      }));
      return *this;
    }

    /*! \brief in-place addition to other matrix */
    Mat& operator += (const Mat& m) {
      if (this->rows_ != m.n_rows || this->cols_ != m.n_cols) {
        std::cerr << this->n_rows << "x" << this->n_cols << "\t"
                  << m.n_rows << "x" << m.n_cols << std::endl;
        throw std::runtime_error("Mat += invalid");
      }
      TensorOperation<dev>::matrix_call(A_op_B_to_C<dev, T, functor::Plus<dev, T> >({
        this->n_rows, this->n_cols, this->memptr(), this->pitch(),
        m.memptr(), m.pitch(), this->memptr(), this->pitch()
      }));
      return *this;
    }

    /*! \brief in-place substraction to value */
    Mat& operator -= (const T& value) {
      TensorOperation<dev>::matrix_call(A_op_v_to_B<dev, T, functor::Minus<dev, T> >({
        this->n_rows, this->n_cols, this->memptr(), this->pitch(),
        value, this->memptr(), this->pitch()
      }));
      return *this;
    }

    /*! \brief in-place substraction to other matrix */
    Mat& operator -= (const Mat& m) {
      if (this->rows_ != m.n_rows || this->cols_ != m.n_cols) {
        std::cerr << this->n_rows << "x" << this->n_cols << "\t"
                  << m.n_rows << "x" << m.n_cols << std::endl;
        throw std::runtime_error("Mat -= invalid");
      }
      TensorOperation<dev>::matrix_call(A_op_B_to_C<dev, T, functor::Minus<dev, T> >({
        this->n_rows, this->n_cols, this->memptr(), this->pitch(),
        m.memptr(), m.pitch(), this->memptr(), this->pitch()
      }));
      return *this;
    }


    /* \brief in-place element-wise multiplication to other matrix */
    Mat& operator %= (const Mat& m) {
      if (this->rows_ != m.n_rows || this->cols_ != m.n_cols) {
        std::cerr << this->n_rows << "x" << this->n_cols << "\t"
                  << m.n_rows << "x" << m.n_cols << std::endl;
        throw std::runtime_error("Mat %= invalid");
      }
      TensorOperation<dev>::matrix_call(A_op_B_to_C<dev, T, functor::Multiplies<dev, T> >({
        this->n_rows, this->n_cols, this->memptr(), this->pitch(),
        m.memptr(), m.pitch(), this->memptr(), this->pitch()
      }));
      return *this;
    }

    /* \brief in-place element-wise division to value */
    Mat& operator /= (const T& value) {
      TensorOperation<dev>::matrix_call(A_op_v_to_B<dev, T, functor::Divides<dev, T> >({
        this->n_rows, this->n_cols, this->memptr(), this->pitch(),
        value, this->memptr(), this->pitch()
      }));
      return *this;
    }

    /* \brief in-place element-wise division to other matrix */
    Mat& operator /= (const Mat& m) {
      if (this->rows_ != m.n_rows || this->cols_ != m.n_cols) {
        std::cerr << this->n_rows << "x" << this->n_cols << "\t"
                  << m.n_rows << "x" << m.n_cols << std::endl;
        throw std::runtime_error("Mat /= invalid");
      }
      TensorOperation<dev>::matrix_call(A_op_B_to_C<dev, T, functor::Divides<dev, T> >({
        this->n_rows, this->n_cols, this->memptr(), this->pitch(),
        m.memptr(), m.pitch(), this->memptr(), this->pitch()
      }));
      return *this;
    }


    /* \brief in-place element-wise multiplication to value */
    Mat& operator *= (const T& value) {
      TensorOperation<dev>::matrix_call(A_op_v_to_B<dev, T, functor::Multiplies<dev, T> >({
        this->n_rows, this->n_cols, this->memptr(), this->pitch(),
        value, this->memptr(), this->pitch()
      }));
      return *this;
    }

    /*! \brief Element-wise equality evaluation of two matrices */
    friend bool operator == (const Mat<dev, T>& m0, const Mat<dev, T>& m1) {
      if (m0.n_rows != m1.n_rows || m0.n_cols != m1.n_cols) {
        return false;
      }
      T ret = norm(m0 - m1);
      if (ret < static_cast<T>(1e-7)) {
        return true;
      }
      return false;
    }


    /*
     * Friend Functions
     */

    /*! \brief Element-wise addition of value and matrix */
    friend Mat<dev, T> operator + (const T& value, const Mat<dev, T>& m0) {
      return m0 + value;
    }

    /*! \brief Element-wise addition of matrix and value */
    friend Mat<dev, T> operator + (const Mat<dev, T>& m0, const T& value) {
      Mat<dev, T> m(m0.n_rows, m0.n_cols);
      TensorOperation<dev>::matrix_call(A_op_v_to_B<dev, T, functor::Plus<dev, T> >({
        m.n_rows, m.n_cols, m0.memptr(), m0.pitch(),
        value, m.memptr(), m.pitch()
      }));
      return m;
    }

    /*! \brief Element-wise addition of two matrices */
    friend Mat<dev, T> operator + (const Mat<dev, T>& m0, const Mat<dev, T>& m1) {
      Mat<dev, T> m(m0.n_rows, m0.n_cols);
      TensorOperation<dev>::matrix_call(A_op_B_to_C<dev, T, functor::Plus<dev, T> >({
        m.n_rows, m.n_cols, m0.memptr(), m0.pitch(),
        m1.memptr(), m1.pitch(), m.memptr(), m.pitch()
      }));
      return m;
    }

    /*! \brief -mat */
    friend Mat<dev, T> operator - (const Mat<dev, T>& m0) {
      return static_cast<T>(0) - m0;
    }

    /*! \brief Element-wise substraction of value and matrix */
    friend Mat<dev, T> operator - (const T& value, const Mat<dev, T>& m0) {
      Mat<dev, T> m(m0.n_rows, m0.n_cols);
      TensorOperation<dev>::matrix_call(v_op_A_to_B<dev, T, functor::Minus<dev, T> >({
        m.n_rows, m.n_cols, value,
        m0.memptr(), m0.pitch(), m.memptr(), m.pitch()
      }));
      return m;
    }

    /*! \brief Element-wise substraction of matrix and value */
    friend Mat<dev, T> operator - (const Mat<dev, T>& m0, const T& value) {
      Mat<dev, T> m(m0.n_rows, m0.n_cols);
      TensorOperation<dev>::matrix_call(A_op_v_to_B<dev, T, functor::Minus<dev, T> >({
        m.n_rows, m.n_cols, m0.memptr(), m0.pitch(),
        value, m.memptr(), m.pitch()
      }));
      return m;
    }

    /*! \brief Element-wise substraction of matrices */
    friend Mat<dev, T> operator - (const Mat<dev, T>& m0, const Mat<dev, T>& m1) {
      Mat<dev, T> m(m0.n_rows, m0.n_cols);
      TensorOperation<dev>::matrix_call(A_op_B_to_C<dev, T, functor::Minus<dev, T> >({
        m.n_rows, m.n_cols, m0.memptr(), m0.pitch(),
        m1.memptr(), m1.pitch(), m.memptr(), m.pitch()
      }));
      return m;
    }

    /*! \brief Element-wise multiplication of value and matrix */
    friend Mat<dev, T> operator * (const T& value, const Mat<dev, T>& m0) {
      return m0 * value;
    }

    /*! \brief Element-wise multiplication of matrix and value */
    friend Mat<dev, T> operator * (const Mat<dev, T>& m0, const T& value) {
      Mat<dev, T> m(m0.n_rows, m0.n_cols);
      TensorOperation<dev>::matrix_call(A_op_v_to_B<dev, T, functor::Multiplies<dev, T> >({
        m.n_rows, m.n_cols, m0.memptr(), m0.pitch(),
        value, m.memptr(), m.pitch()
      }));
      return m;
    }

    /*! \brief Element-wise multiplication of matrices */
    friend Mat<dev, T> operator % (const Mat<dev, T>& m0, const Mat<dev, T>& m1) {
      Mat<dev, T> m(m0.n_rows, m0.n_cols);
      TensorOperation<dev>::matrix_call(A_op_B_to_C<dev, T, functor::Multiplies<dev, T> >({
        m.n_rows, m.n_cols, m0.memptr(), m0.pitch(),
        m1.memptr(), m1.pitch(), m.memptr(), m.pitch()
      }));
      return m;
    }

    /*! \brief Element-wise division of value and matrix */
    friend Mat<dev, T> operator / (const T& value, const Mat<dev, T>& m0) {
      Mat<dev, T> m(m0.n_rows, m0.n_cols);
      TensorOperation<dev>::matrix_call(v_op_A_to_B<dev, T, functor::Divides<dev, T> >({
        m.n_rows, m.n_cols, value,
        m0.memptr(), m0.pitch(), m.memptr(), m.pitch()
      }));
      return m;
    }

    /*! \brief Element-wise division of matrix and value */
    friend Mat<dev, T> operator / (const Mat<dev, T>& m0, const T& value) {
      Mat<dev, T> m(m0.n_rows, m0.n_cols);
      TensorOperation<dev>::matrix_call(A_op_v_to_B<dev, T, functor::Divides<dev, T> >({
        m.n_rows, m.n_cols, m0.memptr(), m0.pitch(),
        value, m.memptr(), m.pitch()
      }));
      return m;
    }

    /*! \brief Element-wise division of matrices */
    friend Mat<dev, T> operator / (const Mat<dev, T>& m0, const Mat<dev, T>& m1) {
      Mat<dev, T> m(m0.n_rows, m0.n_cols);
      TensorOperation<dev>::matrix_call(A_op_B_to_C<dev, T, functor::Divides<dev, T> >({
        m.n_rows, m.n_cols, m0.memptr(), m0.pitch(),
        m1.memptr(), m1.pitch(), m.memptr(), m.pitch()
      }));
      return m;
    }

    /* \brief const reference */
    const int& n_rows;
    const int& n_cols;
    const int& n_elem;

  private:

    Mat (T* elem, int rows, int cols, int pitch)
    : Mat() {
      this->elem_ = elem;
      this->rows_ = rows;
      this->cols_ = cols;
      this->pitch_ = pitch;
      this->size_ = rows * cols;
    }


    template <Device dev2>
    void constructor (const Mat<dev2, T>& m) {
      if (this->memptr() != NULL && this->n_rows == m.n_rows && this->n_cols == m.n_cols) {
        TensorOperation<dev2>::template A_to_B<T>( // copy from dev2 to dev
          this->n_rows, this->n_cols,
          m.memptr(), m.pitch(), this->memptr(), this->pitch(), dev
        );
      } else {
        this->release();
        this->rows_ = m.n_rows;
        this->cols_ = m.n_cols;
        this->pitch_ = this->n_rows;
        this->size_ = this->n_rows * this->n_cols;
        this->elem_ = Mempool<dev>::instance().template malloc<T>(this->n_elem);
        TensorOperation<dev2>::template A_to_B<T>(
          this->n_rows, this->n_cols,
          m.memptr(), m.pitch(), this->memptr(), this->pitch(), dev
        );
      }
    }

    /*!
     * \brief Release the resource
     */
    void release () {
      if (this->elem_ != NULL) {
        Mempool<dev>::instance().template free<T>(this->elem_);
        this->elem_ = NULL;
      }
    }

    void fillData (T* data, int pitch) {
      TensorOperation<CPU>::template A_to_B<T>(
        this->n_rows, this->n_cols, data, pitch,
        this->memptr(), this->pitch(), dev
      );
    }

    /*! \brief NULL or a pointer to data */
    T* elem_;
    /*! \brief Rows of matrix */
    int rows_;
    /*! \brief Cols of matrix */
    int cols_;
    /*! \brief Number of element, aka. rows * cols */
    int size_;
    /*! \brief Pitch of matrix, for now, it's just same as rows */
    int pitch_;

  };

} // namespace tensor

#endif //MAT_HPP
