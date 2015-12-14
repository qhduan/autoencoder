#include <tensor.hpp>
#include "interface.hpp"

namespace neural {

  using namespace tensor;

  template <Device dev, typename T>
  class Sparse : public LayerInterface<dev, T> {

  public:
    Sparse (int input, int output, T sp_ = 0.1, T beta_ = 3.0) {
      sp = sp_;
      beta = beta_;
      sparsing = true;

      W = Mat<dev, T>(output, input);
      b = Mat<dev, T>(output, 1);

      Wgrad = Mat<dev, T>(output, input, fill::zeros);
      bgrad = Mat<dev, T>(output, 1, fill::zeros);

      Wgrad_last = Mat<dev, T>(output, input, fill::zeros);
      bgrad_last = Mat<dev, T>(output, 1, fill::zeros);

      T r = sqrt(6.0) / sqrt(output + input + 1);

      W.randu(-r, r);
      b.zeros();
    }

    ~Sparse () {}

    const Mat<dev, T>& forward (const Mat<dev, T>& x) {
      z = W * x;
      z.each_col() += b;
      a = sigmoid(z);
      return a;
    }

    T outputCost (const Mat<dev, T>& x, const Mat<dev, T>& y, const T& lambda) {
      throw std::logic_error("Sparse layer only hiddenCost");
    }

    T hiddenCost (const Mat<dev, T>& x, const Mat<dev, T>& lastDelta, const T& lambda) {
      // forward(x);
      T m = static_cast<T>(x.n_cols);

      Mat<dev, T> rho;

      delta = lastDelta;

      if (sparsing) {
        rho = sum(a, 1) / m;
        delta.each_col() += beta * ( - sp / rho + (1.0 - sp) / (1.0 - rho) );
      }

      delta %= dsigmoid(z);

      Wgrad = (1.0 / m) * ( delta * x.t() ) + lambda * W;
      bgrad = (1.0 / m) * sum(delta, 1);

      T error1 = (lambda / 2.0) * accu(pow(W, 2.0));
      T error2 = 0;

      if (sparsing) {
        error2 = beta * accu( log(sp / rho) * sp + log((1.0 - sp) / (1.0 - rho)) * (1.0 - sp) );
      }

      T error = error1 + error2;
      return error;
    }

    void update (const T& alpha, const T& momentum) {
      Wgrad_last = momentum * Wgrad_last + alpha * Wgrad;
      bgrad_last = momentum * bgrad_last + alpha * bgrad;
      W -= Wgrad_last;
      b -= bgrad_last;
    }

    void clear () {
      z = Mat<dev, T>();
      a = Mat<dev, T>();

      W = Mat<dev, T>();
      b = Mat<dev, T>();

      Wgrad = Mat<dev, T>();
      bgrad = Mat<dev, T>();

      Wgrad_last = Mat<dev, T>();
      bgrad_last = Mat<dev, T>();
    }

    const Mat<dev, T>& getOutput () {
      return a;
    }

    const Mat<dev, T>& getDelta () {
      return delta;
    }

    const Mat<dev, T>& getWeight () {
      return W;
    }

    bool sparsing;
    T sp;
    T beta;

    Mat<dev, T> delta;

    Mat<dev, T> W;
    Mat<dev, T> b;

    Mat<dev, T> Wgrad;
    Mat<dev, T> bgrad;

    Mat<dev, T> Wgrad_last;
    Mat<dev, T> bgrad_last;

    Mat<dev, T> z;
    Mat<dev, T> a;

  };

}
