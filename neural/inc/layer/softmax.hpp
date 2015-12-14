#include <tensor.hpp>
#include "interface.hpp"

namespace neural {

  using namespace tensor;

  template <Device dev, typename T>
  class Softmax : public LayerInterface<dev, T> {

  public:
    Softmax (int input, int output) {

      theta = Mat<dev, T>(output, input);

      thetaGrad = Mat<dev, T>(output, input, fill::zeros);

      thetaGrad_last = Mat<dev, T>(output, input, fill::zeros);

      T r = sqrt(6.0) / sqrt(output + input + 1);
      theta.randu(-r, r);
    }

    ~Softmax () {}

    const Mat<dev, T>& forward (const Mat<dev, T>& x) {

      auto theta_data = theta * x;
      theta_data.each_row() -= max(theta_data, 0);

      auto prob_data = exp(theta_data);
      auto temp2 = sum(prob_data, 0);
      prob_data.each_row() /= temp2;

      a = prob_data;
      return a;
    }

    T outputCost (const Mat<dev, T>& x, const Mat<dev, T>& y, const T& lambda) {

      T m = static_cast<T>(x.n_cols);
      T error1 = - accu(log(a) % y) / m ;
      T error2 = (lambda / 2.0) * accu(pow(theta, 2.0));
      T error = error1 + error2;

      thetaGrad = (y - a) * x.t() * (-1.0 / m)  + lambda * theta;

      delta = theta.t() * (y - a) * -1.0;

      return error;
    }

    T hiddenCost (const Mat<dev, T>& x, const Mat<dev, T>& lastDelta, const T& lambda) {
      throw std::logic_error("Softmax layer only output");
    }

    void update (const T& alpha, const T& momentum) {
      thetaGrad_last = momentum * thetaGrad_last + alpha * thetaGrad;
      theta -= thetaGrad_last;
    }

    void clear () {
      a = Mat<dev, T>();
      theta = Mat<dev, T>();
      thetaGrad = Mat<dev, T>();
    }

    const Mat<dev, T>& getOutput () {
      return a;
    }

    const Mat<dev, T>& getDelta () {
      return delta;
    }

    const Mat<dev, T>& getWeight () {
      return theta;
    }

    Mat<dev, T> a;

    Mat<dev, T> theta;
    Mat<dev, T> thetaGrad;
    Mat<dev, T> thetaGrad_last;
    Mat<dev, T> delta;

  };

}
