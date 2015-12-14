
#include <memory>

#include "../inc/layer/neuralnetwork.hpp"
#include "../inc/layer/sparse.hpp"
#include "../inc/layer/softmax.hpp"

namespace neural {

  using namespace tensor;

  template <tensor::Device dev, typename T>
  class Train {
  public:
    Train () {
      alpha = 0.02;
      lambda = 0.0001;
      momentum = 0.0;
    }

    ~Train () { }

    const Mat<dev, T>& forward (const Mat<dev, T>& x) {
      // forward
      int size = layers.size();
      for (int j = 0; j < size; j++) {
        if (j == 0) { // first layer
          layers[j]->forward(x);
        } else {
          layers[j]->forward( layers[j - 1]->getOutput() );
        }
      }
      return layers[layers.size() - 1]->getOutput();
    }

    void add (LayerInterface<dev, T>* layer) {
      layers.push_back(std::shared_ptr< LayerInterface<dev, T> >(layer));
    }

    void add (const std::shared_ptr< LayerInterface<dev, T> >& layer) {
      layers.push_back(layer);
    }

    void remove (int i) {
      layers.erase(layers.begin() + i);
    }

    void run (int count, int batch, const Mat<dev, T>& x, const Mat<dev, T>& y) {
      for (int i = 0; i < count; i++) {
        int start_col = rand() % (x.n_cols - batch);
        int end_col = start_col + batch - 1;

        if (batch == 0) {
          start_col = 0;
          end_col = x.n_cols - 1;
        }

        T error = 0.0;
        int size = layers.size();

        auto X = x.cols(start_col, end_col);
        auto Y = y.cols(start_col, end_col);

        // forward
        forward(X);
        // backward
        for (int j = size - 1; j >= 0; j--) {
          if (j == (size - 1) && j == 0) { // output layer
            error += layers[j]->outputCost(X, Y, lambda);
          } else if (j == (size - 1)) { // output layer
            error += layers[j]->outputCost(layers[j - 1]->getOutput(), Y, lambda);
          } else if (j == 0) { // first layer
            // error +=
            layers[j]->hiddenCost(X, layers[j + 1]->getDelta(), lambda);
          } else { // other layers
            // error +=
            layers[j]->hiddenCost(layers[j - 1]->getOutput(), layers[j + 1]->getDelta(), lambda);
          }
        }
        // update
        for (int j = 0; j < size; j++) {
          layers[j]->update(alpha, momentum);
        }
        //
        std::cout<<i<<": "<<error<<std::endl;
      }
    }

    std::vector< std::shared_ptr< LayerInterface<dev, T> > > layers;
    T alpha;
    T lambda;
    T momentum;
  };
}
