#ifndef INTERFACE_HPP
#define INTERFACE_HPP

#include <tensor.hpp>

namespace neural {

  using namespace tensor;


  template <Device dev, typename T>
  class LayerInterface {
  public:
    LayerInterface () {}
    virtual ~LayerInterface () {};
    virtual const Mat<dev, T>& forward (const Mat<dev, T>&) = 0;
    virtual T outputCost (const Mat<dev, T>&, const Mat<dev, T>&, const T&) = 0;
    virtual T hiddenCost (const Mat<dev, T>&, const Mat<dev, T>&, const T&) = 0;
    virtual void update (const T&, const T&) = 0;
    virtual void clear () = 0;
    virtual const Mat<dev, T>& getOutput () = 0;
    virtual const Mat<dev, T>& getDelta () = 0;
    virtual const Mat<dev, T>& getWeight () = 0;
  };

}

#endif // INTERFACE_HPP
