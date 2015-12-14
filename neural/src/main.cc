
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "../inc/train.hpp"

int ReverseInt (int i) {
  unsigned char ch1, ch2, ch3, ch4;
  ch1=i&255;
  ch2=(i>>8)&255;
  ch3=(i>>16)&255;
  ch4=(i>>24)&255;
  return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
}

template <tensor::Device dev, typename T>
tensor::Mat<dev, T> MNIST_images (const std::string& name) {
  std::ifstream file;
  file.open(name, std::ios::binary);

  int magic;
  int number_of_images;
  int number_of_rows;
  int number_of_columns;

  file.read((char*)&magic, 4);
  file.read((char*)&number_of_images, 4);
  file.read((char*)&number_of_rows, 4);
  file.read((char*)&number_of_columns, 4);

  magic = ReverseInt(magic);
  number_of_images = ReverseInt(number_of_images);
  number_of_rows = ReverseInt(number_of_rows);
  number_of_columns = ReverseInt(number_of_columns);

  std::cout << magic << std::endl;
  std::cout << number_of_images << std::endl;
  std::cout << number_of_rows << std::endl;
  std::cout << number_of_columns << std::endl;


  tensor::Mat<tensor::CPU, T> ret(28*28, number_of_images);

  unsigned char buf;
  for (int i = 0; i < number_of_images; i++) {
    for (int j = 0; j < (28 * 28); j++) {
      file.read((char*)&buf, 1);
      T temp = static_cast<double>(buf);
      ret(j, i) = temp;
    }
  }

  return ret;
}

template <tensor::Device dev, typename T>
tensor::Mat<dev, T> MNIST_labels (const std::string& name) {
  std::ifstream file;
  file.open(name, std::ios::binary);

  int magic;
  int number_of_items;

  file.read((char*)&magic, 4);
  file.read((char*)&number_of_items, 4);

  magic = ReverseInt(magic);
  number_of_items = ReverseInt(number_of_items);

  std::cout << magic << std::endl;
  std::cout << number_of_items << std::endl;

  tensor::Mat<tensor::CPU, T> ret(10, number_of_items);
  ret.fill(0.0);
  unsigned char buf;
  for (int i = 0; i < number_of_items; i++) {
    file.read((char*)&buf, 1);
    int index = buf;
    ret(index, i) = 1.0;
  }

  return ret;
}

template <tensor::Device dev, typename T>
tensor::Mat<dev, T> MNIST_labels2 (const std::string& name) {
  std::ifstream file;
  file.open(name, std::ios::binary);

  int magic;
  int number_of_items;

  file.read((char*)&magic, 4);
  file.read((char*)&number_of_items, 4);

  magic = ReverseInt(magic);
  number_of_items = ReverseInt(number_of_items);

  std::cout << magic << std::endl;
  std::cout << number_of_items << std::endl;

  tensor::Mat<tensor::CPU, T> ret(1, number_of_items);
  ret.fill(0.0);
  unsigned char buf;
  for (int i = 0; i < number_of_items; i++) {
    file.read((char*)&buf, 1);
    int index = buf;
    ret(0, i) = index;
  }

  return ret;
}




template <tensor::Device dev, typename T>
void MNIST_test (const tensor::Mat<dev, T>& result, const tensor::Mat<dev, T>& test_images, const tensor::Mat<dev, T>& test_labels) {
  int total = 0;
  int right = 0;
  std::cout << test_images.n_rows << " " << test_images.n_cols << "\n";
  std::cout << test_labels.n_rows << " " << test_labels.n_cols << "\n";
  std::cout << result.n_rows << " " << result.n_cols << "\n";
  for (int i = 0; i < result.n_cols; i++) {
    int max_index = -1;
    T max_value = -1;
    for (int j = 0; j < result.n_rows; j++) {
      T temp = result(j, i);
      if (max_value == -1 || temp > max_value) {
        max_value = temp;
        max_index = j;
      }
    }
    if (test_labels(max_index, i) == 1.0) {
      right++;
    }
    total++;
  }
  std::cout<<"total: "<< total << ", right: " << right << ", ratio: " << (static_cast<double>(right) / total) << std::endl;
}



template <tensor::Device dev, typename T>
void MNIST_layer () {
  auto train_images = MNIST_images<dev, T>("mnist/train-images.idx3-ubyte");
  std::cout << "train images loaded\n";
  auto train_labels = MNIST_labels<dev, T>("mnist/train-labels.idx1-ubyte");
  std::cout << "train labels loaded\n";
  auto test_images = MNIST_images<dev, T>("mnist/t10k-images.idx3-ubyte");
  std::cout << "test images loaded\n";
  auto test_labels = MNIST_labels<dev, T>("mnist/t10k-labels.idx1-ubyte");
  std::cout << "test labels loaded\n";
  // train_images /= 255.0;
  // test_images /= 255.0;
  std::cout << "data loaded\n";


  neural::Train<dev, T> t;
  t.add(new neural::NeuralNetwork<dev, T>(784, 300));
  t.add(new neural::NeuralNetwork<dev, T>(300, 10));

  t.alpha = 0.2;
  t.lambda = 0.003;
  t.momentum = 0.4;

  t.run(3000, 100, train_images, train_labels);

  auto result = t.forward(test_images);
  MNIST_test(result, test_images, test_labels);

}



template <tensor::Device dev, typename T>
void MNIST_sparse_layer () {
  auto train_images = MNIST_images<dev, T>("mnist/train-images.idx3-ubyte");
  auto train_labels = MNIST_labels<dev, T>("mnist/train-labels.idx1-ubyte");
  auto test_images = MNIST_images<tensor::CPU, T>("mnist/t10k-images.idx3-ubyte");
  auto test_labels = MNIST_labels<tensor::CPU, T>("mnist/t10k-labels.idx1-ubyte");
  train_images /= 255.0;
  test_images /= 255.0;
  std::cout << "data loaded\n";

  neural::Train<dev, T> t;
  t.add(new neural::Sparse<dev, T>(784, 300));
  t.add(new neural::NeuralNetwork<dev, T>(300, 10));

  t.alpha = 0.2;
  t.lambda = 0.003;

  std::cout << "start train\n";

  t.run(10000, 100, train_images, train_labels);

  tensor::Mat<tensor::CPU, T> result = t.forward(test_images);
  MNIST_test<tensor::CPU, T>(result, test_images, test_labels);

}


template <tensor::Device dev, typename T>
void MNIST_dnn_layer () {
  auto train_images = MNIST_images<dev, T>("mnist/train-images.idx3-ubyte");
  auto train_labels = MNIST_labels<dev, T>("mnist/train-labels.idx1-ubyte");
  auto test_images = MNIST_images<tensor::CPU, T>("mnist/t10k-images.idx3-ubyte");
  auto test_labels = MNIST_labels<tensor::CPU, T>("mnist/t10k-labels.idx1-ubyte");
  train_images /= 255.0;
  test_images /= 255.0;
  std::cout << "data loaded\n";

  int hidden_size = 600;
  int batch = 512;

  std::cout<<"batch: "<<batch<<std::endl;
  std::cout<<"hidden_size: "<<hidden_size<<std::endl;

  std::cout<<"Start train sparse layer 1"<<std::endl;

  neural::Train<dev, T> t1;
  t1.add(new neural::Sparse<dev, T>(784, hidden_size));
  t1.add(new neural::NeuralNetwork<dev, T>(hidden_size, 784));
  t1.alpha = 0.5;
  t1.lambda = 0.003;
  t1.momentum = 0.9;
  t1.run(5000, batch, train_images, train_images);
  t1.remove(1);

  auto new_train = t1.forward(train_images);

  std::cout<<"Start train sparse layer 2"<<std::endl;

  neural::Train<dev, T> t2;
  t2.add(new neural::Sparse<dev, T>(hidden_size, hidden_size));
  t2.add(new neural::NeuralNetwork<dev, T>(hidden_size, hidden_size));
  t2.alpha = 0.5;
  t2.lambda = 0.003;
  t2.momentum = 0.9;
  t2.run(5000, batch, new_train, new_train);
  t2.remove(1);

  new_train = t2.forward(new_train);

  std::cout<<"Start train softmax"<<std::endl;

  neural::Train<dev, T> ts;
  ts.add(new neural::Softmax<dev, T>(hidden_size, 10));
  // ts.add(new neural::NeuralNetwork<dev, T>(600, 10));
  ts.alpha = 0.5;
  ts.lambda = 0.003;
  ts.momentum = 0.9;
  ts.run(10000, batch, new_train, train_labels);

  std::cout<<"Start train fine tune"<<std::endl;

  neural::Train<dev, T> tf;
  std::static_pointer_cast< neural::Sparse<dev, T> >(t1.layers[0])->sparsing = false;
  std::static_pointer_cast< neural::Sparse<dev, T> >(t2.layers[0])->sparsing = false;
  tf.add(t1.layers[0]);
  tf.add(t2.layers[0]);
  tf.add(ts.layers[0]);
  tf.alpha = 0.1;
  tf.lambda = 0.0;
  tf.momentum = 0.9;
  tf.run(50000, batch, train_images, train_labels);

  std::cout<<"train done"<<std::endl;

  tensor::Mat<tensor::CPU, T> result = tf.forward(test_images);
  MNIST_test<tensor::CPU, T>(result, test_images, test_labels);
}






int main () {

  MNIST_dnn_layer<tensor::GPU, float>();

  return 0;
}
