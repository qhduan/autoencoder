# 深度学习实验

## tensor

符号化矩阵库，基于cuda，使用blas（openblas，cublas），C++11

API参考Armadillo (http://arma.sourceforge.net/docs.html)

## neural

包括神经网络，稀疏编码器，softmax回归

算法主要参考有：斯坦福的UFLDL，http://eric-yuan.me/， https://github.com/jatinshah/ufldl_tutorial

测试结果

MNIST，数据/=255.0
两层稀疏编码器，隐藏层600个神经元，SGD算法，batch为512


第一层 稀疏层
  t1.alpha = 0.5; // 学习率
  t1.lambda = 0.003; // L2
  t1.momentum = 0.9;
  t1.run(5000, batch, train_images, train_images); // 5000次GD

第二层 稀疏层
  t2.alpha = 0.5;
  t2.lambda = 0.003;
  t2.momentum = 0.9;
  t2.run(5000, batch, new_train, new_train);

第三层 softmax
  ts.alpha = 0.5;
  ts.lambda = 0.003;
  ts.momentum = 0.9;
  ts.run(10000, batch, new_train, train_labels);

最终调整：
  tf.alpha = 0.1;
  tf.lambda = 0.0;
  tf.momentum = 0.9;
  tf.run(50000, batch, train_images, train_labels);

最终正确率，在MNIST测试集上的binary准确率，不是cost：

total: 10000, right: 9771, ratio: 0.9771

错误率 2.29%
