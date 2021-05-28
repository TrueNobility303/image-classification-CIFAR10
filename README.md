# cifar 图像分类

## 摘要

* 自实现CNN用于Cifar10图像分类，网络架构中包括了残差连接，attention模块等，并做了不同架构、不同优化器(eg.AdamW)等的探究，并且可以在20 epoch 内达到84%的准确率，在200 epoch内达到90%以上的准确率
* 复现了WRN + SAM的模型，在Cifar10数据集200 epoch内达到了97.2%的准确率
* 使用不同方法试图解释CNN
* 探究了BatchNorm对Liptitz平滑化的影响
* 使用了DessiLBI作为优化器，训练一个高准确率但稀疏的网络，并且进行剪枝等操作
