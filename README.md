# CIFAR10Recognition

## 数据集介绍

本项目采用CIFAR-10数据集，该数据集包含10个类型，每个类型下面都包含6000张图片，每张图片的大小为32*32，是三通道的RGB图片，因此每张图片的数据tensor大小为

$$
3*32*32
$$

## 网络结构介绍

该网络采用三层卷积结构，加上一层dropout和线性层



![](https://s3.bmp.ovh/imgs/2022/11/01/22dd627c27780afd.png)

![](https://s3.bmp.ovh/imgs/2022/11/01/447352f3d9a271ba.png)

## 最终效果

最终识别正确率为64%

![](https://s3.bmp.ovh/imgs/2022/11/01/81eea0527c0eb6af.png)