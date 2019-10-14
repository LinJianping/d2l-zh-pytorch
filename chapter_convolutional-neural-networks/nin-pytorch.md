# 网络中的网络（NiN）

前几节介绍的LeNet、AlexNet和VGG在设计上的共同之处是：先以由卷积层构成的模块充分抽取空间特征，再以由全连接层构成的模块来输出分类结果。其中，AlexNet和VGG对LeNet的改进主要在于如何对这两个模块加宽（增加通道数）和加深。本节我们介绍网络中的网络（NiN）[1]。它提出了另外一个思路，即串联多个由卷积层和“全连接”层构成的小网络来构建一个深层网络。


## NiN块

我们知道，卷积层的输入和输出通常是四维数组（样本，通道，高，宽），而全连接层的输入和输出则通常是二维数组（样本，特征）。如果想在全连接层后再接上卷积层，则需要将全连接层的输出变换为四维。回忆在[“多输入通道和多输出通道”](channels.md)一节里介绍的$1\times 1$卷积层。它可以看成全连接层，其中空间维度（高和宽）上的每个元素相当于样本，通道相当于特征。因此，NiN使用$1\times 1$卷积层来替代全连接层，从而使空间信息能够自然传递到后面的层中去。图5.7对比了NiN同AlexNet和VGG等网络在结构上的主要区别。

![左图是AlexNet和VGG的网络结构局部，右图是NiN的网络结构局部](../img/nin.svg)

NiN块是NiN中的基础块。它由一个卷积层加两个充当全连接层的$1\times 1$卷积层串联而成。其中第一个卷积层的超参数可以自行设置，而第二和第三个卷积层的超参数一般是固定的。

```{.python .input  n=1}
import d2lzh as d2l
import torch
import torch.nn as nn
```

```{.python .input  n=17}
def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    blk = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
                        nn.ReLU(),
                        nn.Conv2d(out_channels, out_channels, 1),
                        nn.ReLU(),
                        nn.Conv2d(out_channels, out_channels, 1),
                        nn.ReLU())
    return blk
```

## NiN模型

NiN是在AlexNet问世不久后提出的。它们的卷积层设定有类似之处。NiN使用卷积窗口形状分别为$11\times 11$、$5\times 5$和$3\times 3$的卷积层，相应的输出通道数也与AlexNet中的一致。每个NiN块后接一个步幅为2、窗口形状为$3\times 3$的最大池化层。

除使用NiN块以外，NiN还有一个设计与AlexNet显著不同：NiN去掉了AlexNet最后的3个全连接层，取而代之地，NiN使用了输出通道数等于标签类别数的NiN块，然后使用全局平均池化层对每个通道中所有元素求平均并直接用于分类。这里的全局平均池化层即窗口形状等于输入空间维形状的平均池化层。NiN的这个设计的好处是可以显著减小模型参数尺寸，从而缓解过拟合。然而，该设计有时会造成获得有效模型的训练时间的增加。

```{.python .input  n=18}
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
net = nn.Sequential(nin_block(1, 96, 11, 4, 0),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                    nin_block(96, 256, 5, 1, 2),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                    nin_block(256, 384, 3, 1, 1),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                    nn.Dropout(0.5),
                    nin_block(384, 10, 3, 1, 1),
                    nn.AvgPool2d(kernel_size=5),
                    Flatten()
                   )
```

我们构建一个数据样本来查看每一层的输出形状。

```{.python .input  n=19}
X = torch.randn(size=(1, 1, 224, 224))

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)

net.apply(init_weights)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)
```

```{.json .output n=19}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Sequential output shape:\t torch.Size([1, 96, 54, 54])\nMaxPool2d output shape:\t torch.Size([1, 96, 26, 26])\nSequential output shape:\t torch.Size([1, 256, 26, 26])\nMaxPool2d output shape:\t torch.Size([1, 256, 12, 12])\nSequential output shape:\t torch.Size([1, 384, 12, 12])\nMaxPool2d output shape:\t torch.Size([1, 384, 5, 5])\nDropout output shape:\t torch.Size([1, 384, 5, 5])\nSequential output shape:\t torch.Size([1, 10, 5, 5])\nAvgPool2d output shape:\t torch.Size([1, 10, 1, 1])\nFlatten output shape:\t torch.Size([1, 10])\n"
 }
]
```

## 获取数据和训练模型

我们依然使用Fashion-MNIST数据集来训练模型。NiN的训练与AlexNet和VGG的类似，但这里使用的学习率更大。

```{.python .input  n=20}
lr, num_epochs, batch_size, device = 0.1, 5, 128, d2l.try_gpu()

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)

#criterion
criterion = nn.CrossEntropyLoss()

d2l.train_ch5(net, train_iter, test_iter, criterion, num_epochs, batch_size, device, lr)
```

```{.json .output n=20}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "training on cuda:0\nepoch 1, loss 0.0143, train acc 0.303, test acc 0.406, time 39.1 sec\nepoch 2, loss 0.0144, train acc 0.346, test acc 0.347, time 39.6 sec\nepoch 3, loss 0.0138, train acc 0.346, test acc 0.365, time 39.4 sec\nepoch 4, loss 0.0131, train acc 0.379, test acc 0.382, time 39.8 sec\nepoch 5, loss 0.0126, train acc 0.404, test acc 0.413, time 40.7 sec\n"
 }
]
```

## 小结

* NiN重复使用由卷积层和代替全连接层的$1\times 1$卷积层构成的NiN块来构建深层网络。
* NiN去除了容易造成过拟合的全连接输出层，而是将其替换成输出通道数等于标签类别数的NiN块和全局平均池化层。
* NiN的以上设计思想影响了后面一系列卷积神经网络的设计。

## 练习

* 调节超参数，提高分类准确率。
* 为什么NiN块里要有两个$1\times 1$卷积层？去除其中的一个，观察并分析实验现象。




## 参考文献

[1] Lin, M., Chen, Q., & Yan, S. (2013). Network in network. arXiv preprint arXiv:1312.4400.

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/1661)

![](../img/qr_nin.svg)
