# 使用重复元素的网络（VGG）

AlexNet在LeNet的基础上增加了3个卷积层。但AlexNet作者对它们的卷积窗口、输出通道数和构造顺序均做了大量的调整。虽然AlexNet指明了深度卷积神经网络可以取得出色的结果，但并没有提供简单的规则以指导后来的研究者如何设计新的网络。我们将在本章的后续几节里介绍几种不同的深度网络设计思路。

本节介绍VGG，它的名字来源于论文作者所在的实验室Visual Geometry Group [1]。VGG提出了可以通过重复使用简单的基础块来构建深度模型的思路。

## VGG块

VGG块的组成规律是：连续使用数个相同的填充为1、窗口形状为$3\times 3$的卷积层后接上一个步幅为2、窗口形状为$2\times 2$的最大池化层。卷积层保持输入的高和宽不变，而池化层则对其减半。我们使用`vgg_block`函数来实现这个基础的VGG块，它可以指定卷积层的数量`num_convs`和输出通道数`num_channels`。

```{.python .input  n=1}
                                                                                                                                            import d2lzh as d2l
import torch
import torch.nn as nn

def vgg_block(num_convs, in_channels, out_channels):
    layers=[]
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    
    blk = nn.Sequential(*layers)
    return blk
```

## VGG网络

与AlexNet和LeNet一样，VGG网络由卷积层模块后接全连接层模块构成。卷积层模块串联数个`vgg_block`，其超参数由变量`conv_arch`定义。该变量指定了每个VGG块里卷积层个数和输出通道数。全连接模块则跟AlexNet中的一样。

现在我们构造一个VGG网络。它有5个卷积块，前2块使用单卷积层，而后3块使用双卷积层。第一块的输出通道是64，之后每次对输出通道数翻倍，直到变为512。因为这个网络使用了8个卷积层和3个全连接层，所以经常被称为VGG-11。

```{.python .input  n=2}
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
```

下面我们实现VGG-11。

```{.python .input  n=3}
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
    
def vgg(conv_arch):
    conv_layers=[]
    in_channels = 1
    
    for (num_convs, out_channels) in conv_arch:
        conv_layers.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels
    
    net = nn.Sequential(
        *conv_layers,
        Flatten(),
        nn.Linear(7*7*512, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, 10)
    )
    return net

net = vgg(conv_arch)
```

下面构造一个高和宽均为224的单通道数据样本来观察每一层的输出形状。

```{.python .input  n=4}
X = torch.randn(size=(1,1,224,224), dtype=torch.float32)
for blk in net:
    X = blk(X)
    print(blk.__class__.__name__,'output shape:\t',X.shape)
```

```{.json .output n=4}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Sequential output shape:\t torch.Size([1, 64, 112, 112])\nSequential output shape:\t torch.Size([1, 128, 56, 56])\nSequential output shape:\t torch.Size([1, 256, 28, 28])\nSequential output shape:\t torch.Size([1, 512, 14, 14])\nSequential output shape:\t torch.Size([1, 512, 7, 7])\nFlatten output shape:\t torch.Size([1, 25088])\nLinear output shape:\t torch.Size([1, 4096])\nReLU output shape:\t torch.Size([1, 4096])\nDropout output shape:\t torch.Size([1, 4096])\nLinear output shape:\t torch.Size([1, 4096])\nReLU output shape:\t torch.Size([1, 4096])\nDropout output shape:\t torch.Size([1, 4096])\nLinear output shape:\t torch.Size([1, 10])\n"
 }
]
```

可以看到，每次我们将输入的高和宽减半，直到最终高和宽变成7后传入全连接层。与此同时，输出通道数每次翻倍，直到变成512。因为每个卷积层的窗口大小一样，所以每层的模型参数尺寸和计算复杂度与输入高、输入宽、输入通道数和输出通道数的乘积成正比。VGG这种高和宽减半以及通道翻倍的设计使得多数卷积层都有相同的模型参数尺寸和计算复杂度。

## 获取数据和训练模型


除了使用了稍大些的学习率，模型训练过程与上一节的AlexNet中的类似。

```{.python .input  n=12}
lr, num_epochs, batch_size, device = 0.05, 5, 64, d2l.try_gpu()
def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)

net.apply(init_weights)
net = net.to(device)

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
criterion = nn.CrossEntropyLoss()
d2l.train_ch5(net, train_iter, test_iter, criterion, num_epochs, batch_size, device, lr)
```

```{.json .output n=12}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "training on cuda:0\n"
 }
]
```

## 小结

* VGG-11通过5个可以重复使用的卷积块来构造网络。根据每块里卷积层个数和输出通道数的不同可以定义出不同的VGG模型。

## 练习

* 与AlexNet相比，VGG通常计算慢很多，也需要更多的内存或显存。试分析原因。
* 尝试将Fashion-MNIST中图像的高和宽由224改为96。这在实验中有哪些影响？
* 参考VGG论文里的表1来构造VGG其他常用模型，如VGG-16和VGG-19 [1]。



## 参考文献

[1] Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556.

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/1277)

![](../img/qr_vgg.svg)
