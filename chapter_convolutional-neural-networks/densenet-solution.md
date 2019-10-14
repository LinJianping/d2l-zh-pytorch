# 稠密连接网络（DenseNet）

ResNet中的跨层连接设计引申出了数个后续工作。本节我们介绍其中的一个：稠密连接网络（DenseNet） [1]。 它与ResNet的主要区别如图5.10所示。

![ResNet（左）与DenseNet（右）在跨层连接上的主要区别：使用相加和使用连结](../img/densenet.svg)

图5.10中将部分前后相邻的运算抽象为模块$A$和模块$B$。与ResNet的主要区别在于，DenseNet里模块$B$的输出不是像ResNet那样和模块$A$的输出相加，而是在通道维上连结。这样模块$A$的输出可以直接传入模块$B$后面的层。在这个设计里，模块$A$直接跟模块$B$后面的所有层连接在了一起。这也是它被称为“稠密连接”的原因。

DenseNet的主要构建模块是稠密块（dense block）和过渡层（transition layer）。前者定义了输入和输出是如何连结的，后者则用来控制通道数，使之不过大。


## 稠密块

DenseNet使用了ResNet改良版的“批量归一化、激活和卷积”结构（参见上一节的练习），我们首先在`conv_block`函数里实现这个结构。

```{.python .input  n=27}
import d2lzh as d2l
import torch
import torch.nn as nn
```

```{.python .input  n=28}
def conv_block(in_channels, out_channels):
    blk = nn.Sequential(nn.BatchNorm2d(in_channels),
                       nn.ReLU(),
                       nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
    return blk
```

稠密块由多个`conv_block`组成，每块使用相同的输出通道数。但在前向计算时，我们将每块的输入和输出在通道维上连结。

```{.python .input  n=29}
class DenseBlock(nn.Module):
    def __init__(self, num_convs, in_channels, num_channels, **kwargs):
        super(DenseBlock, self).__init__(**kwargs)
        blk = []
        for i in range(num_convs):
            blk.append(conv_block((i * num_channels + in_channels), num_channels))
        self.net = nn.Sequential(*blk)
    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X, Y), dim=1)  # 在通道维上将输入和输出连结
        return X
```

在下面的例子中，我们定义一个有2个输出通道数为10的卷积块。使用通道数为3的输入时，我们会得到通道数为$3+2\times 10=23$的输出。卷积块的通道数控制了输出通道数相对于输入通道数的增长，因此也被称为增长率（growth rate）。

```{.python .input  n=30}
blk = DenseBlock(2, 3, 10)
X = torch.randn(size=(4, 3, 8, 8))
Y = blk(X)
Y.shape
```

```{.json .output n=30}
[
 {
  "data": {
   "text/plain": "torch.Size([4, 23, 8, 8])"
  },
  "execution_count": 30,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## 过渡层

由于每个稠密块都会带来通道数的增加，使用过多则会带来过于复杂的模型。过渡层用来控制模型复杂度。它通过$1\times1$卷积层来减小通道数，并使用步幅为2的平均池化层减半高和宽，从而进一步降低模型复杂度。

```{.python .input  n=31}
def transition_block(in_channels, out_channels):
    blk = nn.Sequential(nn.BatchNorm2d(in_channels),
                        nn.ReLU(),
                        nn.Conv2d(in_channels, out_channels, kernel_size=1),
                        nn.AvgPool2d(kernel_size=2, stride=2))
    return blk
```

对上一个例子中稠密块的输出使用通道数为10的过渡层。此时输出的通道数减为10，高和宽均减半。

```{.python .input  n=32}
blk = transition_block(23, 10)
blk(Y).shape
```

```{.json .output n=32}
[
 {
  "data": {
   "text/plain": "torch.Size([4, 10, 4, 4])"
  },
  "execution_count": 32,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## DenseNet模型

我们来构造DenseNet模型。DenseNet首先使用同ResNet一样的单卷积层和最大池化层。

```{.python .input  n=33}
layers = []
layers.append(nn.Conv2d(1, 64, kernel_size=7, padding=3, stride=2))
layers.append(nn.BatchNorm2d(64))
layers.append(nn.ReLU())
layers.append(nn.MaxPool2d(kernel_size=3, padding=1, stride=2))
```

类似于ResNet接下来使用的4个残差块，DenseNet使用的是4个稠密块。同ResNet一样，我们可以设置每个稠密块使用多少个卷积层。这里我们设成4，从而与上一节的ResNet-18保持一致。稠密块里的卷积层通道数（即增长率）设为32，所以每个稠密块将增加128个通道。

ResNet里通过步幅为2的残差块在每个模块之间减小高和宽。这里我们则使用过渡层来减半高和宽，并减半通道数。

```{.python .input  n=34}
num_channels, growth_rate = 64, 32  # num_channels为当前的通道数
num_convs_in_dense_blocks = [4, 4, 4, 4]

for i, num_convs in enumerate(num_convs_in_dense_blocks):
    layers.append(DenseBlock(num_convs, num_channels, growth_rate))
    # 上一个稠密块的输出通道数
    num_channels += num_convs * growth_rate
    # 在稠密块之间加入通道数减半的过渡层
    if i != len(num_convs_in_dense_blocks) - 1:
        layers.append(transition_block(num_channels, num_channels // 2))
        num_channels //= 2
```

同ResNet一样，最后接上全局池化层和全连接层来输出。

```{.python .input  n=35}
class Flatten(nn.Module):
    def forward(self, X):
        return X.view(X.size(0), -1)
layers.append(nn.BatchNorm2d(num_channels))
layers.append(nn.ReLU())
layers.append(nn.AdaptiveAvgPool2d((1,1)))
layers.append(Flatten())
layers.append(nn.Linear(num_channels, 10))
net = nn.Sequential(*layers)
```

## 获取数据并训练模型

由于这里使用了比较深的网络，本节里我们将输入高和宽从224降到96来简化计算。

```{.python .input  n=37}
lr, num_epochs, batch_size, device = 0.1, 5, 64, d2l.try_gpu()
def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        
net.apply(init_weights)
net = net.to(device)
criterion = nn.CrossEntropyLoss()
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch5(net, train_iter, test_iter, criterion, num_epochs, batch_size, device, lr)
```

```{.json .output n=37}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "training on cuda:0\nepoch 1, loss 0.0070, train acc 0.839, test acc 0.855, time 28.8 sec\nepoch 2, loss 0.0044, train acc 0.898, test acc 0.833, time 28.7 sec\nepoch 3, loss 0.0036, train acc 0.915, test acc 0.843, time 28.7 sec\nepoch 4, loss 0.0032, train acc 0.924, test acc 0.903, time 28.7 sec\nepoch 5, loss 0.0029, train acc 0.932, test acc 0.909, time 28.8 sec\n"
 }
]
```

```{.python .input  n=44}
for params in net.named_parameters():
    (name, param) = params
    print(name, param.shape)
```

```{.json .output n=44}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "0.weight torch.Size([64, 1, 7, 7])\n0.bias torch.Size([64])\n1.weight torch.Size([64])\n1.bias torch.Size([64])\n4.net.0.0.weight torch.Size([64])\n4.net.0.0.bias torch.Size([64])\n4.net.0.2.weight torch.Size([32, 64, 3, 3])\n4.net.0.2.bias torch.Size([32])\n4.net.1.0.weight torch.Size([96])\n4.net.1.0.bias torch.Size([96])\n4.net.1.2.weight torch.Size([32, 96, 3, 3])\n4.net.1.2.bias torch.Size([32])\n4.net.2.0.weight torch.Size([128])\n4.net.2.0.bias torch.Size([128])\n4.net.2.2.weight torch.Size([32, 128, 3, 3])\n4.net.2.2.bias torch.Size([32])\n4.net.3.0.weight torch.Size([160])\n4.net.3.0.bias torch.Size([160])\n4.net.3.2.weight torch.Size([32, 160, 3, 3])\n4.net.3.2.bias torch.Size([32])\n5.0.weight torch.Size([192])\n5.0.bias torch.Size([192])\n5.2.weight torch.Size([96, 192, 1, 1])\n5.2.bias torch.Size([96])\n6.net.0.0.weight torch.Size([96])\n6.net.0.0.bias torch.Size([96])\n6.net.0.2.weight torch.Size([32, 96, 3, 3])\n6.net.0.2.bias torch.Size([32])\n6.net.1.0.weight torch.Size([128])\n6.net.1.0.bias torch.Size([128])\n6.net.1.2.weight torch.Size([32, 128, 3, 3])\n6.net.1.2.bias torch.Size([32])\n6.net.2.0.weight torch.Size([160])\n6.net.2.0.bias torch.Size([160])\n6.net.2.2.weight torch.Size([32, 160, 3, 3])\n6.net.2.2.bias torch.Size([32])\n6.net.3.0.weight torch.Size([192])\n6.net.3.0.bias torch.Size([192])\n6.net.3.2.weight torch.Size([32, 192, 3, 3])\n6.net.3.2.bias torch.Size([32])\n7.0.weight torch.Size([224])\n7.0.bias torch.Size([224])\n7.2.weight torch.Size([112, 224, 1, 1])\n7.2.bias torch.Size([112])\n8.net.0.0.weight torch.Size([112])\n8.net.0.0.bias torch.Size([112])\n8.net.0.2.weight torch.Size([32, 112, 3, 3])\n8.net.0.2.bias torch.Size([32])\n8.net.1.0.weight torch.Size([144])\n8.net.1.0.bias torch.Size([144])\n8.net.1.2.weight torch.Size([32, 144, 3, 3])\n8.net.1.2.bias torch.Size([32])\n8.net.2.0.weight torch.Size([176])\n8.net.2.0.bias torch.Size([176])\n8.net.2.2.weight torch.Size([32, 176, 3, 3])\n8.net.2.2.bias torch.Size([32])\n8.net.3.0.weight torch.Size([208])\n8.net.3.0.bias torch.Size([208])\n8.net.3.2.weight torch.Size([32, 208, 3, 3])\n8.net.3.2.bias torch.Size([32])\n9.0.weight torch.Size([240])\n9.0.bias torch.Size([240])\n9.2.weight torch.Size([120, 240, 1, 1])\n9.2.bias torch.Size([120])\n10.net.0.0.weight torch.Size([120])\n10.net.0.0.bias torch.Size([120])\n10.net.0.2.weight torch.Size([32, 120, 3, 3])\n10.net.0.2.bias torch.Size([32])\n10.net.1.0.weight torch.Size([152])\n10.net.1.0.bias torch.Size([152])\n10.net.1.2.weight torch.Size([32, 152, 3, 3])\n10.net.1.2.bias torch.Size([32])\n10.net.2.0.weight torch.Size([184])\n10.net.2.0.bias torch.Size([184])\n10.net.2.2.weight torch.Size([32, 184, 3, 3])\n10.net.2.2.bias torch.Size([32])\n10.net.3.0.weight torch.Size([216])\n10.net.3.0.bias torch.Size([216])\n10.net.3.2.weight torch.Size([32, 216, 3, 3])\n10.net.3.2.bias torch.Size([32])\n11.weight torch.Size([248])\n11.bias torch.Size([248])\n15.weight torch.Size([10, 248])\n15.bias torch.Size([10])\n"
 }
]
```

## 小结

* 在跨层连接上，不同于ResNet中将输入与输出相加，DenseNet在通道维上连结输入与输出。
* DenseNet的主要构建模块是稠密块和过渡层。

## 练习

* DenseNet论文中提到的一个优点是模型参数比ResNet的更小，这是为什么？
* DenseNet被人诟病的一个问题是内存或显存消耗过多。真的会这样吗？可以把输入形状换成$224\times 224$，来看看实际的消耗。
* 实现DenseNet论文中的表1提出的不同版本的DenseNet [1]。



## 参考文献

[1] Huang, G., Liu, Z., Weinberger, K. Q., & van der Maaten, L. (2017). Densely connected convolutional networks. In Proceedings of the IEEE conference on computer vision and pattern recognition (Vol. 1, No. 2).

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/1664)

![](../img/qr_densenet.svg)

```{.python .input}

```
