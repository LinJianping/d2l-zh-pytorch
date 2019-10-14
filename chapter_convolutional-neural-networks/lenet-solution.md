# 卷积神经网络（LeNet）

在[“多层感知机的从零开始实现”](../chapter_deep-learning-basics/mlp-scratch.md)一节里我们构造了一个含单隐藏层的多层感知机模型来对Fashion-MNIST数据集中的图像进行分类。每张图像高和宽均是28像素。我们将图像中的像素逐行展开，得到长度为784的向量，并输入进全连接层中。然而，这种分类方法有一定的局限性。

1. 图像在同一列邻近的像素在这个向量中可能相距较远。它们构成的模式可能难以被模型识别。
2. 对于大尺寸的输入图像，使用全连接层容易导致模型过大。假设输入是高和宽均为$1,000$像素的彩色照片（含3个通道）。即使全连接层输出个数仍是256，该层权重参数的形状也是$3,000,000\times 256$：它占用了大约3 GB的内存或显存。这会带来过于复杂的模型和过高的存储开销。

卷积层尝试解决这两个问题。一方面，卷积层保留输入形状，使图像的像素在高和宽两个方向上的相关性均可能被有效识别；另一方面，卷积层通过滑动窗口将同一卷积核与不同位置的输入重复计算，从而避免参数尺寸过大。

卷积神经网络就是含卷积层的网络。本节里我们将介绍一个早期用来识别手写数字图像的卷积神经网络：LeNet [1]。这个名字来源于LeNet论文的第一作者Yann LeCun。LeNet展示了通过梯度下降训练卷积神经网络可以达到手写数字识别在当时最先进的结果。这个奠基性的工作第一次将卷积神经网络推上舞台，为世人所知。

## LeNet模型

LeNet分为卷积层块和全连接层块两个部分。下面我们分别介绍这两个模块。

卷积层块里的基本单位是卷积层后接最大池化层：卷积层用来识别图像里的空间模式，如线条和物体局部，之后的最大池化层则用来降低卷积层对位置的敏感性。卷积层块由两个这样的基本单位重复堆叠构成。在卷积层块中，每个卷积层都使用$5\times 5$的窗口，并在输出上使用sigmoid激活函数。第一个卷积层输出通道数为6，第二个卷积层输出通道数则增加到16。这是因为第二个卷积层比第一个卷积层的输入的高和宽要小，所以增加输出通道使两个卷积层的参数尺寸类似。卷积层块的两个最大池化层的窗口形状均为$2\times 2$，且步幅为2。由于池化窗口与步幅形状相同，池化窗口在输入上每次滑动所覆盖的区域互不重叠。

卷积层块的输出形状为(批量大小, 通道, 高, 宽)。当卷积层块的输出传入全连接层块时，全连接层块会将小批量中每个样本变平（flatten）。也就是说，全连接层的输入形状将变成二维，其中第一维是小批量中的样本，第二维是每个样本变平后的向量表示，且向量长度为通道、高和宽的乘积。全连接层块含3个全连接层。它们的输出个数分别是120、84和10，其中10为输出的类别个数。

下面我们通过`Sequential`类来实现LeNet模型。

```{.python .input  n=12}
import d2lzh as d2l
import time
import torch
import torch.nn as nn

class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)

class Reshape(torch.nn.Module):
    def forward(self, x):
        return x.view(-1,1,28,28)

net = nn.Sequential(Reshape(),
                    nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2),
                    nn.Sigmoid(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
                    nn.Sigmoid(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    Flatten(),
                    nn.Linear(5*5*16, 120),
                    nn.Sigmoid(),
                    nn.Linear(120, 84),
                    nn.Sigmoid(),
                    nn.Linear(84, 10)
                   )

```

接下来我们构造一个高和宽均为28的单通道数据样本，并逐层进行前向计算来查看每个层的输出形状。

```{.python .input  n=13}
X = torch.randn(size=(1, 1, 28, 28))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)
```

```{.json .output n=13}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Reshape output shape:\t torch.Size([1, 1, 28, 28])\nConv2d output shape:\t torch.Size([1, 6, 28, 28])\nSigmoid output shape:\t torch.Size([1, 6, 28, 28])\nMaxPool2d output shape:\t torch.Size([1, 6, 14, 14])\nConv2d output shape:\t torch.Size([1, 16, 10, 10])\nSigmoid output shape:\t torch.Size([1, 16, 10, 10])\nMaxPool2d output shape:\t torch.Size([1, 16, 5, 5])\nFlatten output shape:\t torch.Size([1, 400])\nLinear output shape:\t torch.Size([1, 120])\nSigmoid output shape:\t torch.Size([1, 120])\nLinear output shape:\t torch.Size([1, 84])\nSigmoid output shape:\t torch.Size([1, 84])\nLinear output shape:\t torch.Size([1, 10])\n"
 }
]
```

可以看到，在卷积层块中输入的高和宽在逐层减小。卷积层由于使用高和宽均为5的卷积核，从而将高和宽分别减小4，而池化层则将高和宽减半，但通道数则从1增加到16。全连接层则逐层减少输出个数，直到变成图像的类别数10。


## 获取数据和训练模型

下面我们来实验LeNet模型。实验中，我们仍然使用Fashion-MNIST作为训练数据集。

```{.python .input  n=14}
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)
```

因为卷积神经网络计算比多层感知机要复杂，建议使用GPU来加速计算。

```{.python .input  n=15}
def try_gpu():  # 本函数已保存在d2lzh包中方便以后使用
    """If GPU is available, return torch.device as cuda:0; else return torch.device as cpu."""
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    return device

device = try_gpu()
device
```

```{.json .output n=15}
[
 {
  "data": {
   "text/plain": "device(type='cuda', index=0)"
  },
  "execution_count": 15,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

相应地，我们对[“softmax回归的从零开始实现”](../chapter_deep-learning-basics/softmax-regression-scratch.md)一节中描述的`evaluate_accuracy`函数略作修改。由于数据刚开始存在CPU使用的内存上，当`device`变量代表GPU及相应的显存时，我们通过[“GPU计算”](../chapter_deep-learning-computation/use-gpu.md)一节中介绍的`to`函数将数据复制到显存上，例如`cuda:0`。

```{.python .input  n=16}
# 本函数已保存在d2lzh包中方便以后使用。该函数将被逐步改进：它的完整实现将在“图像增广”一节中
# 描述
def evaluate_accuracy(data_iter, net, device):
    acc_sum,n = torch.tensor([0],dtype=torch.float32,device=device),0
    for X,y in data_iter:
        X = X.to(device)
        y= y.to(device)
        net.eval()
        with torch.no_grad():
            y = y.long()
            acc_sum += ((net(X)).argmax(dim=1) == y).sum()
            n += y.shape[0]
    return acc_sum.item() / n
```

我们同样对[“softmax回归的从零开始实现”](../chapter_deep-learning-basics/softmax-regression-scratch.md)一节中定义的`train_ch3`函数略作修改，确保计算使用的数据和模型同在内存或显存上。

```{.python .input  n=19}
# 本函数已保存在d2lzh包中方便以后使用
def train_ch5(net, train_iter, test_iter,criterion, num_epochs, batch_size, device,lr=None):
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    for epoch in range(num_epochs):
        train_l_sum = torch.tensor([0.0],dtype=torch.float32,device=device)
        train_acc_sum = torch.tensor([0.0],dtype=torch.float32,device=device)
        n, start = 0, time.time()
        for X,y in train_iter:
            X,y = X.to(device), y.to(device)
            net.train()
            optimizer.zero_grad()
            pred_y = net(X)
            train_l = criterion(pred_y, y)
            train_l.backward()
            optimizer.step()
            with torch.no_grad():
                y =y.long()
                train_l_sum += train_l
                train_acc_sum += (torch.argmax(pred_y, dim=1) == y).sum()
                n += y.shape[0]
            
        test_acc = evaluate_accuracy(test_iter,net,device)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, '
              'time %.1f sec'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc,
                 time.time() - start))
```

我们重新将模型参数初始化到设备变量`device`之上，并使用Xavier随机初始化。损失函数和训练算法则依然使用交叉熵损失函数和小批量随机梯度下降。

```{.python .input  n=20}
lr, num_epochs = 0.9, 5
def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
net.apply(init_weights)
net = net.to(device)

criterion = nn.CrossEntropyLoss()
train_ch5(net, train_iter, test_iter, criterion,num_epochs, batch_size,device, lr)
```

```{.json .output n=20}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "training on cuda:0\nepoch 1, loss 0.0091, train acc 0.102, test acc 0.100, time 1.9 sec\nepoch 2, loss 0.0075, train acc 0.254, test acc 0.478, time 2.2 sec\nepoch 3, loss 0.0037, train acc 0.615, test acc 0.626, time 2.1 sec\nepoch 4, loss 0.0029, train acc 0.710, test acc 0.734, time 2.2 sec\nepoch 5, loss 0.0025, train acc 0.753, test acc 0.739, time 2.1 sec\n"
 }
]
```

## 小结

* 卷积神经网络就是含卷积层的网络。
* LeNet交替使用卷积层和最大池化层后接全连接层来进行图像分类。

## 练习

* 尝试基于LeNet构造更复杂的网络来提高分类准确率。例如，调整卷积窗口大小、输出通道数、激活函数和全连接层输出个数。在优化方面，可以尝试使用不同的学习率、初始化方法以及增加迭代周期。




## 参考文献

[1] LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), 2278-2324.

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/737)

![](../img/qr_lenet.svg)
