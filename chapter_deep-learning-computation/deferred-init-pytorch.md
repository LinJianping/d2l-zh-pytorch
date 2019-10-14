# Pytorch没有延后初始化

Pytorch（或任何其他框架）无法预测网络的输入维数。 稍后，当使用卷积网络和图像时，此问题将变得更加相关，因为输入维数（即图像的分辨率）将在很长的范围内影响后续图层的维数。 因此，在编写代码时无需知道参数的维数即可设置参数的能力可以极大地简化统计建模。 在下面的内容中，我们将以初始化为例讨论其工作方式。 毕竟，我们无法初始化我们不知道存在的变量。

```{.python .input  n=1}
import torch
import torch.nn as nn
```

```{.python .input  n=2}
def getnet(in_features,out_features):
    net=nn.Sequential(
    nn.Linear(in_features,256),
    nn.ReLU(),
    nn.Linear(256,out_features))
    
    return net

net=getnet(20,10)
```

```{.python .input  n=3}
for name,param in net.named_parameters():
    print(name,param.shape)
```

```{.json .output n=3}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "0.weight torch.Size([256, 20])\n0.bias torch.Size([256])\n2.weight torch.Size([10, 256])\n2.bias torch.Size([10])\n"
 }
]
```

现在我们知道了它的理论原理，让我们看看初始化实际上是在什么时候触发的。 为此，我们模拟了一个初始化器。初始化时，初始化器__init_weights__会初始化网络的权重，还将神经网络的权重设置为非零值，这有助于神经网络倾向于卡在本地 最小值，因此最好给他们提供许多不同的初始值。 如果它们都从零开始，则不能这样做。

```{.python .input  n=5}
def init_weights(m):
    print("Init", m)

net.apply(init_weights)
print(net[0].weight)
print(net[2].weight)
```

```{.json .output n=5}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Init Linear(in_features=20, out_features=256, bias=True)\nInit ReLU()\nInit Linear(in_features=256, out_features=10, bias=True)\nInit Sequential(\n  (0): Linear(in_features=20, out_features=256, bias=True)\n  (1): ReLU()\n  (2): Linear(in_features=256, out_features=10, bias=True)\n)\nParameter containing:\ntensor([[-0.0640,  0.2114,  0.0559,  ...,  0.1718,  0.1695,  0.1778],\n        [ 0.0199,  0.0930,  0.1277,  ..., -0.1570, -0.1740, -0.1532],\n        [-0.1335, -0.1422, -0.0701,  ..., -0.0086,  0.1304, -0.1691],\n        ...,\n        [-0.1716,  0.1959, -0.2026,  ..., -0.1750,  0.0687, -0.0357],\n        [ 0.0101,  0.1555, -0.1843,  ...,  0.2191,  0.1538, -0.1546],\n        [ 0.1761,  0.0922, -0.1110,  ..., -0.1021,  0.0154,  0.0729]],\n       requires_grad=True)\nParameter containing:\ntensor([[-0.0264,  0.0319, -0.0296,  ...,  0.0204, -0.0403,  0.0604],\n        [-0.0483, -0.0047,  0.0155,  ...,  0.0518,  0.0284, -0.0389],\n        [ 0.0267, -0.0152, -0.0012,  ...,  0.0051,  0.0463, -0.0069],\n        ...,\n        [-0.0026, -0.0261,  0.0462,  ..., -0.0144,  0.0486, -0.0254],\n        [ 0.0499, -0.0604,  0.0409,  ..., -0.0508, -0.0416, -0.0050],\n        [ 0.0176,  0.0617, -0.0445,  ..., -0.0290, -0.0394,  0.0560]],\n       requires_grad=True)\n"
 }
]
```

## 小结

* 系统将真正的参数初始化延后到获得足够信息时才执行的行为叫作延后初始化。
* 延后初始化的主要好处是让模型构造更加简单。例如，我们无须人工推测每个层的输入个数。
* 也可以避免延后初始化。


## 练习

* 如果在下一次前向计算`net(X)`前改变输入`X`的形状，包括批量大小和输入个数，会发生什么？



## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/6320)

![](../img/qr_deferred-init.svg)
