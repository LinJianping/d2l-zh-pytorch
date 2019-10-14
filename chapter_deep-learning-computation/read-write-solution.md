# 读取和存储

到目前为止，我们介绍了如何处理数据以及如何构建、训练和测试深度学习模型。然而在实际中，我们有时需要把训练好的模型部署到很多不同的设备。在这种情况下，我们可以把内存中训练好的模型参数存储在硬盘上供后续读取使用。


## 读写`Tensor`

我们可以直接使用`save`函数和`load`函数分别存储和读取`Tensor`。下面的例子创建了`Tensor`变量`x`，并将其存在文件名同为`x`的文件里。

```{.python .input  n=1}
import torch
import torch.nn as nn
```

```{.python .input  n=3}
x = torch.ones(3)
torch.save(x, 'x')
```

然后我们将数据从存储的文件读回内存。

```{.python .input  n=4}
x2 = torch.load('x')
x2
```

```{.json .output n=4}
[
 {
  "data": {
   "text/plain": "tensor([1., 1., 1.])"
  },
  "execution_count": 4,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

我们还可以存储一列`Tensor`并读回内存。

```{.python .input  n=6}
y = torch.zeros(4)
torch.save([x, y], 'xy')
x2, y2 = torch.load('xy')
(x2, y2)
```

```{.json .output n=6}
[
 {
  "data": {
   "text/plain": "(tensor([1., 1., 1.]), tensor([0., 0., 0., 0.]))"
  },
  "execution_count": 6,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

我们甚至可以存储并读取一个从字符串映射到`NDArray`的字典。

```{.python .input  n=8}
mydict = {'x': x, 'y': y}
torch.save( mydict, 'mydict')
mydict2 = torch.load('mydict')
mydict2
```

```{.json .output n=8}
[
 {
  "data": {
   "text/plain": "{'x': tensor([1., 1., 1.]), 'y': tensor([0., 0., 0., 0.])}"
  },
  "execution_count": 8,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## 读写Pytorch模型的参数

除`Tensor`以外，我们还可以读写Pytorch模型的参数。Pytorch的`Module`类提供了`state_dict()`函数来获取模型参数。为了演示方便，我们先创建一个多层感知机，并将其初始化。

```{.python .input  n=13}
class MLP(nn.Module):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Linear(20, 256)
        self.relu = nn.ReLU()
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(self.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)
```

下面把该模型的参数存成文件，文件名为mlp.params。

```{.python .input  n=15}
torch.save(net.state_dict(),'mlp.params')
```

接下来，我们再实例化一次定义好的多层感知机。与随机初始化模型参数不同，我们在这里直接读取保存在文件里的参数。

```{.python .input  n=16}
net2 = MLP()
net2.load_state_dict(torch.load('mlp.params'))
net2.eval()
```

```{.json .output n=16}
[
 {
  "data": {
   "text/plain": "MLP(\n  (hidden): Linear(in_features=20, out_features=256, bias=True)\n  (relu): ReLU()\n  (output): Linear(in_features=256, out_features=10, bias=True)\n)"
  },
  "execution_count": 16,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

因为这两个实例都有同样的模型参数，那么对同一个输入`X`的计算结果将会是一样的。我们来验证一下。

```{.python .input  n=17}
Y2 = net2(X)
Y2 == Y
```

```{.json .output n=17}
[
 {
  "data": {
   "text/plain": "tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],\n        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.uint8)"
  },
  "execution_count": 17,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## 小结

* 通过`save`函数和`load`函数可以很方便地读写`Tensor`。
* 通过`state_dict`函数和可以很方便地获取Pytorch模型的参数。

## 练习

* 即使无须把训练好的模型部署到不同的设备，存储模型参数在实际中还有哪些好处？



## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/1255)

![](../img/qr_read-write.svg)
