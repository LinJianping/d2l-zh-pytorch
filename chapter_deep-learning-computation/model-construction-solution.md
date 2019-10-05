# 模型构造

让我们回顾一下在[“多层感知机的简洁实现”](../chapter_deep-learning-basics/mlp-gluon.md)一节中含单隐藏层的多层感知机的实现方法。我们首先构造`Sequential`实例，然后依次添加两个全连接层。其中第一层的输出大小为256，即隐藏层单元个数是256；第二层的输出大小为10，即输出层单元个数是10。我们在上一章的其他
节中也使用了`Sequential`类构造模型。这里我们介绍另外一种基于`Module`类的模型构造方法：它让模型构造更加灵活。


## 继承`Module`类来构造模型

`Module`类是`nn`模块里提供的一个模型构造类，我们可以继承它来定义我们想要的模型。下面继承`Module`类构造本节开头提到的多层感知机。这里定义的`MLP`类重载了`Mudule`类的`__init__`函数和`forward`函数。它们分别用于创建模型参数和定义前向计算。前向计算也即正向传播。

```{.python .input  n=20}
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class MLP(nn.Module):
    # 声明带有模型参数的层，这里声明了两个全连接层
    def __init__(self, **kwargs):
        # 调用MLP父类Block的构造函数来进行必要的初始化。这样在构造实例时还可以指定其他函数
        # 参数，如“模型参数的访问、初始化和共享”一节将介绍的模型参数params
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Sequential(nn.Linear(20,256), nn.ReLU())
        self.output = nn.Linear(256,10)
    # 定义模型的前向计算，即如何根据输入x计算返回所需要的模型输出
    def forward(self, x):
        h1 = self.hidden(x)
        return self.output(h1)
```

以上的`MLP`类中无须定义反向传播函数。系统将通过自动求梯度而自动生成反向传播所需的`backward`函数。

我们可以实例化`MLP`类得到模型变量`net`。下面的代码初始化`net`并传入输入数据`X`做一次前向计算。其中，`net(X)`会调用`MLP`继承自`Block`类的`__call__`函数，这个函数将调用`MLP`类定义的`forward`函数来完成前向计算。

```{.python .input  n=21}
net = MLP()
X = torch.randn((2,20))
net(X)
```

```{.json .output n=21}
[
 {
  "ename": "AttributeError",
  "evalue": "cannot assign module before Module.__init__() call",
  "output_type": "error",
  "traceback": [
   "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
   "\u001b[0;31mAttributeError\u001b[0m                            Traceback (most recent call last)",
   "\u001b[0;32m<ipython-input-21-ab6c429b6c79>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m \u001b[0mnet\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mMLP\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      2\u001b[0m \u001b[0mX\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mtorch\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mrandn\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;36m2\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;36m20\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      3\u001b[0m \u001b[0mnet\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mX\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;32m<ipython-input-20-f65c14774477>\u001b[0m in \u001b[0;36m__init__\u001b[0;34m(self, **kwargs)\u001b[0m\n\u001b[1;32m     10\u001b[0m         \u001b[0;31m# \u53c2\u6570\uff0c\u5982\u201c\u6a21\u578b\u53c2\u6570\u7684\u8bbf\u95ee\u3001\u521d\u59cb\u5316\u548c\u5171\u4eab\u201d\u4e00\u8282\u5c06\u4ecb\u7ecd\u7684\u6a21\u578b\u53c2\u6570params\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     11\u001b[0m \u001b[0;31m#         super(MLP, self).__init__(**kwargs)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 12\u001b[0;31m         \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mhidden\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mnn\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mSequential\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mnn\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mLinear\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;36m20\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;36m256\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mnn\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mReLU\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     13\u001b[0m         \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0moutput\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mnn\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mLinear\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;36m256\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;36m10\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     14\u001b[0m     \u001b[0;31m# \u5b9a\u4e49\u6a21\u578b\u7684\u524d\u5411\u8ba1\u7b97\uff0c\u5373\u5982\u4f55\u6839\u636e\u8f93\u5165x\u8ba1\u7b97\u8fd4\u56de\u6240\u9700\u8981\u7684\u6a21\u578b\u8f93\u51fa\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;32m~/miniconda3/envs/gluon/lib/python3.6/site-packages/torch/nn/modules/module.py\u001b[0m in \u001b[0;36m__setattr__\u001b[0;34m(self, name, value)\u001b[0m\n\u001b[1;32m    563\u001b[0m                 \u001b[0;32mif\u001b[0m \u001b[0mmodules\u001b[0m \u001b[0;32mis\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    564\u001b[0m                     raise AttributeError(\n\u001b[0;32m--> 565\u001b[0;31m                         \"cannot assign module before Module.__init__() call\")\n\u001b[0m\u001b[1;32m    566\u001b[0m                 \u001b[0mremove_from\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m__dict__\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_parameters\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_buffers\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    567\u001b[0m                 \u001b[0mmodules\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mname\u001b[0m\u001b[0;34m]\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mvalue\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;31mAttributeError\u001b[0m: cannot assign module before Module.__init__() call"
  ]
 }
]
```

注意，这里并没有将`Module`类命名为`Layer`（层）或者`Model`（模型）之类的名字，这是因为该类是一个可供自由组建的部件。它的子类既可以是一个层（如torch提供的`Dense`类），又可以是一个模型（如这里定义的`MLP`类），或者是模型的一个部分。我们下面通过两个例子来展示它的灵活性。

## `Sequential`类继承自`Module`类

我们刚刚提到，`Module`类是一个通用的部件。事实上，`Sequential`类继承自`Module`类。当模型的前向计算为简单串联各个层的计算时，可以通过更加简单的方式定义模型。这正是`Sequential`类的目的：它提供`add_module`函数来逐一添加串联的`Module`子类实例，而模型的前向计算就是将这些实例按添加的顺序逐一计算。

下面我们实现一个与`Sequential`类有相同功能的`MySequential`类。这或许可以帮助读者更加清晰地理解`Sequential`类的工作机制。

```{.python .input  n=6}
class MySequential(nn.Module):
    def __init__(self, **kwargs):
        super(MySequential, self).__init__(**kwargs)

    def add_module(self, block):
        # module是一个Module子类实例，假设它有一个独一无二的名字。我们将它保存在Module类的
        # 成员变量_children里，其类型是OrderedDict。
        self._modules[block] = block

    def forward(self, x):
        # OrderedDict保证会按照成员添加时的顺序遍历成员
        for module in self._modules.values():
            x = module(x)
        return x
```

我们用`MySequential`类来实现前面描述的`MLP`类，并使用随机初始化的模型做一次前向计算。

```{.python .input  n=7}
net = MySequential()
net.add_module(nn.Linear(20,256))
net.add_module(nn.ReLU())
net.add_module(nn.Linear(256,10))
x = torch.randn(2,20)
net(X)
```

```{.json .output n=7}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Linear(in_features=20, out_features=256, bias=True)\nReLU()\nLinear(in_features=256, out_features=10, bias=True)\n"
 },
 {
  "data": {
   "text/plain": "tensor([[-0.1633,  0.0289,  0.1495, -0.0458, -0.1448, -0.0628, -0.2764,  0.1655,\n         -0.1896, -0.0275],\n        [-0.0330,  0.4147,  0.2681,  0.0520,  0.0607,  0.0231,  0.0757,  0.2142,\n          0.3637,  0.2295]], grad_fn=<AddmmBackward>)"
  },
  "execution_count": 7,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

可以观察到这里`MySequential`类的使用跟[“多层感知机的简洁实现”](../chapter_deep-learning-basics/mlp-gluon.md)一节中`Sequential`类的使用没什么区别。


## 构造复杂的模型

虽然`Sequential`类可以使模型构造更加简单，且不需要定义`forward`函数，但直接继承`Module`类可以极大地拓展模型构造的灵活性。下面我们构造一个稍微复杂点的网络`FancyMLP`。在这个网络中，我们通过`Parameter`创建训练中不被迭代的参数，即常数参数。在前向计算中，除了使用创建的常数参数外，我们还使用`Tensor`的函数和Python的控制流，并多次调用相同的层。

```{.python .input  n=28}
class FancyMLP(nn.Module):
    def __init__(self, **kwargs):
        super(FancyMLP, self).__init__(**kwargs)
        # 使用get_constant创建的随机权重参数不会在训练中被迭代（即常数参数）
        self.dense1 = nn.Sequential(nn.Linear(20,20), nn.ReLU())
        self.rand_weight = nn.Parameter(torch.empty(20,20).uniform_(0,1))
        self.dense = nn.Sequential(nn.Linear(20,256), nn.ReLU())
        self.register_buffer('random_weights', self.rand_weight)

    def forward(self, x):
        x = self.dense1(x)
        x = F.relu(torch.mm(x, Variable(self.rand_weight).data)+1)
        x = self.dense(x)
        while x.norm().item() > 1:
            x /= 2
        if x.norm().item() < 0.8:
            x *= 10
        return x.sum()
```

在这个`FancyMLP`模型中，我们使用了常数权重`rand_weight`（注意它不是模型参数）、做了矩阵乘法操作（`nd.dot`）并重复使用了相同的`Dense`层。下面我们来测试该模型的随机初始化和前向计算。

```{.python .input  n=29}
net = FancyMLP()
x = torch.randn(2,20)
net(X)
```

```{.json .output n=29}
[
 {
  "data": {
   "text/plain": "tensor(11.1770, grad_fn=<SumBackward0>)"
  },
  "execution_count": 29,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

因为`FancyMLP`和`Sequential`类都是`Module`类的子类，所以我们可以嵌套调用它们。

```{.python .input  n=31}
class NestMLP(nn.Module):
    def __init__(self, **kwargs):
        super(NestMLP, self).__init__(**kwargs)
        self.net = nn.Sequential(nn.Linear(20,64),nn.ReLU(),nn.Linear(64, 32),nn.ReLU())
#         self.net = [nn.Linear(20,64),nn.ReLU(),nn.Linear(64, 32),nn.ReLU()]
        self.dense = nn.Sequential(nn.Linear(32,20),nn.ReLU())

    def forward(self, x):
        return self.dense(self.net(x))

net = nn.Sequential()
net.add_module("Linear1",nn.Linear(20,20))
net.add_module("NestNlp",NestMLP())
net.add_module("FancyMLP",FancyMLP())

net(X)
```

```{.json .output n=31}
[
 {
  "ename": "TypeError",
  "evalue": "'list' object is not callable",
  "output_type": "error",
  "traceback": [
   "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
   "\u001b[0;31mTypeError\u001b[0m                                 Traceback (most recent call last)",
   "\u001b[0;32m<ipython-input-31-43860fa6c5fe>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m     14\u001b[0m \u001b[0mnet\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0madd_module\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"FancyMLP\"\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0mFancyMLP\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     15\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 16\u001b[0;31m \u001b[0mnet\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mX\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
   "\u001b[0;32m~/miniconda3/envs/gluon/lib/python3.6/site-packages/torch/nn/modules/module.py\u001b[0m in \u001b[0;36m__call__\u001b[0;34m(self, *input, **kwargs)\u001b[0m\n\u001b[1;32m    491\u001b[0m             \u001b[0mresult\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_slow_forward\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0minput\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    492\u001b[0m         \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 493\u001b[0;31m             \u001b[0mresult\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mforward\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0minput\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    494\u001b[0m         \u001b[0;32mfor\u001b[0m \u001b[0mhook\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_forward_hooks\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mvalues\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    495\u001b[0m             \u001b[0mhook_result\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mhook\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0minput\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mresult\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;32m~/miniconda3/envs/gluon/lib/python3.6/site-packages/torch/nn/modules/container.py\u001b[0m in \u001b[0;36mforward\u001b[0;34m(self, input)\u001b[0m\n\u001b[1;32m     90\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mforward\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0minput\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     91\u001b[0m         \u001b[0;32mfor\u001b[0m \u001b[0mmodule\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_modules\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mvalues\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 92\u001b[0;31m             \u001b[0minput\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mmodule\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0minput\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     93\u001b[0m         \u001b[0;32mreturn\u001b[0m \u001b[0minput\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     94\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;32m~/miniconda3/envs/gluon/lib/python3.6/site-packages/torch/nn/modules/module.py\u001b[0m in \u001b[0;36m__call__\u001b[0;34m(self, *input, **kwargs)\u001b[0m\n\u001b[1;32m    491\u001b[0m             \u001b[0mresult\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_slow_forward\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0minput\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    492\u001b[0m         \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 493\u001b[0;31m             \u001b[0mresult\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mforward\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0minput\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    494\u001b[0m         \u001b[0;32mfor\u001b[0m \u001b[0mhook\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_forward_hooks\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mvalues\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    495\u001b[0m             \u001b[0mhook_result\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mhook\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0minput\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mresult\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;32m<ipython-input-31-43860fa6c5fe>\u001b[0m in \u001b[0;36mforward\u001b[0;34m(self, x)\u001b[0m\n\u001b[1;32m      7\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      8\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mforward\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mx\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 9\u001b[0;31m         \u001b[0;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdense\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mnet\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mx\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     10\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     11\u001b[0m \u001b[0mnet\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mnn\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mSequential\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;31mTypeError\u001b[0m: 'list' object is not callable"
  ]
 }
]
```

## 小结

* 可以通过继承`Module`类来构造模型。
* `Sequential`类继承自`Module`类。
* 虽然`Sequential`类可以使模型构造更加简单，但直接继承`Module`类可以极大地拓展模型构造的灵活性。


## 练习

* 如果不在`MLP`类的`__init__`函数里调用父类的`__init__`函数，会出现什么样的错误信息？
* 如果去掉`FancyMLP`类里面的`item`函数，会有什么问题？
* 如果将`NestMLP`类中通过`Sequential`实例定义的`self.net`改为`self.net = [nn.Dense(64, activation='relu'), nn.Dense(32, activation='relu')]`，会有什么问题？





## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/986)

![](../img/qr_model-construction.svg)

```{.python .input}

```
