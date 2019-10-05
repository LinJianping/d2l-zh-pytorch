# 模型参数的访问、初始化和共享

在[“线性回归的简洁实现”](../chapter_deep-learning-basics/linear-regression-gluon.md)一节中，我们通过`init`模块来初始化模型的全部参数。我们也介绍了访问模型参数的简单方法。本节将深入讲解如何访问和初始化模型参数，以及如何在多个层之间共享同一份模型参数。

我们先定义一个与上一节中相同的含单隐藏层的多层感知机。我们依然使用默认方式初始化它的参数，并做一次前向计算。与之前不同的是，在这里我们从torch.nn中导入了`init`模块，它包含了多种模型初始化方法。

```{.python .input  n=12}
import torch
import torch.nn as nn

net = nn.Sequential()
net.add_module('Linear_1', nn.Linear(20, 256, bias = False))
net.add_module('relu', nn.ReLU())
net.add_module('Linear_2', nn.Linear(256, 10, bias = True))

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)

net.apply(init_weights)
X = torch.randn(size=(2, 20))
Y = net(X)  # 前向计算
```

## 访问模型参数

对于使用`Sequential`类构造的神经网络，我们可以通过方括号`[]`来访问网络的任一层。回忆一下上一节中提到的`Sequential`类与`Module`类的继承关系。对于`Sequential`实例中含模型参数的层，我们可以通过`Module`类的`parameters`方法来访问该层包含的所有参数。下面，访问多层感知机`net`中隐藏层的所有参数。索引0表示隐藏层为`Sequential`实例最先添加的层。

```{.python .input  n=6}
net[0].parameters, type(net[0].parameters)
net[1].parameters, type(net[1].parameters)
net[2].parameters, type(net[2].parameters)
```

```{.json .output n=6}
[
 {
  "data": {
   "text/plain": "(<bound method Module.parameters of Linear(in_features=256, out_features=10, bias=False)>,\n method)"
  },
  "execution_count": 6,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

而且可以看到，该参数的形状为(256, 20)，且数据类型为32位浮点数（`float32`）。为了访问特定参数，我们既可以通过名字来访问net的属性，也可以直接使用它的变量名。下面两种方法是等价的，但通常后者的代码可读性更好。

```{.python .input  n=7}
net.Linear_1.weight, net[0].weight
```

```{.json .output n=7}
[
 {
  "data": {
   "text/plain": "(Parameter containing:\n tensor([[-0.1184, -0.1072, -0.0410,  ...,  0.0831, -0.0951, -0.0994],\n         [-0.1227, -0.0289, -0.0253,  ...,  0.1278, -0.0175,  0.0619],\n         [-0.1444,  0.0389, -0.0702,  ...,  0.0839,  0.0869,  0.1215],\n         ...,\n         [-0.1218,  0.0759, -0.0928,  ..., -0.1371,  0.0182,  0.0183],\n         [ 0.0364, -0.1004,  0.0806,  ..., -0.0889,  0.1213,  0.0697],\n         [-0.0666,  0.0800,  0.0178,  ..., -0.0990,  0.0368,  0.0956]],\n        requires_grad=True), Parameter containing:\n tensor([[-0.1184, -0.1072, -0.0410,  ...,  0.0831, -0.0951, -0.0994],\n         [-0.1227, -0.0289, -0.0253,  ...,  0.1278, -0.0175,  0.0619],\n         [-0.1444,  0.0389, -0.0702,  ...,  0.0839,  0.0869,  0.1215],\n         ...,\n         [-0.1218,  0.0759, -0.0928,  ..., -0.1371,  0.0182,  0.0183],\n         [ 0.0364, -0.1004,  0.0806,  ..., -0.0889,  0.1213,  0.0697],\n         [-0.0666,  0.0800,  0.0178,  ..., -0.0990,  0.0368,  0.0956]],\n        requires_grad=True))"
  },
  "execution_count": 7,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

Gluon里参数类型为`Parameter`类，它包含参数和梯度的数值，可以分别通过`data`函数和`grad`函数来访问。因为我们随机初始化了权重，所以权重参数是一个由随机数组成的形状为(256, 20)的`NDArray`。

```{.python .input  n=8}
net[0].weight.data
```

```{.json .output n=8}
[
 {
  "data": {
   "text/plain": "tensor([[-0.1184, -0.1072, -0.0410,  ...,  0.0831, -0.0951, -0.0994],\n        [-0.1227, -0.0289, -0.0253,  ...,  0.1278, -0.0175,  0.0619],\n        [-0.1444,  0.0389, -0.0702,  ...,  0.0839,  0.0869,  0.1215],\n        ...,\n        [-0.1218,  0.0759, -0.0928,  ..., -0.1371,  0.0182,  0.0183],\n        [ 0.0364, -0.1004,  0.0806,  ..., -0.0889,  0.1213,  0.0697],\n        [-0.0666,  0.0800,  0.0178,  ..., -0.0990,  0.0368,  0.0956]])"
  },
  "execution_count": 8,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

权重梯度的形状和权重的形状一样。因为我们还没有进行反向传播计算，所以梯度的值全为0。

```{.python .input  n=9}
net[0].weight.grad
```

类似地，我们可以访问其他层的参数，如输出层的偏差值。

```{.python .input  n=13}
net[2].bias.data
```

```{.json .output n=13}
[
 {
  "data": {
   "text/plain": "tensor([ 0.0470, -0.0136, -0.0305, -0.0189,  0.0512,  0.0382, -0.0407, -0.0264,\n        -0.0258, -0.0539])"
  },
  "execution_count": 13,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

如上所述，访问参数可能有点乏味，尤其是当我们有更复杂的块或块的块（甚至块的块的块）时，因为我们需要以相反的顺序遍历整个树 被建造。 为了避免这种情况，块带有`state_dict`方法，该方法在一个字典中获取网络的所有参数，以便我们可以轻松遍历它。 它通过遍历块的所有组成部分来实现，并根据需要在子块上调用`state_dict`。 要查看差异，请考虑以下几点：

```{.python .input  n=15}
print(net[0].state_dict()) # parameters only for first layer
print(net.state_dict()) # parameters for entire network
```

```{.json .output n=15}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "OrderedDict([('weight', tensor([[-0.1296,  0.0031, -0.0010,  ...,  0.0870, -0.0066,  0.0901],\n        [-0.1274,  0.0452, -0.0678,  ..., -0.0841,  0.0131,  0.0623],\n        [-0.0844,  0.0204, -0.0070,  ..., -0.0820, -0.0201, -0.0992],\n        ...,\n        [-0.0007, -0.0500,  0.0397,  ..., -0.1109,  0.0186,  0.0592],\n        [ 0.1239,  0.0470,  0.1458,  ..., -0.1275, -0.0779,  0.1200],\n        [-0.0429,  0.0933,  0.0470,  ..., -0.0533,  0.0910, -0.0535]]))])\nOrderedDict([('Linear_1.weight', tensor([[-0.1296,  0.0031, -0.0010,  ...,  0.0870, -0.0066,  0.0901],\n        [-0.1274,  0.0452, -0.0678,  ..., -0.0841,  0.0131,  0.0623],\n        [-0.0844,  0.0204, -0.0070,  ..., -0.0820, -0.0201, -0.0992],\n        ...,\n        [-0.0007, -0.0500,  0.0397,  ..., -0.1109,  0.0186,  0.0592],\n        [ 0.1239,  0.0470,  0.1458,  ..., -0.1275, -0.0779,  0.1200],\n        [-0.0429,  0.0933,  0.0470,  ..., -0.0533,  0.0910, -0.0535]])), ('Linear_2.weight', tensor([[-0.1419,  0.1267,  0.1476,  ...,  0.0472, -0.1243,  0.0971],\n        [-0.0245, -0.0080,  0.0833,  ..., -0.0257, -0.0962, -0.0416],\n        [-0.0454,  0.0975,  0.0370,  ..., -0.1165, -0.1371, -0.0604],\n        ...,\n        [ 0.0048, -0.0070,  0.1242,  ...,  0.0943,  0.0631, -0.0230],\n        [ 0.0595,  0.0623,  0.0421,  ...,  0.0432, -0.0130, -0.1083],\n        [-0.0294,  0.0117, -0.0951,  ...,  0.0378,  0.1054, -0.0301]])), ('Linear_2.bias', tensor([ 0.0470, -0.0136, -0.0305, -0.0189,  0.0512,  0.0382, -0.0407, -0.0264,\n        -0.0258, -0.0539]))])\n"
 }
]
```

这为我们提供了访问网络参数的第三种方式。 如果要获取第二个线性层的权重项的值，则可以简单地使用以下代码：

```{.python .input  n=18}
net.state_dict()['Linear_1.weight']
```

```{.json .output n=18}
[
 {
  "data": {
   "text/plain": "tensor([[-0.1296,  0.0031, -0.0010,  ...,  0.0870, -0.0066,  0.0901],\n        [-0.1274,  0.0452, -0.0678,  ..., -0.0841,  0.0131,  0.0623],\n        [-0.0844,  0.0204, -0.0070,  ..., -0.0820, -0.0201, -0.0992],\n        ...,\n        [-0.0007, -0.0500,  0.0397,  ..., -0.1109,  0.0186,  0.0592],\n        [ 0.1239,  0.0470,  0.1458,  ..., -0.1275, -0.0779,  0.1200],\n        [-0.0429,  0.0933,  0.0470,  ..., -0.0533,  0.0910, -0.0535]])"
  },
  "execution_count": 18,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

让我们看看如果在彼此之间嵌套多个Module，参数命名约定将如何工作。 为此，我们首先定义一个生成Module的函数，然后将它们组合在更大的Module中。

```{.python .input  n=25}
def block1():
    net = nn.Sequential(nn.Linear(16,32),
                       nn.ReLU(),
                       nn.Linear(32,16),
                       nn.ReLU())
    return net

def block2():
    net = nn.Sequential()
    for i in range(4):
        net.add_module('block'+str(i), block1())
    return net

rgnet = nn.Sequential()
rgnet.add_module('module', block2())
rgnet.add_module('Last_linear_layer', nn.Linear(16,10))
rgnet.apply(init_weights)
x = torch.randn(2,16)
rgnet(x) # forward computation
```

```{.json .output n=25}
[
 {
  "data": {
   "text/plain": "tensor([[ 0.1384,  0.0483,  0.1123, -0.1618, -0.1237,  0.1700, -0.0660, -0.1770,\n         -0.3983,  0.1206],\n        [ 0.1355,  0.0564,  0.1284, -0.1421, -0.1388,  0.1781, -0.0687, -0.1914,\n         -0.4015,  0.1119]], grad_fn=<AddmmBackward>)"
  },
  "execution_count": 25,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

现在我们已经完成了网络的设计，让我们看看它是如何组织的。 __state_dict__在命名和逻辑结构方面都为我们提供了此信息。

```{.python .input  n=26}
print(rgnet.parameters)
for param in rgnet.parameters():
    print(param.size(), param.dtype) 
```

```{.json .output n=26}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "<bound method Module.parameters of Sequential(\n  (module): Sequential(\n    (block0): Sequential(\n      (0): Linear(in_features=16, out_features=32, bias=True)\n      (1): ReLU()\n      (2): Linear(in_features=32, out_features=16, bias=True)\n      (3): ReLU()\n    )\n    (block1): Sequential(\n      (0): Linear(in_features=16, out_features=32, bias=True)\n      (1): ReLU()\n      (2): Linear(in_features=32, out_features=16, bias=True)\n      (3): ReLU()\n    )\n    (block2): Sequential(\n      (0): Linear(in_features=16, out_features=32, bias=True)\n      (1): ReLU()\n      (2): Linear(in_features=32, out_features=16, bias=True)\n      (3): ReLU()\n    )\n    (block3): Sequential(\n      (0): Linear(in_features=16, out_features=32, bias=True)\n      (1): ReLU()\n      (2): Linear(in_features=32, out_features=16, bias=True)\n      (3): ReLU()\n    )\n  )\n  (Last_linear_layer): Linear(in_features=16, out_features=10, bias=True)\n)>\ntorch.Size([32, 16]) torch.float32\ntorch.Size([32]) torch.float32\ntorch.Size([16, 32]) torch.float32\ntorch.Size([16]) torch.float32\ntorch.Size([32, 16]) torch.float32\ntorch.Size([32]) torch.float32\ntorch.Size([16, 32]) torch.float32\ntorch.Size([16]) torch.float32\ntorch.Size([32, 16]) torch.float32\ntorch.Size([32]) torch.float32\ntorch.Size([16, 32]) torch.float32\ntorch.Size([16]) torch.float32\ntorch.Size([32, 16]) torch.float32\ntorch.Size([32]) torch.float32\ntorch.Size([16, 32]) torch.float32\ntorch.Size([16]) torch.float32\ntorch.Size([10, 16]) torch.float32\ntorch.Size([10]) torch.float32\n"
 }
]
```

## 初始化模型参数

我们在[“数值稳定性和模型初始化”](../chapter_deep-learning-basics/numerical-stability-and-init.md)一节中描述了模型的默认初始化方法：权重参数元素为[-0.07, 0.07]之间均匀分布的随机数，偏差参数则全为0。但我们经常需要使用其他方法来初始化权重。torch.nn的`init`模块里提供了多种预设的初始化方法。在下面的例子中，我们将权重参数初始化成均值为0、标准差为0.01的正态分布随机数，并依然将偏差参数清零。

```{.python .input  n=19}
linear1 = nn.Linear(2,5,bias=True)
torch.nn.init.normal_(linear1.weight, mean=0, std =0.01)  
linear1.weight.data
```

```{.json .output n=19}
[
 {
  "data": {
   "text/plain": "tensor([[-5.7424e-03, -9.4993e-04],\n        [ 1.1245e-03, -6.7954e-03],\n        [-7.7106e-03, -8.0805e-05],\n        [-3.1244e-03,  1.6121e-02],\n        [-1.1150e-02, -3.3599e-03]])"
  },
  "execution_count": 19,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

下面使用常数来初始化权重参数。

```{.python .input  n=21}
linear1 = nn.Linear(2,5,bias=True)
torch.nn.init.constant_(linear1.weight, 1)  
linear1.weight.data
```

```{.json .output n=21}
[
 {
  "data": {
   "text/plain": "tensor([[1., 1.],\n        [1., 1.],\n        [1., 1.],\n        [1., 1.],\n        [1., 1.]])"
  },
  "execution_count": 21,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

如果只想对某个特定参数进行初始化，我们可以调用`Parameter`类的`initialize`函数，它与`Block`类提供的`initialize`函数的使用方法一致。下例中我们对隐藏层的权重使用Xavier随机初始化方法。

```{.python .input  n=28}
torch.nn.init.xavier_normal_(net[0].weight)
net[0].weight.data
```

```{.json .output n=28}
[
 {
  "data": {
   "text/plain": "tensor([[-0.0050, -0.0052, -0.0518,  ..., -0.0945, -0.1353, -0.0203],\n        [ 0.1424,  0.1792, -0.0081,  ..., -0.0110, -0.0454,  0.0520],\n        [-0.0417, -0.0415,  0.0477,  ..., -0.1626, -0.0690, -0.0227],\n        ...,\n        [-0.0540, -0.0346,  0.0004,  ...,  0.1131,  0.0955, -0.0147],\n        [ 0.0893, -0.0906,  0.0538,  ...,  0.0544, -0.0880,  0.0299],\n        [-0.0853,  0.0810,  0.0719,  ...,  0.0110,  0.0087, -0.0144]])"
  },
  "execution_count": 28,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

如果我们只想以不同的方式初始化一个特定的参数，我们可以简单地为该问题只为适当的子块（或参数）设置初始化器。 例如，下面我们将第二层初始化为常数42，并使用Xavier初始化器作为第一层的权重。

```{.python .input  n=29}
block1 = nn.Sequential()
block1.add_module('Linear_1', nn.Linear(2,5,bias=False))
block2 = nn.Sequential()
block2.add_module('Linear_2', nn.Linear(5,5,bias=False))

model = nn.Sequential()
model.add_module('first', block1)
model.add_module('second', block2)

def xavier_normal(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
def init_42(m):
    if type(m) == nn.Linear:
        torch.nn.init.constant_(m.weight, 42)
              
block1.apply(xavier_normal)
block2.apply(init_42)
print(model.state_dict())
```

```{.json .output n=29}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "OrderedDict([('first.Linear_1.weight', tensor([[ 0.3031,  0.3825],\n        [-0.2977, -0.2843],\n        [-0.0184,  0.6712],\n        [ 0.0638, -0.2762],\n        [ 0.6540,  0.6246]])), ('second.Linear_2.weight', tensor([[42., 42., 42., 42., 42.],\n        [42., 42., 42., 42., 42.],\n        [42., 42., 42., 42., 42.],\n        [42., 42., 42., 42., 42.],\n        [42., 42., 42., 42., 42.]]))])\n"
 }
]
```

## 自定义初始化方法

有时，init模块中没有提供我们需要的初始化方法。 此时，我们可以通过编写所需的函数并将其用于初始化权重来实现所需的实现。在下面的例子里，我们令权重有一半概率初始化为0，有另一半概率初始化为$[-10,-5]$和$[5,10]$两个区间里均匀分布的随机数。

```{.python .input  n=35}
def custom(m):
    torch.nn.init.uniform_(m[0].weight, -10,10)
    for i in range(m[0].weight.data.shape[0]):
        for j in range(m[0].weight.data.shape[1]):
            if m[0].weight.data[i][j]<=5 and m[0].weight.data[i][j]>=-5:
                m[0].weight.data[i][j]=0

m = nn.Sequential(nn.Linear(5,5,bias=False))
custom(m)
net[0].weight.data
print(m.state_dict())
```

```{.json .output n=35}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "OrderedDict([('0.weight', tensor([[ 0.0000,  0.0000,  0.0000,  7.3188,  5.7806],\n        [ 6.7462,  0.0000,  9.3246,  0.0000,  5.6143],\n        [ 0.0000,  8.0978,  0.0000,  7.6121,  0.0000],\n        [ 0.0000,  0.0000, -5.3181, -8.6584,  0.0000],\n        [ 0.0000, -7.0897, -7.9511,  0.0000,  0.0000]]))])\n"
 }
]
```

如果此功能还不够，我们可以直接设置参数。 由于__.data__返回张量，因此我们可以像访问任何其他矩阵一样访问它。

```{.python .input  n=36}
m[0].weight.data +=1
m[0].weight.data[0][0] = 42
m[0].weight.data
```

```{.json .output n=36}
[
 {
  "data": {
   "text/plain": "tensor([[42.0000,  1.0000,  1.0000,  8.3188,  6.7806],\n        [ 7.7462,  1.0000, 10.3246,  1.0000,  6.6143],\n        [ 1.0000,  9.0978,  1.0000,  8.6121,  1.0000],\n        [ 1.0000,  1.0000, -4.3181, -7.6584,  1.0000],\n        [ 1.0000, -6.0897, -6.9511,  1.0000,  1.0000]])"
  },
  "execution_count": 36,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## 共享模型参数

在有些情况下，我们希望在多个层之间共享模型参数。[“模型构造”](model-construction.md)一节介绍了如何在`Module`类的`forward`函数里多次调用同一个层来计算。这里再介绍另外一种方法，它在构造层的时候指定使用特定的参数。如果不同层使用同一份参数，那么它们在前向计算和反向传播时都会共享相同的参数。在下面的例子里，我们让模型的第二隐藏层（`shared`变量）和第三隐藏层共享模型参数。

```{.python .input  n=37}
shared = nn.Sequential()
shared.add_module('linear_shared', nn.Linear(8,8,bias=False))
shared.add_module('relu_shared', nn.ReLU())                  
net = nn.Sequential(nn.Linear(20,8,bias=False),
               nn.ReLU(),
               shared,
               shared,
               nn.Linear(8,10,bias=False))

net.apply(init_weights)

print(net[2][0].weight==net[3][0].weight)
```

```{.json .output n=37}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "tensor([[1, 1, 1, 1, 1, 1, 1, 1],\n        [1, 1, 1, 1, 1, 1, 1, 1],\n        [1, 1, 1, 1, 1, 1, 1, 1],\n        [1, 1, 1, 1, 1, 1, 1, 1],\n        [1, 1, 1, 1, 1, 1, 1, 1],\n        [1, 1, 1, 1, 1, 1, 1, 1],\n        [1, 1, 1, 1, 1, 1, 1, 1],\n        [1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.uint8)\n"
 }
]
```

上面的示例显示第二层和第三层的参数是绑定的。 它们是相同的，而不仅仅是相等的。 也就是说，通过更改参数之一，其他参数也将更改。


## 小结

* 有多种方法来访问、初始化和共享模型参数。
* 可以自定义初始化方法。


## 练习

* 查阅有关`init`模块的MXNet文档，了解不同的参数初始化方法。
* 尝试在`net.initialize()`后、`net(X)`前访问模型参数，观察模型参数的形状。
* 构造一个含共享参数层的多层感知机并训练。在训练过程中，观察每一层的模型参数和梯度。



## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/987)

![](../img/qr_parameters.svg)

```{.python .input}

```
