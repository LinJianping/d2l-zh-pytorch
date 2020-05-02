# 数据操作

在深度学习中，我们通常会频繁地对数据进行操作。作为动手学深度学习的基础，本节将介绍如何对内存中的数据进行操作。

我们将从介绍`Tensor`(PyTorch的主要工具，用于存储和转换数据).开始。 如果您以前使用过`NumPy`，您会发现张量在设计上类似于`NumPy`的多维数组。 张量支持CPU，GPU上的异步计算，并支持自动区分。

## 创建`Tensor`

我们先介绍`Tensor`的最基本功能。如果对这里用到的数学操作不是很熟悉，可以参阅附录中[“数学基础”](../chapter_appendix/math.md)一节。

首先导入`torch`模块。

```{.python .input  n=15}
import torch
```

然后我们用`arange`函数创建一个行向量。

```{.python .input  n=25}
x = torch.arange(12, dtype=torch.float32)
x
```

```{.json .output n=25}
[
 {
  "data": {
   "text/plain": "tensor([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11.])"
  },
  "execution_count": 25,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

这时返回了一个`Tensor`实例，其中包含了从0开始的12个连续整数。

我们可以通过`shape`属性来获取`Tensor`实例的形状。

```{.python .input  n=26}
x.shape
```

```{.json .output n=26}
[
 {
  "data": {
   "text/plain": "torch.Size([12])"
  },
  "execution_count": 26,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

我们也能够通过`size`方法得到`Tensor`实例中元素（element）的总数。

```{.python .input  n=27}
x.size()
```

```{.json .output n=27}
[
 {
  "data": {
   "text/plain": "torch.Size([12])"
  },
  "execution_count": 27,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

下面使用`reshape`函数把行向量`x`的形状改为(3, 4)，也就是一个3行4列的矩阵，并记作`X`。除了形状改变之外，`X`中的元素保持不变。

```{.python .input  n=28}
X = x.reshape((3, 4))
X
```

```{.json .output n=28}
[
 {
  "data": {
   "text/plain": "tensor([[ 0.,  1.,  2.,  3.],\n        [ 4.,  5.,  6.,  7.],\n        [ 8.,  9., 10., 11.]])"
  },
  "execution_count": 28,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

注意`X`属性中的形状发生了变化。上面`x.reshape((3, 4))`也可写成`x.reshape((-1, 4))`或`x.reshape((3, -1))`。由于`x`的元素个数是已知的，这里的`-1`是能够通过元素个数和其他维度的大小推断出来的。

接下来，我们创建一个各元素为0，形状为(2, 3, 4)的张量。实际上，之前创建的向量和矩阵都是特殊的张量。

```{.python .input  n=29}
torch.zeros((2, 3, 4))
```

```{.json .output n=29}
[
 {
  "data": {
   "text/plain": "tensor([[[0., 0., 0., 0.],\n         [0., 0., 0., 0.],\n         [0., 0., 0., 0.]],\n\n        [[0., 0., 0., 0.],\n         [0., 0., 0., 0.],\n         [0., 0., 0., 0.]]])"
  },
  "execution_count": 29,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

类似地，我们可以创建各元素为1的张量。

```{.python .input  n=30}
torch.ones((3, 4))
```

```{.json .output n=30}
[
 {
  "data": {
   "text/plain": "tensor([[1., 1., 1., 1.],\n        [1., 1., 1., 1.],\n        [1., 1., 1., 1.]])"
  },
  "execution_count": 30,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

我们也可以通过Python的列表（list）指定需要创建的`Tensor`中每个元素的值。

```{.python .input  n=34}
Y = torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]], dtype=torch.float32)
Y
```

```{.json .output n=34}
[
 {
  "data": {
   "text/plain": "tensor([[2., 1., 4., 3.],\n        [1., 2., 3., 4.],\n        [4., 3., 2., 1.]])"
  },
  "execution_count": 34,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

有些情况下，我们需要随机生成`Tensor`中每个元素的值。下面我们创建一个形状为(3, 4)的`Tensor`。它的每个元素都随机采样于均值为0、标准差为1的正态分布。

```{.python .input  n=35}
torch.randn(3,4)
```

```{.json .output n=35}
[
 {
  "data": {
   "text/plain": "tensor([[ 1.3718,  0.7011,  0.2015, -0.4932],\n        [ 0.8842,  0.2554,  0.7588,  0.1108],\n        [ 0.2218, -1.8006, -1.5810, -1.0849]])"
  },
  "execution_count": 35,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## 运算

`Tensor`支持大量的运算符（operator）。例如，我们可以对之前创建的两个形状为(3, 4)的`Tensor`做按元素加法。所得结果形状不变。

```{.python .input  n=36}
X + Y
```

```{.json .output n=36}
[
 {
  "data": {
   "text/plain": "tensor([[ 2.,  2.,  6.,  6.],\n        [ 5.,  7.,  9., 11.],\n        [12., 12., 12., 12.]])"
  },
  "execution_count": 36,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

按元素乘法：

```{.python .input  n=37}
X * Y
```

```{.json .output n=37}
[
 {
  "data": {
   "text/plain": "tensor([[ 0.,  1.,  8.,  9.],\n        [ 4., 10., 18., 28.],\n        [32., 27., 20., 11.]])"
  },
  "execution_count": 37,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

按元素除法：

```{.python .input  n=38}
X / Y
```

```{.json .output n=38}
[
 {
  "data": {
   "text/plain": "tensor([[ 0.0000,  1.0000,  0.5000,  1.0000],\n        [ 4.0000,  2.5000,  2.0000,  1.7500],\n        [ 2.0000,  3.0000,  5.0000, 11.0000]])"
  },
  "execution_count": 38,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

按元素做指数运算：

```{.python .input  n=39}
Y.exp()
```

```{.json .output n=39}
[
 {
  "data": {
   "text/plain": "tensor([[ 7.3891,  2.7183, 54.5981, 20.0855],\n        [ 2.7183,  7.3891, 20.0855, 54.5981],\n        [54.5981, 20.0855,  7.3891,  2.7183]])"
  },
  "execution_count": 39,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

除了按元素计算外，我们还可以使用`mm`函数做矩阵乘法。下面将`X`与`Y`的转置做矩阵乘法。由于`X`是3行4列的矩阵，`Y`转置为4行3列的矩阵，因此两个矩阵相乘得到3行3列的矩阵。

```{.python .input  n=41}
torch.mm(X, Y.t())
```

```{.json .output n=41}
[
 {
  "data": {
   "text/plain": "tensor([[ 18.,  20.,  10.],\n        [ 58.,  60.,  50.],\n        [ 98., 100.,  90.]])"
  },
  "execution_count": 41,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

我们也可以将多个`Tensor`连结（concatenate）。下面分别在行上（维度0，即形状中的最左边元素）和列上（维度1，即形状中左起第二个元素）连结两个矩阵。可以看到，输出的第一个`Tensor`在维度0的长度（$6$）为两个输入矩阵在维度0的长度之和（$3+3$），而输出的第二个`Tensor`在维度1的长度（$8$）为两个输入矩阵在维度1的长度之和（$4+4$）。

```{.python .input  n=43}
torch.cat((X, Y), dim=0)
```

```{.json .output n=43}
[
 {
  "data": {
   "text/plain": "tensor([[ 0.,  1.,  2.,  3.],\n        [ 4.,  5.,  6.,  7.],\n        [ 8.,  9., 10., 11.],\n        [ 2.,  1.,  4.,  3.],\n        [ 1.,  2.,  3.,  4.],\n        [ 4.,  3.,  2.,  1.]])"
  },
  "execution_count": 43,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=45}
torch.cat((X, Y), dim=1)
```

```{.json .output n=45}
[
 {
  "data": {
   "text/plain": "tensor([[ 0.,  1.,  2.,  3.,  2.,  1.,  4.,  3.],\n        [ 4.,  5.,  6.,  7.,  1.,  2.,  3.,  4.],\n        [ 8.,  9., 10., 11.,  4.,  3.,  2.,  1.]])"
  },
  "execution_count": 45,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

使用条件判断式可以得到元素为0或1的新的`Tensor`。以`X == Y`为例，如果`X`和`Y`在相同位置的条件判断为真（值相等），那么新的`Tensor`在相同位置的值为1；反之为0。

```{.python .input  n=46}
X == Y
```

```{.json .output n=46}
[
 {
  "data": {
   "text/plain": "tensor([[0, 1, 0, 1],\n        [0, 0, 0, 0],\n        [0, 0, 0, 0]], dtype=torch.uint8)"
  },
  "execution_count": 46,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

对`Tensor`中的所有元素求和得到只有一个元素的`Tensor`。

```{.python .input  n=47}
X.sum()
```

```{.json .output n=47}
[
 {
  "data": {
   "text/plain": "tensor(66.)"
  },
  "execution_count": 47,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

我们可以通过`item`函数将结果变换为Python中的标量。下面例子中`X`的$L_2$范数结果同上例一样是单元素`Tensor`，但最后结果变换成了Python中的标量。

```{.python .input  n=52}
X.norm().item()
```

```{.json .output n=52}
[
 {
  "data": {
   "text/plain": "22.494443893432617"
  },
  "execution_count": 52,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

我们也可以把`Y.exp()`、`X.sum()`、`X.norm()`等分别改写为`torch.exp(Y)`、`torch.sum(X)`、`torch.norm(X)`等。

## 广播机制

前面我们看到如何对两个形状相同的`Tensor`做按元素运算。当对两个形状不同的`Tensor`按元素运算时，可能会触发广播（broadcasting）机制：先适当复制元素使这两个`Tensor`形状相同后再按元素运算。

定义两个`Tensor`：

```{.python .input  n=53}
A = torch.arange(3).reshape((3, 1))
B = torch.arange(2).reshape((1, 2))
A, B
```

```{.json .output n=53}
[
 {
  "data": {
   "text/plain": "(tensor([[0],\n         [1],\n         [2]]), tensor([[0, 1]]))"
  },
  "execution_count": 53,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

由于`A`和`B`分别是3行1列和1行2列的矩阵，如果要计算`A + B`，那么`A`中第一列的3个元素被广播（复制）到了第二列，而`B`中第一行的2个元素被广播（复制）到了第二行和第三行。如此，就可以对2个3行2列的矩阵按元素相加。

```{.python .input  n=54}
A + B
```

```{.json .output n=54}
[
 {
  "data": {
   "text/plain": "tensor([[0, 1],\n        [1, 2],\n        [2, 3]])"
  },
  "execution_count": 54,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## 索引

在`Tensor`中，索引（index）代表了元素的位置。`Tensor`的索引从0开始逐一递增。例如，一个3行2列的矩阵的行索引分别为0、1和2，列索引分别为0和1。

在下面的例子中，我们指定了`Tensor`的行索引截取范围`[1:3]`。依据左闭右开指定范围的惯例，它截取了矩阵`X`中行索引为1和2的两行。

```{.python .input  n=55}
X[1:3]
```

```{.json .output n=55}
[
 {
  "data": {
   "text/plain": "tensor([[ 4.,  5.,  6.,  7.],\n        [ 8.,  9., 10., 11.]])"
  },
  "execution_count": 55,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

我们可以指定`Tensor`中需要访问的单个元素的位置，如矩阵中行和列的索引，并为该元素重新赋值。

```{.python .input  n=56}
X[1, 2] = 9
X
```

```{.json .output n=56}
[
 {
  "data": {
   "text/plain": "tensor([[ 0.,  1.,  2.,  3.],\n        [ 4.,  5.,  9.,  7.],\n        [ 8.,  9., 10., 11.]])"
  },
  "execution_count": 56,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

当然，我们也可以截取一部分元素，并为它们重新赋值。在下面的例子中，我们为行索引为1的每一列元素重新赋值。

```{.python .input  n=57}
X[1:2, :] = 12
X
```

```{.json .output n=57}
[
 {
  "data": {
   "text/plain": "tensor([[ 0.,  1.,  2.,  3.],\n        [12., 12., 12., 12.],\n        [ 8.,  9., 10., 11.]])"
  },
  "execution_count": 57,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## 运算的内存开销

在前面的例子里我们对每个操作新开内存来存储运算结果。举个例子，即使像`Y = X + Y`这样的运算，我们也会新开内存，然后将`Y`指向新内存。为了演示这一点，我们可以使用Python自带的`id`函数：如果两个实例的ID一致，那么它们所对应的内存地址相同；反之则不同。

```{.python .input  n=58}
before = id(Y)
Y = Y + X
id(Y) == before
```

```{.json .output n=58}
[
 {
  "data": {
   "text/plain": "False"
  },
  "execution_count": 58,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

如果想指定结果到特定内存，我们可以使用前面介绍的索引来进行替换操作。在下面的例子中，我们先通过`zeros_like`创建和`Y`形状相同且元素为0的`Tensor`，记为`Z`。接下来，我们把`X + Y`的结果通过`[:]`写进`Z`对应的内存中。

```{.python .input  n=60}
Z = torch.zeros_like(Y)
before = id(Z)
Z[:] = X + Y
id(Z) == before
```

```{.json .output n=60}
[
 {
  "data": {
   "text/plain": "True"
  },
  "execution_count": 60,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

实际上，上例中我们还是为`X + Y`开了临时内存来存储计算结果，再复制到`Z`对应的内存。如果想避免这个临时内存开销，我们可以使用运算符全名函数中的`out`参数。

```{.python .input  n=61}
torch.add(X, Y, out=Z)
id(Z) == before
```

```{.json .output n=61}
[
 {
  "data": {
   "text/plain": "True"
  },
  "execution_count": 61,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

如果`X`的值在之后的程序中不会复用，我们也可以用 `X[:] = X + Y` 或者 `X += Y` 来减少运算的内存开销。

```{.python .input  n=62}
before = id(X)
X += Y
id(X) == before
```

```{.json .output n=62}
[
 {
  "data": {
   "text/plain": "True"
  },
  "execution_count": 62,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## `Tensor`和NumPy相互变换

我们可以通过`tensor`函数和`numpy`函数令数据在`Tensor`和NumPy格式之间相互变换。下面将NumPy实例变换成`Tensor`实例。

```{.python .input  n=63}
import numpy as np

P = np.ones((2, 3))
D = torch.tensor(P)
D
```

```{.json .output n=63}
[
 {
  "data": {
   "text/plain": "tensor([[1., 1., 1.],\n        [1., 1., 1.]], dtype=torch.float64)"
  },
  "execution_count": 63,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

再将`Tensor`实例变换成NumPy实例。

```{.python .input  n=64}
D.numpy()
```

```{.json .output n=64}
[
 {
  "data": {
   "text/plain": "array([[1., 1., 1.],\n       [1., 1., 1.]])"
  },
  "execution_count": 64,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## 小结

* `Tensor`是Pytorch中存储和变换数据的主要工具。
* 可以轻松地对`Tensor`创建、运算、指定索引，并与NumPy之间相互变换。


## 练习

* 运行本节中的代码。将本节中条件判断式`X == Y`改为`X < Y`或`X > Y`，看看能够得到什么样的`Tensor`。
* 将广播机制中按元素运算的两个`Tensor`替换成其他形状，结果是否和预期一样？




## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/745)

![](../img/qr_ndarray.svg)
