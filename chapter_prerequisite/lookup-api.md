# 查阅文档

受篇幅所限，本书无法对所有用到的Pytorch函数和类一一详细介绍。读者可以查阅相关文档来做更深入的了解。

## 查找模块里的所有函数和类

当我们想知道一个模块里面提供了哪些可以调用的函数和类的时候，可以使用`dir`函数。下面我们打印`torch.randn`模块中所有的成员或属性。

```{.python .input  n=1}
import torch

print(dir(torch.randn))
```

```{.json .output n=1}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "['__call__', '__class__', '__delattr__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__name__', '__ne__', '__new__', '__qualname__', '__reduce__', '__reduce_ex__', '__repr__', '__self__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__text_signature__']\n"
 }
]
```

通常我们可以忽略掉由`__`开头和结尾的函数（Python的特别对象）或者由`_`开头的函数（一般为内部函数）。通过其余成员的名字我们大致猜测出这个模块提供了各种随机数的生成方法，包括从均匀分布采样（`uniform`）、从正态分布采样（`normal`）、从泊松分布采样（`poisson`）等。

## 查找特定函数和类的使用

想了解某个函数或者类的具体用法时，可以使用`help`函数。让我们以`torch`中的`ones_like`函数为例，查阅它的用法。

```{.python .input  n=2}
help(torch.ones_like)
```

```{.json .output n=2}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Help on built-in function ones_like:\n\nones_like(...)\n    ones_like(input, dtype=None, layout=None, device=None, requires_grad=False) -> Tensor\n    \n    Returns a tensor filled with the scalar value `1`, with the same size as\n    :attr:`input`. ``torch.ones_like(input)`` is equivalent to\n    ``torch.ones(input.size(), dtype=input.dtype, layout=input.layout, device=input.device)``.\n    \n    .. warning::\n        As of 0.4, this function does not support an :attr:`out` keyword. As an alternative,\n        the old ``torch.ones_like(input, out=output)`` is equivalent to\n        ``torch.ones(input.size(), out=output)``.\n    \n    Args:\n        input (Tensor): the size of :attr:`input` will determine size of the output tensor\n        dtype (:class:`torch.dtype`, optional): the desired data type of returned Tensor.\n            Default: if ``None``, defaults to the dtype of :attr:`input`.\n        layout (:class:`torch.layout`, optional): the desired layout of returned tensor.\n            Default: if ``None``, defaults to the layout of :attr:`input`.\n        device (:class:`torch.device`, optional): the desired device of returned tensor.\n            Default: if ``None``, defaults to the device of :attr:`input`.\n        requires_grad (bool, optional): If autograd should record operations on the\n            returned tensor. Default: ``False``.\n    \n    Example::\n    \n        >>> input = torch.empty(2, 3)\n        >>> torch.ones_like(input)\n        tensor([[ 1.,  1.,  1.],\n                [ 1.,  1.,  1.]])\n\n"
 }
]
```

从文档信息我们了解到，`ones_like`函数会创建和输入`Tensor`形状相同且元素为1的新`Tensor`。我们可以验证一下：

```{.python .input  n=5}
x = torch.tensor([[0, 0, 0], [2, 2, 2]])
y = torch.ones_like(x)
y
```

```{.json .output n=5}
[
 {
  "data": {
   "text/plain": "tensor([[1, 1, 1],\n        [1, 1, 1]])"
  },
  "execution_count": 5,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input}

```
