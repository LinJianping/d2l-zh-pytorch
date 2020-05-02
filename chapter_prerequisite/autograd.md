# 自动求梯度

在深度学习中，我们经常需要对函数求梯度（gradient）。本节将介绍如何使用Pytorch提供的`autograd`模块来自动求梯度。如果对本节中的数学概念（如梯度）不是很熟悉，可以参阅附录中[“数学基础”](../chapter_appendix/math.md)一节。

Pytorch中是通过`Variable`来定义需要自动求导的变量。

```{.python .input  n=5}
import torch
from torch.autograd import Variable
```

## 简单例子

我们先看一个简单例子：对函数 $y = 2\boldsymbol{x}^{\top}\boldsymbol{x}$ 求关于列向量 $\boldsymbol{x}$ 的梯度。我们先创建变量`x`，并赋初值。为了求有关变量`x`的梯度，我们需要设置`requires_grad`为$True$。

```{.python .input  n=8}
x = Variable(torch.arange(4, dtype=torch.float32).reshape((4, 1)),requires_grad=True)
x
```

```{.json .output n=8}
[
 {
  "data": {
   "text/plain": "tensor([[0.],\n        [1.],\n        [2.],\n        [3.]], requires_grad=True)"
  },
  "execution_count": 8,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=10}
y = 2 * torch.mm(x.t(), x)
```

由于`x`的形状为（4, 1），`y`是一个标量。接下来我们可以通过调用`backward`函数自动求梯度。需要注意的是，如果`y`不是一个标量，Pytorch将默认先对`y`中元素求和得到新的变量，再求该变量有关`x`的梯度。

```{.python .input  n=11}
y.backward()
```

函数 $y = 2\boldsymbol{x}^{\top}\boldsymbol{x}$ 关于$\boldsymbol{x}$ 的梯度应为$4\boldsymbol{x}$。现在我们来验证一下求出来的梯度是正确的。

```{.python .input  n=12}
assert (x.grad - 4 * x).norm().item() == 0
x.grad
```

```{.json .output n=12}
[
 {
  "data": {
   "text/plain": "tensor([[ 0.],\n        [ 4.],\n        [ 8.],\n        [12.]])"
  },
  "execution_count": 12,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## 训练模式和预测模式

在有些情况下，同一个模型在训练模式和预测模式下的行为并不相同。我们会在后面的章节详细介绍这些区别。


## 对Python控制流求梯度

使用Pytorch的一个便利之处是，即使函数的计算图包含了Python的控制流（如条件和循环控制），我们也有可能对变量求梯度。

考虑下面程序，其中包含Python的条件和循环控制。需要强调的是，这里循环（while循环）迭代的次数和条件判断（if语句）的执行都取决于输入`a`的值。

```{.python .input  n=13}
def f(a):
    b = a * 2
    while b.norm().item() < 1000:
        b = b * 2
    if b.sum().item() > 0:
        c = b
    else:
        c = 100 * b
    return c
```

我们像之前一样调用`backward`函数求梯度。

```{.python .input  n=15}
a = torch.randn(size=(1,))
a.requires_grad = True
c = f(a)
c.backward()
```

我们来分析一下上面定义的`f`函数。事实上，给定任意输入`a`，其输出必然是 `f(a) = x * a`的形式，其中标量系数`x`的值取决于输入`a`。由于`c = f(a)`有关`a`的梯度为`x`，且值为`c / a`，我们可以像下面这样验证对本例中控制流求梯度的结果的正确性。

```{.python .input  n=16}
a.grad == c / a
```

```{.json .output n=16}
[
 {
  "data": {
   "text/plain": "tensor([1], dtype=torch.uint8)"
  },
  "execution_count": 16,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## 小结

* Pytorch提供`autograd`模块来自动化求导过程。
* Pytorch的`autograd`模块可以对一般的命令式程序进行求导。
* Pytorch的运行模式包括训练模式和预测模式。

## 练习

* 在本节对控制流求梯度的例子中，把变量`a`改成一个随机向量或矩阵。此时计算结果`c`不再是标量，运行结果将有何变化？该如何分析该结果？
* 重新设计一个对控制流求梯度的例子。运行并分析结果。




## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/744)

![](../img/qr_autograd.svg)

```{.python .input}

```
