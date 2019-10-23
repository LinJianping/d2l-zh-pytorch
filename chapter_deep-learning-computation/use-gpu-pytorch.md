# GPU计算

到目前为止，我们一直在使用CPU计算。对复杂的神经网络和大规模的数据来说，使用CPU来计算可能不够高效。在本节中，我们将介绍如何使用单块NVIDIA GPU来计算。首先，需要确保已经安装好了至少一块NVIDIA GPU。然后，下载CUDA并按照提示设置好相应的路径（可参考附录中[“使用AWS运行代码”](../chapter_appendix/aws.md)一节）。这些准备工作都完成后，下面就可以通过`nvidia-smi`命令来查看显卡信息了。

```{.python .input  n=1}
!nvidia-smi  # 对Linux/macOS用户有效
```

接下来，我们需要确认安装了MXNet的GPU版本。安装方法见[“获取和运行本书的代码”](../chapter_prerequisite/install.md)一节。运行本节中的程序需要至少2块GPU。

## 计算设备

Pytorch可以指定用来存储和计算的设备，如使用内存的CPU或者使用显存的GPU。默认情况下，Pytorch会将数据创建在内存，然后利用CPU来计算。在Pytorch中，`torch.device('cpu')`（或者在括号里填任意整数）表示所有的物理CPU和内存。这意味着，MXNet的计算会尽量使用所有的CPU核。但`torch.device(cuda)`只代表一块GPU和相应的显存。如果有多块GPU，我们用`torch.device(cuda:i)`来表示第$i$块GPU及相应的显存（$i$从0开始）且`torch.device(cuda:0)`和`torch.device(cuda)`等价。

```{.python .input  n=35}
import torch
import torch.nn as nn

torch.device('cpu'), torch.cuda.device('cuda'), torch.cuda.device('cuda:0')
```

```{.json .output n=35}
[
 {
  "data": {
   "text/plain": "(device(type='cpu'),\n <torch.cuda.device at 0x7fcde66e4b38>,\n <torch.cuda.device at 0x7fcde66e4ba8>)"
  },
  "execution_count": 35,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## `Tensor`的GPU计算

在默认情况下，`Tensor`存在内存上。

```{.python .input  n=36}
x = torch.FloatTensor([1, 2, 3])
x
```

```{.json .output n=36}
[
 {
  "data": {
   "text/plain": "tensor([1., 2., 3.])"
  },
  "execution_count": 36,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

我们可以通过`Tensor`的`device`属性来查看该`Tensor`所在的设备。

```{.python .input  n=37}
x.device
```

```{.json .output n=37}
[
 {
  "data": {
   "text/plain": "device(type='cpu')"
  },
  "execution_count": 37,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=38}
x.cpu()
```

```{.json .output n=38}
[
 {
  "data": {
   "text/plain": "tensor([1., 2., 3.])"
  },
  "execution_count": 38,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

### GPU上的存储

我们有多种方法将`Tensor`存储在显存上。例如，我们可以在创建`Tensor`的时候通过`device`参数指定存储设备。下面我们将`Tensor`变量`a`创建在`gpu(0)`上。注意，在打印`a`时，设备信息变成了`@gpu(0)`。创建在显存上的`Tensor`只消耗同一块显卡的显存。我们可以通过`nvidia-smi`命令查看显存的使用情况。通常，我们需要确保不创建超过显存上限的数据。

假设至少有2块GPU，下面代码将会在`gpu(1)`上创建随机数组。

```{.python .input  n=39}
B = torch.randn(size=(2, 3), device=torch.device('cuda:1'))
B
```

```{.json .output n=39}
[
 {
  "ename": "RuntimeError",
  "evalue": "CUDA error: invalid device ordinal",
  "output_type": "error",
  "traceback": [
   "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
   "\u001b[0;31mRuntimeError\u001b[0m                              Traceback (most recent call last)",
   "\u001b[0;32m<ipython-input-39-1d0f0b7a4449>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m \u001b[0mB\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mtorch\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mrandn\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0msize\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;36m2\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;36m3\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mdevice\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mtorch\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdevice\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m'cuda:1'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      2\u001b[0m \u001b[0mB\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;31mRuntimeError\u001b[0m: CUDA error: invalid device ordinal"
  ]
 }
]
```

除了在创建时指定，我们也可以通过`cuda`函数和`cpu`函数在设备之间传输数据。下面我们将内存上的`Tensor`变量`x`复制到`gpu(0)`上。

```{.python .input  n=40}
y = x.cuda()
y
```

```{.json .output n=40}
[
 {
  "data": {
   "text/plain": "tensor([1., 2., 3.], device='cuda:0')"
  },
  "execution_count": 40,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

需要区分的是，如果源变量和目标变量的`device`一致，`cuda`函数使目标变量和源变量共享源变量的内存或显存。

```{.python .input  n=41}
y.cuda() is y
```

```{.json .output n=41}
[
 {
  "data": {
   "text/plain": "True"
  },
  "execution_count": 41,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

而`copy_`是原位操作，所以没有执行复制。

```{.python .input  n=42}
y.copy_(y) is y
```

```{.json .output n=42}
[
 {
  "data": {
   "text/plain": "True"
  },
  "execution_count": 42,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

### GPU上的计算

MXNet的计算会在数据的`device`属性所指定的设备上执行。为了使用GPU计算，我们只需要事先将数据存储在显存上。计算结果会自动保存在同一块显卡的显存上。

```{.python .input  n=45}
z = x.cuda()
(z + 2).exp() * y
```

```{.json .output n=45}
[
 {
  "data": {
   "text/plain": "tensor([ 20.0855, 109.1963, 445.2395], device='cuda:0')"
  },
  "execution_count": 45,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

注意，Pytorch要求计算的所有输入数据都在内存或同一块显卡的显存上。这样设计的原因是CPU和不同的GPU之间的数据交互通常比较耗时。因此，Pytorch希望用户确切地指明计算的输入数据都在内存或同一块显卡的显存上。例如，如果将内存上的`Tensor`变量`x`和显存上的`Tensor`变量`y`做运算，会出现错误信息。当我们打印`Tensor`或将`Tensor`格式时，如果数据不在内存里，Pytorch会将它先复制到内存，从而造成额外的传输开销。

## Pytorch的GPU计算

同`Tensor`类似，Pytorch的模型可以在初始化时通过`.to`函数指定设备。下面的代码将模型参数初始化在显存上。

```{.python .input  n=46}
net = nn.Sequential(nn.Linear(3,1))
net = net.to(device='cuda:0')
```

当输入是显存上的`Tensor`时，Pytorch会在同一块显卡的显存上计算结果。

```{.python .input  n=47}
net(y)
```

```{.json .output n=47}
[
 {
  "data": {
   "text/plain": "tensor([-0.3202], device='cuda:0', grad_fn=<AddBackward0>)"
  },
  "execution_count": 47,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

下面我们确认一下模型参数存储在同一块显卡的显存上。

```{.python .input  n=49}
net[0].weight.data
```

```{.json .output n=49}
[
 {
  "data": {
   "text/plain": "tensor([[-0.5015, -0.5570,  0.5566]], device='cuda:0')"
  },
  "execution_count": 49,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## 小结

* Pytorch可以指定用来存储和计算的设备，如使用内存的CPU或者使用显存的GPU。在默认情况下，Pytorch会将数据创建在内存，然后利用CPU来计算。
* Pytorch要求计算的所有输入数据都在内存或同一块显卡的显存上。

## 练习

* 试试大一点儿的计算任务，如大矩阵的乘法，看看使用CPU和GPU的速度区别。如果是计算量很小的任务呢？
* GPU上应如何读写模型参数？




## 参考文献

[1] CUDA下载地址。 https://developer.nvidia.com/cuda-downloads

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/988)

![](../img/qr_use-gpu.svg)
