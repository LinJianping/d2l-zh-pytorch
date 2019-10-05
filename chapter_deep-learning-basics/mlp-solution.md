# 多层感知机的从零开始实现

我们已经从上一节里了解了多层感知机的原理。下面，我们一起来动手实现一个多层感知机。首先导入实现所需的包或模块。

```{.python .input  n=21}
%matplotlib inline
import d2lzh as d2l
import torch
```

## 获取和读取数据

这里继续使用Fashion-MNIST数据集。我们将使用多层感知机对图像进行分类。

```{.python .input  n=22}
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```

## 定义模型参数

我们在[“softmax回归的从零开始实现”](softmax-regression-scratch.md)一节里已经介绍了，Fashion-MNIST数据集中图像形状为$28 \times 28$，类别数为10。本节中我们依然使用长度为$28 \times 28 = 784$的向量表示每一张图像。因此，输入个数为784，输出个数为10。实验中，我们设超参数隐藏单元个数为256。

```{.python .input  n=23}
num_inputs, num_outputs, num_hiddens = 784, 10, 256
weight_scale = 0.01

W1 = torch.zeros(size=(num_inputs, num_hiddens)).normal_(std=weight_scale)
b1 = torch.zeros(num_hiddens)
W2 = torch.zeros(size=(num_hiddens, num_outputs)).normal_(std=weight_scale)
b2 = torch.zeros(num_outputs)

params = [W1, b1, W2, b2]
W1.requires_grad_(True)
b1.requires_grad_(True)
W2.requires_grad_(True)
b2.requires_grad_(True)
```

```{.json .output n=23}
[
 {
  "data": {
   "text/plain": "tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], requires_grad=True)"
  },
  "execution_count": 23,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## 定义激活函数

这里我们使用基础的`maximum`函数来实现ReLU，而非直接调用`relu`函数。

```{.python .input  n=24}
def relu(X):
    return X.clamp(min=0)
```

## 定义模型

同softmax回归一样，我们通过`reshape`函数将每张原始图像改成长度为`num_inputs`的向量。然后我们实现上一节中多层感知机的计算表达式。

```{.python .input  n=25}
def net(X):
    X = X.reshape((-1, num_inputs))
    h1 = torch.mm(X,W1) + b1
    h1 = relu(h1)
    output = torch.mm(h1,W2) + b2
    return output
```

## 定义损失函数

为了得到更好的数值稳定性，我们直接使用torch提供的包括softmax运算和交叉熵损失计算的函数。

```{.python .input  n=26}
def softmaxCrossEntropyLoss(y_hat, y):
    y_hat_exp = y_hat.exp()
    out_softmax = y_hat_exp / y_hat_exp.sum(dim=1, keepdim = True)
    return -torch.gather(out_softmax, 1, y.unsqueeze(dim=1)).log()
```

## 训练模型

训练多层感知机的步骤和[“softmax回归的从零开始实现”](softmax-regression-scratch.md)一节中训练softmax回归的步骤没什么区别。我们在这里设超参数迭代周期数为5，学习率为0.5。

```{.python .input  n=27}
num_epochs, lr = 5, 0.5
loss = softmaxCrossEntropyLoss
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).sum().item()
        n += y.size()[0]  # y.size()[0] = batch_size
    return acc_sum / n

def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params=None, lr=None, trainer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()
            l.backward()
            if trainer is None:
                d2l.sgd(params, lr, batch_size)
            else:
                # This will be illustrated in the next section
                trainer.step(batch_size)
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.size()[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))

train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params, lr)
```

```{.json .output n=27}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "epoch 1, loss 0.7978, train acc 0.702, test acc 0.814\nepoch 2, loss 0.4864, train acc 0.820, test acc 0.841\nepoch 3, loss 0.4244, train acc 0.844, test acc 0.844\nepoch 4, loss 0.3924, train acc 0.854, test acc 0.842\nepoch 5, loss 0.3722, train acc 0.863, test acc 0.862\n"
 }
]
```

## 小结

* 可以通过手动定义模型及其参数来实现简单的多层感知机。
* 当多层感知机的层数较多时，本节的实现方法会显得较烦琐，例如在定义模型参数的时候。

## 练习

* 改变超参数`num_hiddens`的值，看看对实验结果有什么影响。
* 试着加入一个新的隐藏层，看看对实验结果有什么影响。



## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/739)

![](../img/qr_mlp-scratch.svg)
