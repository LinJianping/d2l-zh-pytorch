# softmax回归的从零开始实现

这一节我们来动手实现softmax回归。首先导入本节实现所需的包或模块。

```{.python .input  n=1}
%matplotlib inline
import d2lzh as d2l
import torch
```

## 获取和读取数据

我们将使用Fashion-MNIST数据集，并设置批量大小为256。

```{.python .input  n=2}
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```

## 初始化模型参数

跟线性回归中的例子一样，我们将使用向量表示每个样本。已知每个样本输入是高和宽均为28像素的图像。模型的输入向量的长度是$28 \times 28 = 784$：该向量的每个元素对应图像中每个像素。由于图像有10个类别，单层神经网络输出层的输出个数为10，因此softmax回归的权重和偏差参数分别为$784 \times 10$和$1 \times 10$的矩阵。

```{.python .input  n=4}
num_inputs = 784
num_outputs = 10

W = torch.zeros(size=(num_inputs, num_outputs)).normal_(std=0.01)
b = torch.zeros(num_outputs)
```

同之前一样，我们要为模型参数附上梯度。

```{.python .input  n=5}
W.requires_grad_(True)
b.requires_grad_(True)
```

```{.json .output n=5}
[
 {
  "data": {
   "text/plain": "tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], requires_grad=True)"
  },
  "execution_count": 5,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## 实现softmax运算

在介绍如何定义softmax回归之前，我们先描述一下对如何对多维`Tensor`按维度操作。在下面的例子中，给定一个`Tensor`矩阵`X`。我们可以只对其中同一列（`axis=0`）或同一行（`axis=1`）的元素求和，并在结果中保留行和列这两个维度（`keepdims=True`）。

```{.python .input  n=6}
X = torch.tensor([[1, 2, 3], [4, 5, 6]])
torch.sum(X, dim=0, keepdim=True), torch.sum(X, dim=1, keepdim=True)
```

```{.json .output n=6}
[
 {
  "data": {
   "text/plain": "(tensor([[5, 7, 9]]), tensor([[ 6],\n         [15]]))"
  },
  "execution_count": 6,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

下面我们就可以定义前面小节里介绍的softmax运算了。在下面的函数中，矩阵`X`的行数是样本数，列数是输出个数。为了表达样本预测各个输出的概率，softmax运算会先通过`exp`函数对每个元素做指数运算，再对`exp`矩阵同行元素求和，最后令矩阵每行各元素与该行元素之和相除。这样一来，最终得到的矩阵每行元素和为1且非负。因此，该矩阵每行都是合法的概率分布。softmax运算的输出矩阵中的任意一行元素代表了一个样本在各个输出类别上的预测概率。

```{.python .input  n=7}
def softmax(X):

```

可以看到，对于随机输入，我们将每个元素变成了非负数，且每一行和为1。

```{.python .input  n=8}
X = torch.randn(size=(2, 5))
X_prob = softmax(X)
X_prob, X_prob.sum(dim=1)
```

```{.json .output n=8}
[
 {
  "data": {
   "text/plain": "(tensor([[0.1104, 0.5147, 0.0714, 0.0307, 0.2729],\n         [0.1947, 0.3210, 0.0968, 0.1757, 0.2118]]), tensor([1., 1.]))"
  },
  "execution_count": 8,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## 定义模型

有了softmax运算，我们可以定义上节描述的softmax回归模型了。这里通过`reshape`函数将每张原始图像改成长度为`num_inputs`的向量。

```{.python .input  n=9}
def net(X):

```

## 定义损失函数

上一节中，我们介绍了softmax回归使用的交叉熵损失函数。为了得到标签的预测概率，我们可以使用`gather`函数。在下面的例子中，变量`y_hat`是2个样本在3个类别的预测概率，变量`y`是这2个样本的标签类别。通过使用`pick`函数，我们得到了2个样本的标签的预测概率。与[“softmax回归”](softmax-regression.md)一节数学表述中标签类别离散值从1开始逐一递增不同，在代码中，标签类别的离散值是从0开始逐一递增的。

```{.python .input  n=10}
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = torch.tensor([0, 2], dtype=torch.long)
torch.gather(y_hat, 1, y.unsqueeze(dim=1))
```

```{.json .output n=10}
[
 {
  "data": {
   "text/plain": "tensor([[0.1000],\n        [0.5000]])"
  },
  "execution_count": 10,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

下面实现了[“softmax回归”](softmax-regression.md)一节中介绍的交叉熵损失函数。

```{.python .input  n=11}
def cross_entropy(y_hat, y):

```

## 计算分类准确率

给定一个类别的预测概率分布`y_hat`，我们把预测概率最大的类别作为输出类别。如果它与真实类别`y`一致，说明这次预测是正确的。分类准确率即正确预测数量与总预测数量之比。

为了演示准确率的计算，下面定义准确率`accuracy`函数。其中`y_hat.argmax(dim=1)`返回矩阵`y_hat`每行中最大元素的索引，且返回结果与变量`y`形状相同。我们在[“数据操作”](../chapter_prerequisite/ndarray.md)一节介绍过，相等条件判断式`(y_hat.argmax(dim=1) == y)`是一个值为0（相等为假）或1（相等为真）的`Tensor`。由于标签类型为整数，我们先将变量`y`变换为浮点数再进行相等条件判断。

```{.python .input  n=12}
def accuracy(y_hat, y):

```

让我们继续使用在演示`gather`函数时定义的变量`y_hat`和`y`，并将它们分别作为预测概率分布和标签。可以看到，第一个样本预测类别为2（该行最大元素0.6在本行的索引为2），与真实标签0不一致；第二个样本预测类别为2（该行最大元素0.5在本行的索引为2），与真实标签2一致。因此，这两个样本上的分类准确率为0.5。

```{.python .input  n=13}
accuracy(y_hat, y)
```

```{.json .output n=13}
[
 {
  "data": {
   "text/plain": "0.5"
  },
  "execution_count": 13,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

类似地，我们可以评价模型`net`在数据集`data_iter`上的准确率。

```{.python .input  n=17}
# 本函数已保存在d2lzh包中方便以后使用。该函数将被逐步改进：它的完整实现将在“图像增广”一节中
# 描述
def evaluate_accuracy(data_iter, net):

```

因为我们随机初始化了模型`net`，所以这个随机模型的准确率应该接近于类别个数10的倒数0.1。

```{.python .input  n=18}
evaluate_accuracy(test_iter, net)
```

```{.json .output n=18}
[
 {
  "data": {
   "text/plain": "0.0958"
  },
  "execution_count": 18,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## 训练模型

训练softmax回归的实现跟[“线性回归的从零开始实现”](linear-regression-scratch.md)一节介绍的线性回归中的实现非常相似。我们同样使用小批量随机梯度下降来优化模型的损失函数。在训练模型时，迭代周期数`num_epochs`和学习率`lr`都是可以调的超参数。改变它们的值可能会得到分类更准确的模型。

```{.python .input  n=19}
num_epochs, lr = 5, 0.1

# 本函数已保存在d2lzh包中方便以后使用
def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, trainer=None):
    for epoch in range(num_epochs):

train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size,
          [W, b], lr)
```

```{.json .output n=19}
[
 {
  "ename": "RuntimeError",
  "evalue": "Expected object of scalar type Long but got scalar type Float for argument #2 'other'",
  "output_type": "error",
  "traceback": [
   "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
   "\u001b[0;31mRuntimeError\u001b[0m                              Traceback (most recent call last)",
   "\u001b[0;32m<ipython-input-19-622651a59170>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m     23\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     24\u001b[0m train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size,\n\u001b[0;32m---> 25\u001b[0;31m           [W, b], lr)\n\u001b[0m",
   "\u001b[0;32m<ipython-input-19-622651a59170>\u001b[0m in \u001b[0;36mtrain_ch3\u001b[0;34m(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr, trainer)\u001b[0m\n\u001b[1;32m     16\u001b[0m             \u001b[0my\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0my\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfloat\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     17\u001b[0m             \u001b[0mtrain_l_sum\u001b[0m \u001b[0;34m+=\u001b[0m \u001b[0ml\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mitem\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 18\u001b[0;31m             \u001b[0mtrain_acc_sum\u001b[0m \u001b[0;34m+=\u001b[0m \u001b[0;34m(\u001b[0m\u001b[0my_hat\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0margmax\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdim\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;34m==\u001b[0m \u001b[0my\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0msum\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mitem\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     19\u001b[0m             \u001b[0mn\u001b[0m \u001b[0;34m+=\u001b[0m \u001b[0my\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0msize\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m0\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     20\u001b[0m         \u001b[0mtest_acc\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mevaluate_accuracy\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mtest_iter\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mnet\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;31mRuntimeError\u001b[0m: Expected object of scalar type Long but got scalar type Float for argument #2 'other'"
  ]
 }
]
```

## 预测

训练完成后，现在就可以演示如何对图像进行分类了。给定一系列图像（第三行图像输出），我们比较一下它们的真实标签（第一行文本输出）和模型预测结果（第二行文本输出）。

```{.python .input}
for X, y in test_iter:
    break

true_labels = d2l.get_fashion_mnist_labels(y.numpy())
pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

d2l.show_fashion_mnist(X[0:9], titles[0:9])
```

## 小结
 
* 可以使用softmax回归做多类别分类。与训练线性回归相比，你会发现训练softmax回归的步骤和它非常相似：获取并读取数据、定义模型和损失函数并使用优化算法训练模型。事实上，绝大多数深度学习模型的训练都有着类似的步骤。

## 练习

* 在本节中，我们直接按照softmax运算的数学定义来实现softmax函数。这可能会造成什么问题？（提示：试一试计算$\exp(50)$的大小。）
* 本节中的`cross_entropy`函数是按照[“softmax回归”](softmax-regression.md)一节中的交叉熵损失函数的数学定义实现的。这样的实现方式可能有什么问题？（提示：思考一下对数函数的定义域。）
* 你能想到哪些办法来解决上面的两个问题？



## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/741)

![](../img/qr_softmax-regression-scratch.svg)
