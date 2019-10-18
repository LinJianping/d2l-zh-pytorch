# 循环神经网络的从零开始实现

在本节中，我们将从零开始实现一个基于字符级循环神经网络的语言模型，并在周杰伦专辑歌词数据集上训练一个模型来进行歌词创作。首先，我们读取周杰伦专辑歌词数据集：

```{.python .input  n=1}
import d2lzh as d2l
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
```

```{.python .input  n=2}
(corpus_indices, char_to_idx, idx_to_char,
 vocab_size) = d2l.load_data_jay_lyrics()
```

## one-hot向量

为了将词表示成向量输入到神经网络，一个简单的办法是使用one-hot向量。假设词典中不同字符的数量为$N$（即词典大小`vocab_size`），每个字符已经同一个从0到$N-1$的连续整数值索引一一对应。如果一个字符的索引是整数$i$, 那么我们创建一个全0的长为$N$的向量，并将其位置为$i$的元素设成1。该向量就是对原字符的one-hot向量。下面分别展示了索引为0和2的one-hot向量，向量长度等于词典大小。

```{.python .input  n=3}
F.one_hot(torch.tensor([0,2]), vocab_size)
```

```{.json .output n=3}
[
 {
  "data": {
   "text/plain": "tensor([[1, 0, 0,  ..., 0, 0, 0],\n        [0, 0, 1,  ..., 0, 0, 0]])"
  },
  "execution_count": 3,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

我们每次采样的小批量的形状是(批量大小, 时间步数)。下面的函数将这样的小批量变换成数个可以输入进网络的形状为(批量大小, 词典大小)的矩阵，矩阵个数等于时间步数。也就是说，时间步$t$的输入为$\boldsymbol{X}_t \in \mathbb{R}^{n \times d}$，其中$n$为批量大小，$d$为输入个数，即one-hot向量长度（词典大小）。

```{.python .input  n=4}
def to_onehot(X, size):  # 本函数已保存在d2lzh包中方便以后使用
    return F.one_hot(X.long().transpose(0,-1), size)

X = torch.arange(10).reshape((2, 5))
inputs = to_onehot(X, vocab_size)
len(inputs), inputs[0].shape
```

```{.json .output n=4}
[
 {
  "data": {
   "text/plain": "(5, torch.Size([2, 2582]))"
  },
  "execution_count": 4,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## 初始化模型参数

接下来，我们初始化模型参数。隐藏单元个数 `num_hiddens`是一个超参数。

```{.python .input  n=5}
num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
ctx = d2l.try_gpu()
print('will use', ctx)

def get_params():
    def _one(shape):
        return torch.empty(size=shape, device=ctx).normal_(std=0.01)

    # 隐藏层参数
    W_xh = _one((num_inputs, num_hiddens))
    W_hh = _one((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=ctx)
    # 输出层参数
    W_hq = _one((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=ctx)
    # 附上梯度
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params
```

```{.json .output n=5}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "will use cuda:0\n"
 }
]
```

## 定义模型

我们根据循环神经网络的计算表达式实现该模型。首先定义`init_rnn_state`函数来返回初始化的隐藏状态。它返回由一个形状为(批量大小, 隐藏单元个数)的值为0的`Tensor`组成的元组。使用元组是为了更便于处理隐藏状态含有多个`Tensor`的情况。

```{.python .input  n=6}
def init_rnn_state(batch_size, num_hiddens, ctx):
    return (torch.zeros(size=(batch_size, num_hiddens), device=ctx), )
```

下面的`rnn`函数定义了在一个时间步里如何计算隐藏状态和输出。这里的激活函数使用了tanh函数。[“多层感知机”](../chapter_deep-learning-basics/mlp.md)一节中介绍过，当元素在实数域上均匀分布时，tanh函数值的均值为0。

```{.python .input  n=7}
def rnn(inputs, state, params):
    # inputs和outputs皆为num_steps个形状为(batch_size, vocab_size)的矩阵
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        H = torch.tanh(torch.mm(X.float(), W_xh) + torch.mm(H.float(), W_hh) + b_h)
        Y = torch.mm(H.float(), W_hq) + b_q
        outputs.append(Y)
    return outputs, (H,)
```

做个简单的测试来观察输出结果的个数（时间步数），以及第一个时间步的输出层输出的形状和隐藏状态的形状。

```{.python .input  n=8}
state = init_rnn_state(X.shape[0], num_hiddens, ctx)
inputs = to_onehot(X.to(ctx), vocab_size)
params = get_params()
outputs, state_new = rnn(inputs, state, params)
len(outputs), outputs[0].shape, state_new[0].shape
```

```{.json .output n=8}
[
 {
  "data": {
   "text/plain": "(5, torch.Size([2, 2582]), torch.Size([2, 256]))"
  },
  "execution_count": 8,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## 定义预测函数

以下函数基于前缀`prefix`（含有数个字符的字符串）来预测接下来的`num_chars`个字符。这个函数稍显复杂，其中我们将循环神经单元`rnn`设置成了函数参数，这样在后面小节介绍其他循环神经网络时能重复使用这个函数。

```{.python .input  n=9}
# 本函数已保存在d2lzh包中方便以后使用
def predict_rnn(prefix, num_chars, rnn, params, init_rnn_state,
                num_hiddens, vocab_size, ctx, idx_to_char, char_to_idx):
    state = init_rnn_state(1, num_hiddens, ctx)
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        # 将上一时间步的输出作为当前时间步的输入
        X = to_onehot(torch.tensor([output[-1]], device=ctx), vocab_size)
        X = X.unsqueeze(dim=0)
        # 计算输出和更新隐藏状态
        (Y, state) = rnn(X, state, params)
        # 下一个时间步的输入是prefix里的字符或者当前的最佳预测字符
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(Y[0].argmax(dim=1).item()))
    return ''.join([idx_to_char[i] for i in output])
```

我们先测试一下`predict_rnn`函数。我们将根据前缀“分开”创作长度为10个字符（不考虑前缀长度）的一段歌词。因为模型参数为随机值，所以预测结果也是随机的。

```{.python .input  n=10}
predict_rnn('分开', 10, rnn, params, init_rnn_state, num_hiddens, vocab_size,
            ctx, idx_to_char, char_to_idx)
```

```{.json .output n=10}
[
 {
  "data": {
   "text/plain": "'\u5206\u5f00\u6dcb\u6dcb\u6dcb\u6dcb\u6dcb\u6dcb\u6dcb\u6dcb\u6dcb\u6dcb'"
  },
  "execution_count": 10,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## 裁剪梯度

循环神经网络中较容易出现梯度衰减或梯度爆炸。我们会在[“通过时间反向传播”](bptt.md)一节中解释原因。为了应对梯度爆炸，我们可以裁剪梯度（clip gradient）。假设我们把所有模型参数梯度的元素拼接成一个向量 $\boldsymbol{g}$，并设裁剪的阈值是$\theta$。裁剪后的梯度

$$ \min\left(\frac{\theta}{\|\boldsymbol{g}\|}, 1\right)\boldsymbol{g}$$

的$L_2$范数不超过$\theta$。

```{.python .input  n=11}
# 本函数已保存在d2lzh包中方便以后使用
def grad_clipping(params, theta, ctx):
    norm = torch.tensor([0], device=ctx, dtype=torch.float32)
    for param in params:
        norm += (param.grad ** 2).sum()
    norm = norm.sqrt().item()
    if norm > theta:
        for param in params:
            param.grad.data.mul_(theta / norm)
```

## 困惑度

我们通常使用困惑度（perplexity）来评价语言模型的好坏。回忆一下[“softmax回归”](../chapter_deep-learning-basics/softmax-regression.md)一节中交叉熵损失函数的定义。困惑度是对交叉熵损失函数做指数运算后得到的值。特别地，

* 最佳情况下，模型总是把标签类别的概率预测为1，此时困惑度为1；
* 最坏情况下，模型总是把标签类别的概率预测为0，此时困惑度为正无穷；
* 基线情况下，模型总是预测所有类别的概率都相同，此时困惑度为类别个数。

显然，任何一个有效模型的困惑度必须小于类别个数。在本例中，困惑度必须小于词典大小`vocab_size`。

## 定义模型训练函数

跟之前章节的模型训练函数相比，这里的模型训练函数有以下几点不同：

1. 使用困惑度评价模型。
2. 在迭代模型参数前裁剪梯度。
3. 对时序数据采用不同采样方法将导致隐藏状态初始化的不同。相关讨论可参考[“语言模型数据集（周杰伦专辑歌词）”](lang-model-dataset.md)一节。

另外，考虑到后面将介绍的其他循环神经网络，为了更通用，这里的函数实现更长一些。

```{.python .input  n=12}
# 本函数已保存在d2lzh包中方便以后使用
def train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                          vocab_size, ctx, corpus_indices, idx_to_char,
                          char_to_idx, is_random_iter, num_epochs, num_steps,
                          lr, clipping_theta, batch_size, pred_period,
                          pred_len, prefixes):
    if is_random_iter:
        data_iter_fn = d2l.data_iter_random
    else:
        data_iter_fn = d2l.data_iter_consecutive
    params = get_params()
    loss = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        if not is_random_iter:  # 如使用相邻采样，在epoch开始时初始化隐藏状态
            state = init_rnn_state(batch_size, num_hiddens, ctx)
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_fn(corpus_indices, batch_size, num_steps, ctx)
        for X, Y in data_iter:
            if is_random_iter:  # 如使用随机采样，在每个小批量更新前初始化隐藏状态
                state = init_rnn_state(batch_size, num_hiddens, ctx)
            else:  # 否则需要使用detach函数从计算图分离隐藏状态
                for s in state:
                    s.detach_()
            with torch.enable_grad():
                inputs = to_onehot(X, vocab_size)
                # outputs有num_steps个形状为(batch_size, vocab_size)的矩阵
                (outputs, state) = rnn(inputs, state, params)
#                 print('outputs', outputs)
#                 print('state:', state.shape)
                # 拼接之后形状为(num_steps * batch_size, vocab_size)
                outputs = torch.cat(outputs, dim=0)
                # Y的形状是(batch_size, num_steps)，转置后再变成长度为
                # batch * num_steps 的向量，这样跟输出的行一一对应
                y = Y.t().reshape((-1,))
                # 使用交叉熵损失计算平均分类误差
                l = loss(outputs, y.long()).mean()
                l.backward()
            with torch.no_grad():
                grad_clipping(params, clipping_theta, ctx)  # 裁剪梯度
                d2l.sgd(params, lr, 1)  # 因为误差已经取过均值，梯度不用再做平均
            l_sum += l.item() * y.numel()
            n += y.numel()

        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, math.exp(l_sum / n), time.time() - start))
            for prefix in prefixes:
                print(' -', predict_rnn(
                    prefix, pred_len, rnn, params, init_rnn_state,
                    num_hiddens, vocab_size, ctx, idx_to_char, char_to_idx))
```

## 训练模型并创作歌词

现在我们可以训练模型了。首先，设置模型超参数。我们将根据前缀“分开”和“不分开”分别创作长度为50个字符（不考虑前缀长度）的一段歌词。我们每过50个迭代周期便根据当前训练的模型创作一段歌词。

```{.python .input  n=13}
num_epochs, num_steps, batch_size, lr, clipping_theta = 250, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']
```

下面采用随机采样训练模型并创作歌词。

```{.python .input  n=14}
train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                      vocab_size, ctx, corpus_indices, idx_to_char,
                      char_to_idx, True, num_epochs, num_steps, lr,
                      clipping_theta, batch_size, pred_period, pred_len,
                      prefixes)
```

```{.json .output n=14}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "epoch 50, perplexity 48.211329, time 0.84 sec\n - \u5206\u5f00\u59cb \u4e0d\u77e5\u9ebb\u89c9 \u6211\u4e0d\u60f3\u8981 \u4e00\u904d\u7684\u65f6\u5149 \u4f60\u4e0d\u8981\u6211\u60f3 \u4f60\u4e0d\u8981 \u4e0d\u8981 \u6211 \u6211\u4e0d\u60f3 \u4f60\u8bf4 \u6211\u4e0d\u8981 \u4e0d\u8981\u6211\u4e0d\u8981\u518d\n - \u4e0d\u5206\u5f00  \u6ca1\u6709\u4f60\u7684\u6211 \u6211\u77e5\u9053\u7684\u8bf1 \u6709\u5929 \u6211\u7528                               \nepoch 100, perplexity 17.658490, time 0.84 sec\n - \u5206\u5f00\u59cb\u91cd\u529b \u6211\u7528\u7b2c\u4e00\u8def\u79f0 \u5728\u6c5f\u5e95\u4e66  \u5728\u98d8\u5f85\u98d8   \u54ce\u54df  \u54ce\u54df  \u54ce\u54df  \u54ce\u54df \u529fe  \u8d76\u7d27\u7a7f\u4e0a\u65d7\u888d \n - \u4e0d\u5206\u5f00 \u5bd2\u7af9\u6bdb \u7684\u4f24\u4eba\u5728\u5355\u5916 \u6211\u4e0d\u662f\u4e00\u79cd\u7684\u5c0f\u4e11 \u662f\u4e00\u9897\u8c01 \u8ba9\u6211\u4eec\u8ffd\u7740\u9633\u5149\u3000 \u6211\u7684\u7b11\u3000 \u4f60\u52a1\u4f1a\u4e0d\u53bb \u4f60\u7684\u8f6c\u53d8\nepoch 150, perplexity 9.422075, time 0.84 sec\n - \u5206\u5f00 \u7231\u8ba9\u6211\u8fd8\u5728\u4f60 \u6709\u6ca1\u6709\u53e3\u529b\u7684\u592a\u70ba \u53ea\u6709\u6ca1\u6709\u671f\u96e8 \u77e5\u9053\u98ce\u5728\u98d8\u3000 \u662f\u4ec0\u4e48\u88ab\u6211\u4eec \u5c31\u5f97\u5f88\u96be\u7ec6   \u4f60  \u6211\n - \u4e0d\u5206\u5f00  \u5206\u624b\u4e0d\u8be5   \u6211 \u6211\u7528 \u7275\u7740\u4f60\u7684 \u522b\u95ee \u6211\u4e0d\u518d\u518d\u60f3 \u6211\u4e0d\u8981\u518d\u60f3 \u6211\u4e0d\u8981\u518d\u60f3 \u6211\u4e0d\u8981\u518d\u60f3 \u6211\u4e0d\u8981\u518d\nepoch 200, perplexity 6.784410, time 0.84 sec\n - \u5206\u5f00 \u7231\u7684\u5929\u7a7a \u627e\u627e\u4e00\u79cd\u70ed\u7ca5 \u914d\u56de\u5fc6\u5bf9\u7740\u6211 oh oh                         \n - \u4e0d\u5206\u5f00 \u5929\u771f\u7684\u771f\u7684 \u6211\u5728\u8d77   \u4f60\u5f39                                   \nepoch 250, perplexity 4.724835, time 0.85 sec\n - \u5206\u5f00 \u7231\u8ba9\u6211\u8fd8\u5728\u56de\u5fc6 \u6211\u8bf4\u4f60\u8d70\u5230\u6700\u540e\u7684\u4e00\u53e3 \u4f46\u7275\u624b\u4e2d\u5f00\u4e86 \u6211\u53ea\u6709\u4e00\u79cd\u5c0f\u5fc3 \u6211\u8bf4\u4f60\u4e5f\u4f1a\u4e0d\u8be5 \u4e0d\u8981\u5728\u8fd9\u4e00\u5757\u624d\n - \u4e0d\u5206\u5f00 \u80fd\u6bd4\u6211\u4eec\u7684\u7231\u60c5 \u7c97\u9192\u6210\u5728\u98de\u5730\u65b9\u7684\u58f0 \u4e00\u8d77\u8fc7\u53bb \u6211\u7b49\u5f85 \u65ad\u5f00 \u6211\u4e0d \u6211\u4e0d\u8981\u518d\u60f3\u4f60 \u4e0d\u77e5\u4e0d\u89c9 \u4f60\u5df2\u7ecf\u79bb\n"
 }
]
```

接下来采用相邻采样训练模型并创作歌词。

```{.python .input  n=15}
train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                      vocab_size, ctx, corpus_indices, idx_to_char,
                      char_to_idx, False, num_epochs, num_steps, lr,
                      clipping_theta, batch_size, pred_period, pred_len,
                      prefixes)
```

```{.json .output n=15}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "epoch 50, perplexity 37.490142, time 0.83 sec\n - \u5206\u5f00\u4e86 \u7231\u60c5\u4f60\u7684\u624b \u4f60\u5728\u7075\u96ea \u4f60\u5df2\u5728\u7b49\u4f60 \u6211\u4e0d\u8981 \u6709\u4e00\u4e2a\u7684\u68a6\u5feb \u4f60\u8bf4\u4f60\u8bf4\u5f00 \u6709\u4eba\u6807\u4e0d\u80fd \u6211\u4e0d\u8be5\u8981\u8fd9\u6837 \u6211\n - \u4e0d\u5206\u5f00 \u5bd2\u8fdc\u6211\u7684\u624b \u6c89\u4e00\u70b9\u5473\u9053 \u4e0d\u9700\u662f \u4f60\u4e0d\u60f3\u8981\u4e0d\u8981 \u6211\u8bf4\u4f60\u8fd9\u6837 \u6709\u4f60\u4f1a\u540e\u7ecf\u4e0d\u662f  \u4f60\u5728\u6211 \u6211\u77e5\u6307\u7b11\u677e \u6211\nepoch 100, perplexity 16.972292, time 0.84 sec\n - \u5206\u5f00\u4e86  \u4f60\u8bf4\u4e0d\u662f\u6211\u77e5 \u5982\u679c\u4f60\u4f1a\u7ecf\u4f60\u7684\u597d\u751f \u6211\u5fae\u7b11\u4f60\u7684\u624b\u7b11 \u4f60\u8bf4\u4f60\u8bf4\u4e0d\u89c1\u4f60\u7684\u6837\u5ea6 \u5feb\u683c \u6211\u624b\u4e2d\u7684\u6a21\u5bc2 \u4f60\n - \u4e0d\u5206\u5f00  \u6211\u7528\u4e0d\u89c1\u4f60 \u8fc7\u7684\u60f3\u5e94 \u4e5f\u5728\u4e0d\u77e5 \u6211\u4e86\u7231\u4f60\u597d\u4e0d\u6765 \u5355\u7fbd\u662f\u4e00\u4e2a\u7684\u89d2\u5728 \u4f60\u8bf4\u4f60\u7684\u6307 \u5c06\u5728  \u7684\u7075\u9b42 \u4f60\nepoch 150, perplexity 11.197058, time 0.77 sec\n - \u5206\u5f00\u59cb\u4e0d\u591a \u4e3a\u4ec0\u4e48\u5728\u5e72\u871c\u53cb\u7eed\u7684\u98d8 \u50cf\u4e00\u70b9\u4e0d\u597d \u4f60\u4eec\u53d1\u8ddf\u76f8\u6e10   \u6211\u8981 \u4f60\u7231\u4eba \u7529\u4e86 \u53e3         \n - \u4e0d\u5206\u5f00  \u5206\u624b\u8bf4\u4e0d\u60f3\u6765 \u851a\u84dd\u7684\u73ca\u745a \u6211\u6211\u9047\u4e0d\u53bb   \u7528\u8fc7\u53bb\u4e0d\u662f\u6211\u8981\u4e00\u5b9a\u73cd \u4e5f\u75af\u7528\u6210\u6c14\u4e0d\u77e5\u9053\u4e0d\u5e73 \u6211\u7528\u65e0\u95f4 \nepoch 200, perplexity 8.522952, time 0.80 sec\n - \u5206\u5f00\u4e86\u5343\u5c11 \u7b49\u7740\u6211\u5728\u51fa \u79bb\u5f00\u4f60\u53bb\u4e86 \u5979\u4eba\u4e0d\u662f\u6211\u624b\u4e00\u53e3\u73cd\u54ee \u8c22\u6210  \u6211\u5728\u4f60\u5f39\u51fa \u6211\u624b\u7231\u4f60 \u5996\u8272S\u4eba\u4e0d\u8d77 \u4e00\n - \u4e0d\u5206\u5f00  \u5206\u624b\u8bf4\u4e0d\u8981\u6765 \u6d77\u9e1f\u8ddf\u73ca\u7684\u611f\u5149 \u6211\u8bf4\u5e97\u4e0d\u4e86 \u5c0f\u975e\u5c31\u56de\u4e0d\u5230 \u5148\u7a7a\u7684\u73ca\u524d \u96e8\u7a7a\u7684\u767d\u5468 \u6211\u4f4e\u5934\u5bb3\u7f9e \u5c06\u95ed\nepoch 250, perplexity 6.570468, time 0.82 sec\n - \u5206\u5f00\u59cb\u81ea\u9ed8\u7684\u5929\u7a7a \u4f60\u7684\u8138 \u4f60\u600e\u4e48\u4e00\u5b9a\u53e4\u89e3 \u9ed1\u8272\u7684\u8ba9\u6211\u9762\u7ea2\u7684\u53ef\u7231\u5973\u4eba \u574f\u574f\u7684\u8ba9\u6211\u611f\u52a8\u7684\u53ef\u7231\u5973\u4eba \u574f\u574f\u7684\u8ba9\u6211\n - \u4e0d\u5206\u5f00 \u6cea\u8fd8\u662f\u4f60\u5fae\u7b11\u5821 \u4e00\u904d\u5f00\u5929\u7684\u7ed3\u5c40 \u6211\u8bf4\u4f60\u7684\u5e9f\u7269 \u5728\u96e8\u9762\u7684\u624d\u5feb \u4f60\u5fae\u7b11\u7684\u626d\u52a8 \u5728\u6211\u4eec\u76d8    \u6211\u7528\u4f60\u7684\n"
 }
]
```

## 小结

* 可以用基于字符级循环神经网络的语言模型来生成文本序列，例如创作歌词。
* 当训练循环神经网络时，为了应对梯度爆炸，可以裁剪梯度。
* 困惑度是对交叉熵损失函数做指数运算后得到的值。


## 练习

* 调调超参数，观察并分析对运行时间、困惑度以及创作歌词的结果造成的影响。
* 不裁剪梯度，运行本节中的代码，结果会怎样？
* 将`pred_period`变量设为1，观察未充分训练的模型（困惑度高）是如何创作歌词的。你获得了什么启发？
* 将相邻采样改为不从计算图分离隐藏状态，运行时间有没有变化？
* 将本节中使用的激活函数替换成ReLU，重复本节的实验。



## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/989)

![](../img/qr_rnn-scratch.svg)
