# 循环神经网络的简洁实现

本节将使用Gluon来更简洁地实现基于循环神经网络的语言模型。首先，我们读取周杰伦专辑歌词数据集。

```{.python .input  n=1}
import d2lzh as d2l
import math
import torch
import torch.nn as nn
import time
import torch.nn.functional as F
```

```{.python .input  n=2}
(corpus_indices, char_to_idx, idx_to_char,
 vocab_size) = d2l.load_data_jay_lyrics()
```

## 定义模型

Gluon的`rnn`模块提供了循环神经网络的实现。下面构造一个含单隐藏层、隐藏单元个数为256的循环神经网络层`rnn_layer`，并对权重做初始化。

```{.python .input  n=3}
num_hiddens = 256
n_layers = 1
rnn_layer = nn.RNN(vocab_size, num_hiddens, num_layers=n_layers)
```

接下来调用`rnn_layer`的成员函数`begin_state`来返回初始化的隐藏状态列表。它有一个形状为(隐藏层个数, 批量大小, 隐藏单元个数)的元素。

```{.python .input  n=4}
batch_size = 2
state = torch.randn((n_layers, batch_size, num_hiddens))
state[0].shape
```

```{.json .output n=4}
[
 {
  "data": {
   "text/plain": "torch.Size([2, 256])"
  },
  "execution_count": 4,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

与上一节中实现的循环神经网络不同，这里`rnn_layer`的输入形状为(时间步数, 批量大小, 输入个数)。其中输入个数即one-hot向量长度（词典大小）。此外，`rnn_layer`作为Pytorch的`rnn.RNN`实例，在前向计算后会分别返回输出和隐藏状态，其中输出指的是隐藏层在各个时间步上计算并输出的隐藏状态，它们通常作为后续输出层的输入。需要强调的是，该“输出”本身并不涉及输出层计算，形状为(时间步数, 批量大小, 隐藏单元个数)。而`rnn.RNN`实例在前向计算返回的隐藏状态指的是隐藏层在最后时间步的可用于初始化下一时间步的隐藏状态：当隐藏层有多层时，每一层的隐藏状态都会记录在该变量中；对于像长短期记忆这样的循环神经网络，该变量还会包含其他信息。我们会在本章的后面介绍长短期记忆和深度循环神经网络。

```{.python .input  n=5}
num_steps = 35
X = torch.randn(size=(num_steps, batch_size, vocab_size))
Y, state_new = rnn_layer(X, state)
Y.shape, len(state_new), state_new[0].shape
```

```{.json .output n=5}
[
 {
  "data": {
   "text/plain": "(torch.Size([35, 2, 256]), 1, torch.Size([2, 256]))"
  },
  "execution_count": 5,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

接下来我们继承`Module`类来定义一个完整的循环神经网络。它首先将输入数据使用one-hot向量表示后输入到`rnn_layer`中，然后使用全连接输出层得到输出。输出个数等于词典大小`vocab_size`。

```{.python .input  n=6}
class RNNModel(nn.Module):
    """RNN model."""

    def __init__(self, rnn_layer, num_inputs, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.Linear = nn.Linear(num_inputs, vocab_size)

    def forward(self, inputs, state):
        """Forward function"""
        X = F.one_hot(inputs.long().transpose(0,-1), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        output = self.Linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, num_hiddens, device, batch_size=1, num_layers=1):
        """Return the begin state"""
        if num_layers == 1:
          return  torch.zeros(size=(1, batch_size, num_hiddens), dtype=torch.float32, device=device)
        else:
          return (torch.zeros(size=(1, batch_size, num_hiddens), dtype=torch.float32, device=device),
                  torch.zeros(size=(1, batch_size, num_hiddens), dtype=torch.float32, device=device))
```

## 训练模型

同上一节一样，下面定义一个预测函数。这里的实现区别在于前向计算和初始化隐藏状态的函数接口。

```{.python .input  n=7}
def predict_rnn_nn(prefix, pred_len, model, num_hiddens, vocab_size, device, idx_to_char, char_to_idx, num_layers=1):
    """Predict next chars with a RNN model."""
    # Use the model's member function to initialize the hidden state
    state = model.begin_state(num_hiddens=num_hiddens, device=device, num_layers=num_layers)
    output = [char_to_idx[prefix[0]]]
    for t in range(pred_len + len(prefix) - 1):
        X = torch.tensor([output[-1]], dtype=torch.float32, device=device).reshape((1, 1))
        # Forward computation does not require incoming model parameters
        (Y, state) = model(X, state)
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(Y.argmax(dim=1).item()))
    return ''.join([idx_to_char[i] for i in output])
```

让我们使用权重为随机值的模型来预测一次。

```{.python .input  n=8}
device = d2l.try_gpu()
model = RNNModel(rnn_layer, num_hiddens, vocab_size)
model.to(device)
predict_rnn_nn('分开', 10, model, num_hiddens, vocab_size, device, idx_to_char, char_to_idx)
```

```{.json .output n=8}
[
 {
  "data": {
   "text/plain": "'\u5206\u5f00?\u5403\u61fc\u5a25\u5f04\u5a6a\u80af?\u534a\u526a'"
  },
  "execution_count": 8,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

接下来实现训练函数。算法同上一节的一样，但这里只使用了相邻采样来读取数据。

```{.python .input  n=10}
def train_and_predict_rnn_nn(model, num_hiddens, vocab_size, device,
                                corpus_indices, idx_to_char, char_to_idx,
                                num_epochs, num_steps, lr, clipping_theta,
                                batch_size, pred_period, pred_len, prefixes, num_layers=1):
    """Train a RNN model and predict the next item in the sequence."""
    loss =  nn.CrossEntropyLoss()
    optm = torch.optim.Adam(model.parameters(), lr=lr)
    start = time.time()
    count = 0
    for epoch in range(1, num_epochs+1):
        l_sum, n = 0.0, 0
        data_iter = d2l.data_iter_consecutive(
            corpus_indices, batch_size, num_steps, device)
        state = model.begin_state(batch_size=batch_size, num_hiddens=num_hiddens, device=device ,num_layers=num_layers)
        for X, Y in data_iter:
            for s in state:
                s.detach()
            X = X.to(dtype=torch.long)
            (output, state) = model(X, state)
            y = Y.t().reshape((-1,))
            l = loss(output, y.long()).mean()
            optm.zero_grad()
            l.backward(retain_graph=True)
            with torch.no_grad():
                # Clip the gradient
                d2l.grad_clipping(model.parameters(), clipping_theta, device)
                # Since the error has already taken the mean, the gradient does
                # not need to be averaged
                optm.step()
            l_sum += l.item() * y.numel()
            n += y.numel()

        if epoch % (num_epochs // 4) == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch, math.exp(l_sum / n), time.time() - start))
            start = time.time()
        if epoch % (num_epochs // 2) == 0:
            for prefix in prefixes:
                print(' -', predict_rnn_nn(
                    prefix, pred_len, model, num_hiddens, vocab_size, device, idx_to_char, char_to_idx))
```

使用和上一节实验中一样的超参数来训练模型。

```{.python .input  n=11}
num_epochs, batch_size, lr, clipping_theta = 1000, 32, 1e-3, 1e-2
pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']
train_and_predict_rnn_nn(model, num_hiddens, vocab_size, device,
                            corpus_indices, idx_to_char, char_to_idx,
                            num_epochs, num_steps, lr, clipping_theta,
                            batch_size, pred_period, pred_len, prefixes)
```

```{.json .output n=11}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "epoch 250, perplexity 19.913002, time 507.07 sec\nepoch 500, perplexity 1.042176, time 491.49 sec\n - \u5206\u5f00 \u8fd9\u6837\u7684\u9f13\u52b1 \u662f\u5426\u592a\u76f4\u63a5\u592a\u8bbd\u523a \u8001\u5e08\u5728\u8bb2 \u5230\u5e95\u6709\u6ca1\u6709\u5728\u542c\u554a  \u544a\u8bc9\u4f60 \u505a\u81ea\u5df1\u80dc\u4e8e\u8ddf\u592a\u7d27 \u6700\u5927\u7684\u654c\u4eba \n - \u4e0d\u5206\u5f00 \u8fd9\u6837\u7684\u9f13\u52b1 \u662f\u5426\u592a\u76f4\u63a5\u592a\u8bbd\u523a \u8001\u5e08\u5728\u8bb2 \u5230\u5e95\u6709\u6ca1\u6709\u5728\u542c\u554a  \u544a\u8bc9\u4f60 \u505a\u81ea\u5df1\u80dc\u4e8e\u8ddf\u592a\u7d27 \u6700\u5927\u7684\u654c\u4eba \nepoch 750, perplexity 1.017083, time 489.12 sec\nepoch 1000, perplexity 1.011839, time 489.83 sec\n - \u5206\u5f00 \u53eb\u51fa\u6765\u81ea \u6211\u7528\u4ed6\u771f\u7684\u5174\u5c31\u6c14\u76f8\u89c1 \u4e3a\u4e00\u8f6c\u5f2f\u98d8\u79fb  \u52a0\u8db3\u4e86\u9a6c\u529b\u98d8\u5230\u5e95\u770b\u4ed4\u7ec6  \u96f6\u5230\u4e00\u767e\u516c\u91cc\u8c01\u6562\u4e0e\u6211\u4e3a\u654c\n - \u4e0d\u5206\u5f00 \u4e0d\u51fa\u6765\u81ea \u5206\u624b\u8bf4\u4e0d\u51fa\u6765 \u851a\u84dd\u7684\u73ca\u745a\u6d77 \u9519\u8fc7\u77ac\u95f4\u82cd\u767d \u5f53\u521d\u5f7c\u6b64 \u4f60\u6211\u90fd  \u4e0d\u591f\u6210\u719f\u5766\u767d  \u4e0d\u5e94\u8be5  \n"
 }
]
```

## 小结

* torch的`nn`模块提供了循环神经网络层的实现。
* torch的`nn.RNN`实例在前向计算后会分别返回输出和隐藏状态。该前向计算并不涉及输出层计算。

## 练习

* 与上一节的实现进行比较。看看torch的实现是不是运行速度更快？如果你觉得差别明显，试着找找原因。


## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/4089)

![](../img/qr_rnn-gluon.svg)

```{.python .input}

```
