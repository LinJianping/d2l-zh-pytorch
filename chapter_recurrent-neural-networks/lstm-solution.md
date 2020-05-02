# 长短期记忆（LSTM）


本节将介绍另一种常用的门控循环神经网络：长短期记忆（long short-term memory，LSTM）[1]。它比门控循环单元的结构稍微复杂一点。


## 长短期记忆

LSTM 中引入了3个门，即输入门（input gate）、遗忘门（forget gate）和输出门（output gate），以及与隐藏状态形状相同的记忆细胞（某些文献把记忆细胞当成一种特殊的隐藏状态），从而记录额外的信息。


### 输入门、遗忘门和输出门

与门控循环单元中的重置门和更新门一样，如图6.7所示，长短期记忆的门的输入均为当前时间步输入$\boldsymbol{X}_t$与上一时间步隐藏状态$\boldsymbol{H}_{t-1}$，输出由激活函数为sigmoid函数的全连接层计算得到。如此一来，这3个门元素的值域均为$[0,1]$。

![长短期记忆中输入门、遗忘门和输出门的计算](../img/lstm_0.svg)

具体来说，假设隐藏单元个数为$h$，给定时间步$t$的小批量输入$\boldsymbol{X}_t \in \mathbb{R}^{n \times d}$（样本数为$n$，输入个数为$d$）和上一时间步隐藏状态$\boldsymbol{H}_{t-1} \in \mathbb{R}^{n \times h}$。
时间步$t$的输入门$\boldsymbol{I}_t \in \mathbb{R}^{n \times h}$、遗忘门$\boldsymbol{F}_t \in \mathbb{R}^{n \times h}$和输出门$\boldsymbol{O}_t \in \mathbb{R}^{n \times h}$分别计算如下：

$$
\begin{aligned}
\boldsymbol{I}_t &= \sigma(\boldsymbol{X}_t \boldsymbol{W}_{xi} + \boldsymbol{H}_{t-1} \boldsymbol{W}_{hi} + \boldsymbol{b}_i),\\
\boldsymbol{F}_t &= \sigma(\boldsymbol{X}_t \boldsymbol{W}_{xf} + \boldsymbol{H}_{t-1} \boldsymbol{W}_{hf} + \boldsymbol{b}_f),\\
\boldsymbol{O}_t &= \sigma(\boldsymbol{X}_t \boldsymbol{W}_{xo} + \boldsymbol{H}_{t-1} \boldsymbol{W}_{ho} + \boldsymbol{b}_o),
\end{aligned}
$$

其中的$\boldsymbol{W}_{xi}, \boldsymbol{W}_{xf}, \boldsymbol{W}_{xo} \in \mathbb{R}^{d \times h}$和$\boldsymbol{W}_{hi}, \boldsymbol{W}_{hf}, \boldsymbol{W}_{ho} \in \mathbb{R}^{h \times h}$是权重参数，$\boldsymbol{b}_i, \boldsymbol{b}_f, \boldsymbol{b}_o \in \mathbb{R}^{1 \times h}$是偏差参数。


### 候选记忆细胞

接下来，长短期记忆需要计算候选记忆细胞$\tilde{\boldsymbol{C}}_t$。它的计算与上面介绍的3个门类似，但使用了值域在$[-1, 1]$的tanh函数作为激活函数，如图6.8所示。

![长短期记忆中候选记忆细胞的计算](../img/lstm_1.svg)


具体来说，时间步$t$的候选记忆细胞$\tilde{\boldsymbol{C}}_t \in \mathbb{R}^{n \times h}$的计算为

$$\tilde{\boldsymbol{C}}_t = \text{tanh}(\boldsymbol{X}_t \boldsymbol{W}_{xc} + \boldsymbol{H}_{t-1} \boldsymbol{W}_{hc} + \boldsymbol{b}_c),$$

其中$\boldsymbol{W}_{xc} \in \mathbb{R}^{d \times h}$和$\boldsymbol{W}_{hc} \in \mathbb{R}^{h \times h}$是权重参数，$\boldsymbol{b}_c \in \mathbb{R}^{1 \times h}$是偏差参数。


### 记忆细胞

我们可以通过元素值域在$[0, 1]$的输入门、遗忘门和输出门来控制隐藏状态中信息的流动，这一般也是通过使用按元素乘法（符号为$\odot$）来实现的。当前时间步记忆细胞$\boldsymbol{C}_t \in \mathbb{R}^{n \times h}$的计算组合了上一时间步记忆细胞和当前时间步候选记忆细胞的信息，并通过遗忘门和输入门来控制信息的流动：

$$\boldsymbol{C}_t = \boldsymbol{F}_t \odot \boldsymbol{C}_{t-1} + \boldsymbol{I}_t \odot \tilde{\boldsymbol{C}}_t.$$


如图6.9所示，遗忘门控制上一时间步的记忆细胞$\boldsymbol{C}_{t-1}$中的信息是否传递到当前时间步，而输入门则控制当前时间步的输入$\boldsymbol{X}_t$通过候选记忆细胞$\tilde{\boldsymbol{C}}_t$如何流入当前时间步的记忆细胞。如果遗忘门一直近似1且输入门一直近似0，过去的记忆细胞将一直通过时间保存并传递至当前时间步。这个设计可以应对循环神经网络中的梯度衰减问题，并更好地捕捉时间序列中时间步距离较大的依赖关系。

![长短期记忆中记忆细胞的计算。这里的$\odot$是按元素乘法](../img/lstm_2.svg)


### 隐藏状态

有了记忆细胞以后，接下来我们还可以通过输出门来控制从记忆细胞到隐藏状态$\boldsymbol{H}_t \in \mathbb{R}^{n \times h}$的信息的流动：

$$\boldsymbol{H}_t = \boldsymbol{O}_t \odot \text{tanh}(\boldsymbol{C}_t).$$

这里的tanh函数确保隐藏状态元素值在-1到1之间。需要注意的是，当输出门近似1时，记忆细胞信息将传递到隐藏状态供输出层使用；当输出门近似0时，记忆细胞信息只自己保留。图6.10展示了长短期记忆中隐藏状态的计算。

![长短期记忆中隐藏状态的计算。这里的$\odot$是按元素乘法](../img/lstm_3.svg)


## 读取数据集

下面我们开始实现并展示长短期记忆。和前几节中的实验一样，这里依然使用周杰伦歌词数据集来训练模型作词。

```{.python .input  n=1}
import d2lzh as d2l
import torch
import torch.nn as nn
```

```{.python .input  n=2}
(corpus_indices, char_to_idx, idx_to_char,
 vocab_size) = d2l.load_data_jay_lyrics()
```

## 从零开始实现

我们先介绍如何从零开始实现长短期记忆。

### 初始化模型参数

下面的代码对模型参数进行初始化。超参数`num_hiddens`定义了隐藏单元的个数。

```{.python .input  n=3}
num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
device = d2l.try_gpu()

def get_params():
    def _one(shape):
        return torch.randn(size=shape, dtype=torch.float32, device=device).normal_(std=0.01)

    def _three():
        return (_one((num_inputs, num_hiddens)),
                _one((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))

    W_xi, W_hi, b_i = _three()  # 输入门参数
    W_xf, W_hf, b_f = _three()  # 遗忘门参数
    W_xo, W_ho, b_o = _three()  # 输出门参数
    W_xc, W_hc, b_c = _three()  # 候选记忆细胞参数
    # 输出层参数
    W_hq = _one((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # 附上梯度
    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc,
              b_c, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params
```

## 定义模型

在初始化函数中，长短期记忆的隐藏状态需要返回额外的形状为(批量大小, 隐藏单元个数)的值为0的记忆细胞。

```{.python .input  n=4}
def init_lstm_state(batch_size, num_hiddens, device):
    return (torch.zeros(size=(batch_size, num_hiddens), device=device),
            torch.zeros(size=(batch_size, num_hiddens), device=device))
```

下面根据长短期记忆的计算表达式定义模型。需要注意的是，只有隐藏状态会传递到输出层，而记忆细胞不参与输出层的计算。

```{.python .input  n=5}
def lstm(inputs, state, params):
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c,
     W_hq, b_q] = params
    (H, C) = state
    outputs = []
    for X in inputs:
        Ig = torch.sigmoid(torch.mm(X.float(), W_xi) + torch.mm(H.float(), W_hi) + b_i)
        Fg = torch.sigmoid(torch.mm(X.float(), W_xf) + torch.mm(H.float(), W_hf) + b_f)
        Og = torch.sigmoid(torch.mm(X.float(), W_xo) + torch.mm(H.float(), W_ho) + b_o)
        C_dilta = torch.tanh(torch.mm(X.float(), W_xc) + torch.mm(H.float(), W_hc) + b_c)
        C = Fg * C + Ig * C_dilta
        H = Og * torch.tanh(C)
        Y = torch.mm(H.float(), W_hq) + b_q
        outputs.append(Y)
    return outputs, (H, C)
```

### 训练模型并创作歌词

同上一节一样，我们在训练模型时只使用相邻采样。设置好超参数后，我们将训练模型并根据前缀“分开”和“不分开”分别创作长度为50个字符的一段歌词。

```{.python .input  n=24}
num_epochs, num_steps, batch_size, lr, clipping_theta = 160, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 40, 50, ['分开', '不分开']
```

我们每过40个迭代周期便根据当前训练的模型创作一段歌词。

```{.python .input  n=25}
d2l.train_and_predict_rnn(lstm, get_params, init_lstm_state, num_hiddens,
                          vocab_size, device, corpus_indices, idx_to_char,
                          char_to_idx, False, num_epochs, num_steps, lr,
                          clipping_theta, batch_size, pred_period, pred_len,
                          prefixes)
```

```{.json .output n=25}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "epoch 40, perplexity 88.930481, time 2.04 sec\n - \u5206\u5f00 \u6211\u7528\u4f60\u7684\u624b \u4f60\u8bf4\u7684\u611f\u6cea \u6211\u7528\u7231\u4f60\u4e0d\u5f00 \u4f60\u8bf4\u7684\u7231\u6211 \u6211\u5728\u4f60\u7684\u4e16\u754c \u4f60\u7684\u7b11\u7b11 \u6211\u7684\u4e16\u754c \u4f60\u7684\u611f\u7b11 \u4f60\u7684\n - \u4e0d\u5206\u5f00 \u4f60\u8bf4\u4f60\u7684\u624b \u4f60\u8bf4\u7684\u611f\u5fc6 \u6211\u77e5\u9053\u4f60\u4e0d\u89c1 \u4f60\u7684\u7b11\u7b11 \u4f60\u8bf4\u6211\u7684\u611f\u7b11 \u6211\u77e5\u9053\u4f60\u4e0d\u89c1 \u4f60\u7684\u7b11\u7b11 \u4f60\u8bf4\u6211\u7684\u611f\u7b11\nepoch 80, perplexity 16.069202, time 2.04 sec\n - \u5206\u5f00 \u4e0d\u7528\u9ebb\u70e6\u4e86\u6211\u53bb\u7684\u4e16\u754c \u6ca1\u60f3\u4e86\u4f60\u7684\u624b \u6211\u5728\u7b49\u7740\u4e00\u53e3 \u8fd9\u6837\u7684\u6837\u89c9 \u6211\u5728\u7b49\u5f85\u5f88\u4eba \u6211\u542c\u4f60\u7684\u4f60 \u5c06\u4e00\u79cd\u5473\u9053\n - \u4e0d\u5206\u5f00 \u6ca1\u6709\u4f60\u53bb \u6211\u4f1a\u662f                                   \u4f60\u7684\u4f60\u7684\u9214\u7968\nepoch 120, perplexity 6.373573, time 2.82 sec\n - \u5206\u5f00 \u4e0d\u8981\u4f60 \u4e00\u9996\u597d \u4f60\u597d\u597d\u60f3\u5230\u4f60 \u624b\u5feb\u7684\u5feb\u5019 \u591a\u591a\u6211\u4eec\u592a\u4e16\u754c \u6ca1\u6709\u4f60\u7684\u624b\u9ed8 \u4e0d\u7528\u5979 \u5728\u89d2\u5730\u95f4\u4e0b\u4e45 \u5728\u4eba\n - \u4e0d\u5206\u5f00 \u5c31\u7b97\u4f60\u966a\u6211\u662f\u591a\u4eba\u4e50\u7f8e\u4e86\u773c\u6cea \u5927\u7740\u4f60\u7684\u4e16\u754c \u8868\u4f60\u53d8\u6210\u7761\u6c14 \u6211\u60f3\u4e0d\u61c2\u4f60\u4e00\u9996\u89e3 \u4e5f\u662f\u5f00\u4e86\u6ca1\u4eba\u6709\u5f97\u4f60\u6211\u7684\u611f\nepoch 160, perplexity 3.707634, time 2.22 sec\n - \u5206\u5f00\u59cb\u4e0d\u4f1a \u8ba9\u6211\u4eec\u7ecf\u4e86\u5b64\u7f8e   \u6211\u5f8c\u62f3\u5176\u5176\u5f00\u6211\u7684\u613f\u5802 \u5c31\u7b97\u6211\u542c\u51fa\u4e0a \u56e0\u4e3a\u4f60\u4f1a\u4e0d\u4f1a\u6700\u7f8e \u6211\u4e0d\u80fd\u5c31\u8fd9\u6837\u6ca1\u6709\u89e3\n - \u4e0d\u5206\u5f00 \u5c31\u662f\u4f60\u8fc7\u53bb\u7684\u6e29\u5ea6 \u662f\u6211\u5728\u7b49\u4e0a\u4e0a\u4e0a\u4e0a \u522b\u4eba\u7684\u68a6\u522b\u4eba\u592a\u591a \u4ed6\u662f\u6d77\u771f\u7684\u8baf\u53f7 \u800c\u6211\u7684\u4e4e\u60d1 \u4f60\u8bf4\u7684\u79bb\u79bb \u6211\u5728\n"
 }
]
```

## 简洁实现

在pytorch中我们可以直接调用`nn`模块中的`LSTM`类。此处因为lstm缓存了Memory Cell和Hidden state, 所以设置num_layers=2。此处为沿袭原教程说法，虽然用num_layers容易使人混淆。

```{.python .input  n=6}
num_epochs, num_steps, batch_size, lr, clipping_theta = 1000, 35, 32, 1e-3, 1e-2
pred_period, pred_len, prefixes = 40, 50, ['分开', '不分开']
```

```{.python .input  n=7}
import time
lstm_layer = nn.LSTM(input_size=num_inputs, hidden_size=num_hiddens)
model = d2l.RNNModel(lstm_layer, num_hiddens, vocab_size)
model.to(device)
d2l.train_and_predict_rnn_nn(model, num_hiddens, vocab_size, device,
                                corpus_indices, idx_to_char, char_to_idx,
                                num_epochs, num_steps, lr, clipping_theta,
                                batch_size, pred_period, pred_len, prefixes, num_layers=2)
```

```{.json .output n=7}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "epoch 250, perplexity 80.796695, time 1211.80 sec\nepoch 500, perplexity 5.247390, time 1214.42 sec\n - \u5206\u5f00\u4e86\u5e74\u7684\u660e \u6211\u7684\u4e00\u4e2a \u6211\u4eec\u7684\u5728 \u6211\u4eec\u7684\u773c\u6cea \u6211\u4e0d\u4f1a\u6709\u4e00\u4e2a \u6211\u4e0d\u4f1a \u6211\u4e0d\u4f1a \u6211\u4e0d\u4f1a \u6211\u4e0d\u4f1a \u6211\u4e0d\u4f1a \u6211\u4e0d\n - \u4e0d\u5206\u5f00\u4e86 \u4e00\u4e2a\u73b0 \u6211\u7684\u5728\u6709 \u6211\u7684\u5728\u4e00\u5929 \u6211\u4eec\u7684\u773c\u6cea \u6211\u4e0d\u4f1a\u6709\u4e00\u4e2a \u6211\u4e0d\u4f1a \u6211\u4e0d\u4f1a \u6211\u4e0d\u4f1a \u6211\u4e0d\u4f1a \u6211\u4e0d\u4f1a \nepoch 750, perplexity 1.035023, time 1210.53 sec\nepoch 1000, perplexity 1.006592, time 1213.82 sec\n - \u5206\u5f00 \u4e0d\u61c2\u8bf4\u4ec0\u4e48\u5f00 \u5929\u5802\u8fd8\u662f\u4f1a \u4f60\u53eb\u4ed6 \u8bf4\u4e86\u4e00\u4e2a \u5168\u9762\u7684\u89e3 \u4e00\u80a1\u81ea\u4fe1\u7684\u9a84\u50b2\u6211\u770b\u5f97\u5230 \u5979\u7c89\u5ae9\u6e05\u79c0\u7684\u5916\u8868 \u50cf\n - \u4e0d\u5206\u5f00 \u8c6a\u6c14\u6325\u6b63\u6977\u7ed9\u4e00\u62f3\u5bf9\u767d \u7ed3\u5c40\u5e73\u8eba\u4e0b\u6765\u770b\u8c01\u5389\u5bb3 \u54fc \u2026\u2026 \u8e72\u5c0f\u50f5\u5c38\u8e72\u5c0f\u50f5\u5c38\u8e72 \u53c8\u8e72\u5c0f\u50f5\u5c38\u8e72\u6697\u5df7\u70b9\u706f \u53c8\n"
 }
]
```

## 小结

* 长短期记忆的隐藏层输出包括隐藏状态和记忆细胞。只有隐藏状态会传递到输出层。
* 长短期记忆的输入门、遗忘门和输出门可以控制信息的流动。
* 长短期记忆可以应对循环神经网络中的梯度衰减问题，并更好地捕捉时间序列中时间步距离较大的依赖关系。



## 练习

* 调节超参数，观察并分析对运行时间、困惑度以及创作歌词的结果造成的影响。
* 在相同条件下，比较长短期记忆、门控循环单元和不带门控的循环神经网络的运行时间。
* 既然候选记忆细胞已通过使用tanh函数确保值域在-1到1之间，为什么隐藏状态还需要再次使用tanh函数来确保输出值域在-1到1之间？


## 参考文献

[1] Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural computation, 9(8), 1735-1780.

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/4049)

![](../img/qr_lstm.svg)
