# 文本情感分类：使用卷积神经网络（textCNN）

在“卷积神经网络”一章中我们探究了如何使用二维卷积神经网络来处理二维图像数据。在之前的语言模型和文本分类任务中，我们将文本数据看作是只有一个维度的时间序列，并很自然地使用循环神经网络来表征这样的数据。其实，我们也可以将文本当作一维图像，从而可以用一维卷积神经网络来捕捉临近词之间的关联。本节将介绍将卷积神经网络应用到文本分析的开创性工作之一：textCNN [1]。

首先导入实验所需的包和模块。

```{.python .input  n=21}
import d2lzh as d2l
import torch
import os
import random
import torch.nn as nn
import torch.nn.functional as F
import torchtext
```

## 一维卷积层

在介绍模型前我们先来解释一维卷积层的工作原理。与二维卷积层一样，一维卷积层使用一维的互相关运算。在一维互相关运算中，卷积窗口从输入数组的最左方开始，按从左往右的顺序，依次在输入数组上滑动。当卷积窗口滑动到某一位置时，窗口中的输入子数组与核数组按元素相乘并求和，得到输出数组中相应位置的元素。如图10.4所示，输入是一个宽为7的一维数组，核数组的宽为2。可以看到输出的宽度为$7-2+1=6$，且第一个元素是由输入的最左边的宽为2的子数组与核数组按元素相乘后再相加得到的：$0\times1+1\times2=2$。

![一维互相关运算](../img/conv1d.svg)

下面我们将一维互相关运算实现在`corr1d`函数里。它接受输入数组`X`和核数组`K`，并输出数组`Y`。

```{.python .input  n=2}
def corr1d(X, K):
    w = K.shape[0]
    Y = torch.zeros((X.shape[0] - w + 1))
    for i in range(Y.shape[0]):
        Y[i] = (X[i: i + w] * K).sum()
    return Y
```

让我们复现图10.4中一维互相关运算的结果。

```{.python .input  n=3}
X, K = torch.tensor([0, 1, 2, 3, 4, 5, 6]), torch.tensor([1, 2])
corr1d(X, K)
```

```{.json .output n=3}
[
 {
  "data": {
   "text/plain": "tensor([ 2.,  5.,  8., 11., 14., 17.])"
  },
  "execution_count": 3,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

多输入通道的一维互相关运算也与多输入通道的二维互相关运算类似：在每个通道上，将核与相应的输入做一维互相关运算，并将通道之间的结果相加得到输出结果。图10.5展示了含3个输入通道的一维互相关运算，其中阴影部分为第一个输出元素及其计算所使用的输入和核数组元素：$0\times1+1\times2+1\times3+2\times4+2\times(-1)+3\times(-3)=2$。

![含3个输入通道的一维互相关运算](../img/conv1d-channel.svg)

让我们复现图10.5中多输入通道的一维互相关运算的结果。

```{.python .input  n=5}
def corr1d_multi_in(X, K):
    # 首先沿着X和K的第0维（通道维）遍历。然后使用*将结果列表变成add_n函数的位置参数
    #（positional argument）来进行相加
    result = corr1d(X[0], K[0])
    for i in range(1,X.shape[0]):
        result += corr1d(X[i], K[i])
    return result

X = torch.tensor([[0, 1, 2, 3, 4, 5, 6],
              [1, 2, 3, 4, 5, 6, 7],
              [2, 3, 4, 5, 6, 7, 8]])
K = torch.tensor([[1, 2], [3, 4], [-1, -3]])
corr1d_multi_in(X, K)
```

```{.json .output n=5}
[
 {
  "data": {
   "text/plain": "tensor([ 2.,  8., 14., 20., 26., 32.])"
  },
  "execution_count": 5,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

由二维互相关运算的定义可知，多输入通道的一维互相关运算可以看作单输入通道的二维互相关运算。如图10.6所示，我们也可以将图10.5中多输入通道的一维互相关运算以等价的单输入通道的二维互相关运算呈现。这里核的高等于输入的高。图10.6中的阴影部分为第一个输出元素及其计算所使用的输入和核数组元素：$2\times(-1)+3\times(-3)+1\times3+2\times4+0\times1+1\times2=2$。

![单输入通道的二维互相关运算](../img/conv1d-2d.svg)

图10.4和图10.5中的输出都只有一个通道。我们在[“多输入通道和多输出通道”](../chapter_convolutional-neural-networks/channels.md)一节中介绍了如何在二维卷积层中指定多个输出通道。类似地，我们也可以在一维卷积层指定多个输出通道，从而拓展卷积层中的模型参数。


## 时序最大池化层

类似地，我们有一维池化层。textCNN中使用的时序最大池化（max-over-time pooling）层实际上对应一维全局最大池化层：假设输入包含多个通道，各通道由不同时间步上的数值组成，各通道的输出即该通道所有时间步中最大的数值。因此，时序最大池化层的输入在各个通道上的时间步数可以不同。

为提升计算性能，我们常常将不同长度的时序样本组成一个小批量，并通过在较短序列后附加特殊字符（如0）令批量中各时序样本长度相同。这些人为添加的特殊字符当然是无意义的。由于时序最大池化的主要目的是抓取时序中最重要的特征，它通常能使模型不受人为添加字符的影响。


## 读取和预处理IMDb数据集

我们依然使用和上一节中相同的IMDb数据集做情感分析。以下读取和预处理数据集的步骤与上一节中的相同。

```{.python .input  n=10}
def read_imdb(folder='train'):  # 本函数已保存在d2lzh包中方便以后使用
    data = []
    for label in ['pos', 'neg']:
        folder_name = os.path.join('../data/aclImdb/', folder, label)
        for file in os.listdir(folder_name):
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '').lower()
                data.append([review, 1 if label == 'pos' else 0])
    random.shuffle(data)
    return data

train_data, test_data = read_imdb('train'), read_imdb('test')

def get_tokenized_imdb(data):  # 本函数已保存在d2lzh包中方便以后使用
    def tokenizer(text):
        return [tok.lower() for tok in text.split(' ')]
    return [tokenizer(review) for review, _ in data]

def get_vocab_imdb(data):  # 本函数已保存在d2lzh包中方便以后使用
    tokenized_data = get_tokenized_imdb(data)
    tokens = [word for sentence in tokenized_data for word in sentence]
    return d2l.Vocab(tokens, min_freq=5)

vocab = get_vocab_imdb(train_data)

def preprocess_imdb(data, vocab):  # 本函数已保存在d2lzh包中方便以后使用
    max_l = 500  # 将每条评论通过截断或者补0，使得长度变成500

    def pad(x):
        return x[:max_l] if len(x) > max_l else x + [0] * (max_l - len(x))

    tokenized_data = get_tokenized_imdb(data)
    features = torch.tensor([pad(vocab[x]) for x in tokenized_data])
    labels = torch.tensor([score for _, score in data])
    return features, labels

batch_size = 64
train_set = torch.utils.data.TensorDataset(*preprocess_imdb(train_data, vocab))
test_set = torch.utils.data.TensorDataset(*preprocess_imdb(test_data, vocab))
train_iter = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True)
test_iter = torch.utils.data.DataLoader(test_set, batch_size)
```

## textCNN模型

textCNN模型主要使用了一维卷积层和时序最大池化层。假设输入的文本序列由$n$个词组成，每个词用$d$维的词向量表示。那么输入样本的宽为$n$，高为1，输入通道数为$d$。textCNN的计算主要分为以下几步。

1. 定义多个一维卷积核，并使用这些卷积核对输入分别做卷积计算。宽度不同的卷积核可能会捕捉到不同个数的相邻词的相关性。
2. 对输出的所有通道分别做时序最大池化，再将这些通道的池化输出值连结为向量。
3. 通过全连接层将连结后的向量变换为有关各类别的输出。这一步可以使用丢弃层应对过拟合。

图10.7用一个例子解释了textCNN的设计。这里的输入是一个有11个词的句子，每个词用6维词向量表示。因此输入序列的宽为11，输入通道数为6。给定2个一维卷积核，核宽分别为2和4，输出通道数分别设为4和5。因此，一维卷积计算后，4个输出通道的宽为$11-2+1=10$，而其他5个通道的宽为$11-4+1=8$。尽管每个通道的宽不同，我们依然可以对各个通道做时序最大池化，并将9个通道的池化输出连结成一个9维向量。最终，使用全连接将9维向量变换为2维输出，即正面情感和负面情感的预测。

![textCNN的设计](../img/textcnn.svg)

下面我们来实现textCNN模型。与上一节相比，除了用一维卷积层替换循环神经网络外，这里我们还使用了两个嵌入层，一个的权重固定，另一个则参与训练。

```{.python .input  n=43}
class TextCNN(nn.Module):
    def __init__(self, vocab, embed_size, kernel_sizes, num_channels,
                 **kwargs):
        super(TextCNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(len(vocab), embed_size)
        # 不参与训练的嵌入层
        self.constant_embedding = nn.Embedding(len(vocab), embed_size)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Linear(sum(num_channels), 2)
        # 时序最大池化层没有权重，所以可以共用一个实例
        # 创建多个一维卷积层
        self.convs = nn.ModuleList([nn.Conv1d(embed_size * 2, c, k) for c,k in zip(num_channels, kernel_sizes)])

    def forward(self, inputs):
        # 将两个形状是(批量大小, 词数, 词向量维度)的嵌入层的输出按词向量连结
        embeddings = torch.cat([self.embedding(inputs), self.constant_embedding(inputs)], dim=2)
        # 根据Conv1D要求的输入格式，将词向量维，即一维卷积层的通道维，变换到前一维
        embeddings = embeddings.transpose(1,2)
        # 对于每个一维卷积层，在时序最大池化后会得到一个形状为(批量大小, 通道大小)的
        # Tensor
        x = [F.relu(conv(embeddings)) for conv in self.convs] 
        x = [F.max_pool1d(line,line.size(2)).squeeze(2) for line in x]
        x = torch.cat(x,1)
        # 应用丢弃法后使用全连接层得到输出
        outputs = self.decoder(self.dropout(x))
        return outputs
```

创建一个`TextCNN`实例。它有3个卷积层，它们的核宽分别为3、4和5，输出通道数均为100。

```{.python .input  n=44}
embed_size, kernel_sizes, nums_channels = 100, [3, 4, 5], [100, 100, 100]
ctx = d2l.try_gpu()
net = TextCNN(vocab, embed_size, kernel_sizes, nums_channels)
```

### 加载预训练的词向量

同上一节一样，加载预训练的100维GloVe词向量，并分别初始化嵌入层`embedding`和`constant_embedding`，前者参与训练，而后者权重固定。

```{.python .input  n=45}
glove_6b50d = torchtext.vocab.GloVe(name='6B', dim=100)
glove_embedding = glove_6b50d.get_vecs_by_tokens(vocab.idx_to_token)
net.embedding.weight.data.copy_(glove_embedding)
net.constant_embedding.weight.data.copy_(glove_embedding)
for param in net.constant_embedding.parameters():
    param.requires_grad_(False)
```

### 训练并评价模型

现在就可以训练模型了。

```{.python .input  n=46}
lr, num_epochs = 0.001, 5
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = torch.nn.CrossEntropyLoss(reduction='sum')

import time
def train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs):
    print('training on', ctx)
    net.to(ctx)
    for epoch in range(num_epochs):
        net.train()
        train_l_sum, train_acc_sum, n, m, start = 0.0, 0.0, 0, 0, time.time()
        for Xs, ys in train_iter:
            trainer.zero_grad()
            Xs, ys = Xs.to(ctx), ys.to(ctx)
            y_hats = net(Xs)
            l = loss(y_hats, ys)
            l.backward()
            trainer.step()
            train_l_sum += l.item()
            n += Xs.shape[0]
            train_acc_sum += (y_hats.argmax(dim=1) == ys).sum().item()
#             print('train_acc_sum', train_acc_sum)
#             train_l_sum += sum([l.sum().item() for l in ls])
#             n += sum([l.size(0) for l in ls])
#             train_acc_sum += sum([(y_hat.argmax(dim=1) == y).sum().item()
#                                  for y_hat, y in zip(y_hats, ys)])
#             m += sum([y.size(0) for y in ys])
        test_acc = d2l.evaluate_accuracy(test_iter, net, ctx)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, '
              'time %.1f sec'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc,
                 time.time() - start))
        
train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs)
```

```{.json .output n=46}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "training on cuda:0\nepoch 1, loss 0.4874, train acc 0.755, test acc 0.843, time 13.3 sec\nepoch 2, loss 0.3169, train acc 0.864, test acc 0.869, time 13.6 sec\nepoch 3, loss 0.2059, train acc 0.918, test acc 0.880, time 13.3 sec\nepoch 4, loss 0.1172, train acc 0.958, test acc 0.871, time 13.3 sec\nepoch 5, loss 0.0593, train acc 0.980, test acc 0.871, time 13.3 sec\n"
 }
]
```

下面，我们使用训练好的模型对两个简单句子的情感进行分类。

```{.python .input  n=48}
# 本函数已保存在d2lzh包中方便以后使用
def predict_sentiment(net, vocab, sentence):
    sentence = torch.tensor(vocab[sentence], device=d2l.try_gpu())
    label = torch.argmax(net(sentence.reshape((1, -1))), dim=1)
    return 'positive' if label.item() == 1 else 'negative'
```

```{.python .input  n=49}
predict_sentiment(net, vocab, ['this', 'movie', 'is', 'so', 'great'])
```

```{.json .output n=49}
[
 {
  "data": {
   "text/plain": "'positive'"
  },
  "execution_count": 49,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=50}
predict_sentiment(net, vocab, ['this', 'movie', 'is', 'so', 'bad'])
```

```{.json .output n=50}
[
 {
  "data": {
   "text/plain": "'negative'"
  },
  "execution_count": 50,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## 小结

* 可以使用一维卷积来表征时序数据。
* 多输入通道的一维互相关运算可以看作单输入通道的二维互相关运算。
* 时序最大池化层的输入在各个通道上的时间步数可以不同。
* textCNN主要使用了一维卷积层和时序最大池化层。


## 练习

* 动手调参，从准确率和运行效率比较情感分析的两类方法：使用循环神经网络和使用卷积神经网络。
* 使用上一节练习中介绍的3种方法（调节超参数、使用更大的预训练词向量和使用spaCy分词工具），能使模型在测试集上的准确率进一步提高吗？
* 还能将textCNN应用于自然语言处理的哪些任务中？





## 参考文献

[1] Kim, Y. (2014). Convolutional neural networks for sentence classification. arXiv preprint arXiv:1408.5882.

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/7762)

![](../img/qr_sentiment-analysis-cnn.svg)
