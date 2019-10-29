# 文本情感分类：使用循环神经网络

文本分类是自然语言处理的一个常见任务，它把一段不定长的文本序列变换为文本的类别。本节关注它的一个子问题：使用文本情感分类来分析文本作者的情绪。这个问题也叫情感分析，并有着广泛的应用。例如，我们可以分析用户对产品的评论并统计用户的满意度，或者分析用户对市场行情的情绪并用以预测接下来的行情。

同搜索近义词和类比词一样，文本分类也属于词嵌入的下游应用。在本节中，我们将应用预训练的词向量和含多个隐藏层的双向循环神经网络，来判断一段不定长的文本序列中包含的是正面还是负面的情绪。

在实验开始前，导入所需的包或模块。

```{.python .input  n=1}
import collections
import d2lzh as d2l
import torch
import os
import random
import tarfile
import requests
import torchtext
import torch.nn as nn
from io import BytesIO
```

## 文本情感分类数据

我们使用斯坦福的IMDb数据集（Stanford's Large Movie Review Dataset）作为文本情感分类的数据集 [1]。这个数据集分为训练和测试用的两个数据集，分别包含25,000条从IMDb下载的关于电影的评论。在每个数据集中，标签为“正面”和“负面”的评论数量相等。

###  读取数据

首先下载这个数据集到`../data`路径下，然后解压至`../data/aclImdb`下。

```{.python .input  n=2}
# 本函数已保存在d2lzh包中方便以后使用
def download_imdb(data_dir='../data'):
    url = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
    sha1 = '01ada507287d82875905620988597833ad4e0903'
    print("Downloading fra-eng.zip from '{0}'".format(url))


    headers={"User-Agent": "XY"}#dummy user agent 
    response = requests.get(url,headers=headers ,stream=True)
    filename = os.path.join(data_dir, 'aclImdb_v1.tar.gz')
    
    with open(filename, 'wb') as handle:
        for chunk in response.iter_content(chunk_size=512):
            if chunk:  # filter out keep-alive new chunks
                handle.write(chunk)

    with tarfile.open(filename, 'r') as f:
        f.extractall(data_dir)

download_imdb()
```

```{.json .output n=2}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Downloading fra-eng.zip from 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'\n"
 }
]
```

接下来，读取训练数据集和测试数据集。每个样本是一条评论及其对应的标签：1表示“正面”，0表示“负面”。

```{.python .input  n=3}
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
train_data[0]
```

```{.json .output n=3}
[
 {
  "data": {
   "text/plain": "[\"sad story of a downed b-17 pilot. brady is shot down over occupied territory. the local ranchers extended him kindness and protection at the cost of their own lives. i had never heard of this movie and it snagged me for two hours. after the film is over, i'm glad i took the time. it's an entire story told to explain the look on brady's face at the start of the film.\",\n 1]"
  },
  "execution_count": 3,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

### 预处理数据

我们需要对每条评论做分词，从而得到分好词的评论。这里定义的`get_tokenized_imdb`函数使用最简单的方法：基于空格进行分词。

```{.python .input  n=4}
def get_tokenized_imdb(data):  # 本函数已保存在d2lzh包中方便以后使用
    def tokenizer(text):
        return [tok.lower() for tok in text.split(' ')]
    return [tokenizer(review) for review, _ in data]
```

现在，我们可以根据分好词的训练数据集来创建词典了。我们在这里过滤掉了出现次数少于5的词。

```{.python .input  n=5}
def get_vocab_imdb(data):  # 本函数已保存在d2lzh包中方便以后使用
    tokenized_data = get_tokenized_imdb(data)
    tokens = [word for sentence in tokenized_data for word in sentence]
    return d2l.Vocab(tokens, min_freq=5)

vocab = get_vocab_imdb(train_data)
'# words in vocab:', len(vocab)
```

```{.json .output n=5}
[
 {
  "data": {
   "text/plain": "('# words in vocab:', 46151)"
  },
  "execution_count": 5,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

因为每条评论长度不一致所以不能直接组合成小批量，我们定义`preprocess_imdb`函数对每条评论进行分词，并通过词典转换成词索引，然后通过截断或者补0来将每条评论长度固定成500。

```{.python .input  n=6}
def preprocess_imdb(data, vocab):  # 本函数已保存在d2lzh包中方便以后使用
    max_l = 500  # 将每条评论通过截断或者补0，使得长度变成500

    def pad(x):
        return x[:max_l] if len(x) > max_l else x + [0] * (max_l - len(x))

    tokenized_data = get_tokenized_imdb(data)
    features = torch.tensor([pad(vocab[x]) for x in tokenized_data])
    labels = torch.tensor([score for _, score in data])
    return features, labels
```

### 创建数据迭代器

现在，我们创建数据迭代器。每次迭代将返回一个小批量的数据。

```{.python .input  n=7}
batch_size = 64
train_set = torch.utils.data.TensorDataset(*preprocess_imdb(train_data, vocab))
test_set = torch.utils.data.TensorDataset(*preprocess_imdb(test_data, vocab))
train_iter = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True)
test_iter = torch.utils.data.DataLoader(test_set, batch_size)
```

打印第一个小批量数据的形状以及训练集中小批量的个数。

```{.python .input  n=8}
for X, y in train_iter:
    print('X', X.shape, 'y', y.shape)
    break
'#batches:', len(train_iter)
```

```{.json .output n=8}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "X torch.Size([64, 500]) y torch.Size([64])\n"
 },
 {
  "data": {
   "text/plain": "('#batches:', 391)"
  },
  "execution_count": 8,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## 使用循环神经网络的模型

在这个模型中，每个词先通过嵌入层得到特征向量。然后，我们使用双向循环神经网络对特征序列进一步编码得到序列信息。最后，我们将编码的序列信息通过全连接层变换为输出。具体来说，我们可以将双向长短期记忆在最初时间步和最终时间步的隐藏状态连结，作为特征序列的表征传递给输出层分类。在下面实现的`BiRNN`类中，`Embedding`实例即嵌入层，`LSTM`实例即为序列编码的隐藏层，`Dense`实例即生成分类结果的输出层。

```{.python .input  n=92}
class BiRNN(nn.Module):
    def __init__(self, vocab, embed_size, num_hiddens, num_layers, **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_direcitions = 2
        self.num_layers = num_layers
        self.embedding = nn.Embedding(len(vocab), embed_size)
        # bidirectional设为True即得到双向循环神经网络
        self.encoder = nn.LSTM(embed_size, num_hiddens, num_layers=num_layers,
                                bidirectional=True)
        self.decoder = nn.Linear(self.num_direcitions * 2 * num_hiddens, 2)

    def forward(self, inputs):
        # inputs的形状是(批量大小, 词数)，因为LSTM需要将序列作为第一维，所以将输入转置后
        # 再提取词特征，输出形状为(词数, 批量大小, 词向量维度)
#         print('inputs.shape', inputs.shape)
        embeddings = self.embedding(inputs.t())
#         print('embeddings.shape', embeddings.shape)

        # rnn.LSTM只传入输入embeddings，因此只返回最后一层的隐藏层在各时间步的隐藏状态。
        # outputs形状是(词数, 批量大小, 2 * 隐藏单元个数)
        outputs, (h_n, c_n) = self.encoder(embeddings)
#         print('outputs.shape', outputs.shape)
#         print('h_n.shape', h_n.shape)
#         print('c_n.shape', c_n.shape)
        # 连结初始时间步和最终时间步的隐藏状态作为全连接层输入。它的形状为
        # (批量大小, 4 * 隐藏单元个数)。
        encoding = torch.cat([outputs[0], outputs[-1]], dim=1)
#         print('encoding.shape', encoding.shape)
        outs = self.decoder(encoding)
        return outs
```

创建一个含两个隐藏层的双向循环神经网络。

```{.python .input  n=93}
embed_size, num_hiddens, num_layers, ctx = 100, 100, 2, d2l.try_gpu()
# embed_size, num_hiddens, num_layers, ctx = 100, 100, 2, 'cpu'
net = BiRNN(vocab, embed_size, num_hiddens, num_layers)
```

### 加载预训练的词向量

由于情感分类的训练数据集并不是很大，为应对过拟合，我们将直接使用在更大规模语料上预训练的词向量作为每个词的特征向量。这里，我们为词典`vocab`中的每个词加载100维的GloVe词向量。

```{.python .input  n=94}
glove_6b50d = torchtext.vocab.GloVe(name='6B', dim=100)
glove_embedding = glove_6b50d.get_vecs_by_tokens(vocab.idx_to_token)
print(glove_embedding.shape)
```

```{.json .output n=94}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "torch.Size([46151, 100])\n"
 }
]
```

然后，我们将用这些词向量作为评论中每个词的特征向量。注意，预训练词向量的维度需要与创建的模型中的嵌入层输出大小`embed_size`一致。此外，在训练中我们不再更新这些词向量。

```{.python .input  n=95}
net.embedding.weight.data.copy_(glove_embedding)
for param in net.embedding.parameters():
    param.requires_grad_(False)
```

### 训练并评价模型

这时候就可以开始训练模型了。

```{.python .input  n=105}
lr, num_epochs = 0.01, 5
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = torch.nn.CrossEntropyLoss(reduction='sum')
# d2l.train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs)

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

```{.json .output n=105}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "training on cuda:0\nepoch 1, loss 0.4433, train acc 0.802, test acc 0.823, time 29.7 sec\nepoch 2, loss 0.3727, train acc 0.837, test acc 0.846, time 29.3 sec\nepoch 3, loss 0.3308, train acc 0.860, test acc 0.804, time 29.3 sec\nepoch 4, loss 0.3074, train acc 0.873, test acc 0.855, time 30.2 sec\nepoch 5, loss 0.2835, train acc 0.882, test acc 0.843, time 33.0 sec\n"
 }
]
```

最后，定义预测函数。

```{.python .input  n=111}
# 本函数已保存在d2lzh包中方便以后使用
def predict_sentiment(net, vocab, sentence):
    sentence = torch.tensor(vocab[sentence], device=d2l.try_gpu())
    label = torch.argmax(net(sentence.reshape((1, -1))), dim=1)
    return 'positive' if label.item() == 1 else 'negative'
```

下面使用训练好的模型对两个简单句子的情感进行分类。

```{.python .input  n=112}
predict_sentiment(net, vocab, ['this', 'movie', 'is', 'so', 'great'])
```

```{.json .output n=112}
[
 {
  "data": {
   "text/plain": "'positive'"
  },
  "execution_count": 112,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=113}
predict_sentiment(net, vocab, ['this', 'movie', 'is', 'so', 'bad'])
```

```{.json .output n=113}
[
 {
  "data": {
   "text/plain": "'negative'"
  },
  "execution_count": 113,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## 小结

* 文本分类把一段不定长的文本序列变换为文本的类别。它属于词嵌入的下游应用。
* 可以应用预训练的词向量和循环神经网络对文本的情感进行分类。


## 练习

* 增加迭代周期。训练后的模型能在训练和测试数据集上得到怎样的准确率？再调节其他超参数试试？

* 使用更大的预训练词向量，如300维的GloVe词向量，能否提升分类准确率？

* 使用spaCy分词工具，能否提升分类准确率？你需要安装spaCy（`pip install spacy`），并且安装英文包（`python -m spacy download en`）。在代码中，先导入spacy（`import spacy`）。然后加载spacy英文包（`spacy_en = spacy.load('en')`）。最后定义函数`def tokenizer(text): return [tok.text for tok in spacy_en.tokenizer(text)]`并替换原来的基于空格分词的`tokenizer`函数。需要注意的是，GloVe词向量对于名词词组的存储方式是用“-”连接各个单词，例如，词组“new york”在GloVe词向量中的表示为“new-york”，而使用spaCy分词之后“new york”的存储可能是“new york”。






## 参考文献

[1] Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011, June). Learning word vectors for sentiment analysis. In Proceedings of the 49th annual meeting of the association for computational linguistics: Human language technologies-volume 1 (pp. 142-150). Association for Computational Linguistics.

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/6155)

![](../img/qr_sentiment-analysis.svg)

```{.python .input}

```
