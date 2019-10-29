# 求近义词和类比词

在[“word2vec的实现”](./word2vec-gluon.md)一节中，我们在小规模数据集上训练了一个word2vec词嵌入模型，并通过词向量的余弦相似度搜索近义词。实际中，在大规模语料上预训练的词向量常常可以应用到下游自然语言处理任务中。本节将演示如何用这些预训练的词向量来求近义词和类比词。我们还将在后面两节中继续应用预训练的词向量。

## 使用预训练的词向量

Pytorch的`torchtext`包提供了跟自然语言处理相关的函数和类。下面查看它目前提供的预训练词嵌入的名称。

```{.python .input  n=10}
import torch
import torchtext
from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe
import numpy as np
```

给定词嵌入名称，可以查看该词嵌入提供了哪些预训练的模型。每个模型的词向量维度可能不同，或是在不同数据集上预训练得到的。

预训练的GloVe模型的命名规范大致是“模型.（数据集.）数据集词数.词向量维度.txt”。更多信息可以参考GloVe和fastText的项目网站 [2,3]。下面我们使用基于维基百科子集预训练的50维GloVe词向量。第一次创建预训练词向量实例时会自动下载相应的词向量，因此需要联网。

```{.python .input  n=16}
glove_6b50d = torchtext.vocab.GloVe(name='6B', dim=50)
```

打印词典大小。其中含有40万个词和1个特殊的未知词符号。

```{.python .input  n=27}
len(glove_6b50d)
```

```{.json .output n=27}
[
 {
  "data": {
   "text/plain": "400000"
  },
  "execution_count": 27,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=32}
print(glove_6b50d.vectors.shape)
```

```{.json .output n=32}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "torch.Size([400000, 50])\n"
 }
]
```

我们可以通过词来获取它在词典中的索引，也可以通过索引获取词。

```{.python .input  n=26}
glove_6b50d.stoi['beautiful'], glove_6b50d.itos[3366], glove_6b50d.vectors[3366],glove_6b50d.get_vecs_by_tokens('beautiful', lower_case_backup=True)
```

```{.json .output n=26}
[
 {
  "data": {
   "text/plain": "(3366,\n 'beautiful',\n tensor([ 0.5462,  1.2042, -1.1288, -0.1325,  0.9553,  0.0405, -0.4786, -0.3397,\n         -0.2806,  0.7176, -0.5369, -0.0046,  0.7322,  0.1210,  0.2809, -0.0881,\n          0.5973,  0.5526,  0.0566, -0.5025, -0.6320,  1.1439, -0.3105,  0.1263,\n          1.3155, -0.5244, -1.5041,  1.1580,  0.6880, -0.8505,  2.3236, -0.4179,\n          0.4452, -0.0192,  0.2897,  0.5326, -0.0230,  0.5896, -0.7240, -0.8522,\n         -0.1776,  0.1443,  0.4066, -0.5200,  0.0908,  0.0830, -0.0220, -1.6214,\n          0.3458, -0.0109]),\n tensor([ 0.5462,  1.2042, -1.1288, -0.1325,  0.9553,  0.0405, -0.4786, -0.3397,\n         -0.2806,  0.7176, -0.5369, -0.0046,  0.7322,  0.1210,  0.2809, -0.0881,\n          0.5973,  0.5526,  0.0566, -0.5025, -0.6320,  1.1439, -0.3105,  0.1263,\n          1.3155, -0.5244, -1.5041,  1.1580,  0.6880, -0.8505,  2.3236, -0.4179,\n          0.4452, -0.0192,  0.2897,  0.5326, -0.0230,  0.5896, -0.7240, -0.8522,\n         -0.1776,  0.1443,  0.4066, -0.5200,  0.0908,  0.0830, -0.0220, -1.6214,\n          0.3458, -0.0109]))"
  },
  "execution_count": 26,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## 应用预训练词向量

下面我们以GloVe模型为例，展示预训练词向量的应用。

### 求近义词

这里重新实现[“word2vec的实现”](./word2vec-gluon.md)一节中介绍过的使用余弦相似度来搜索近义词的算法。为了在求类比词时重用其中的求$k$近邻（$k$-nearest neighbors）的逻辑，我们将这部分逻辑单独封装在`knn`函数中。

```{.python .input  n=44}
def knn(W, x, k):
    # 添加的1e-9是为了数值稳定性
    cos = torch.matmul(W, x.reshape((-1,))) / (
        (torch.sum(W * W, dim=1) + 1e-9).sqrt() * torch.sum(x * x).sqrt())
    topk = torch.topk(cos, k=k).indices.cpu().numpy().astype('int32')
    return topk, [cos[i].item() for i in topk]
```

然后，我们通过预训练词向量实例`embed`来搜索近义词。

```{.python .input  n=45}
def get_similar_tokens(query_token, k, embed):
    topk, cos = knn(embed.vectors,
                    embed.get_vecs_by_tokens([query_token]), k+1)
    for i, c in zip(topk[1:], cos[1:]):  # 除去输入词
        print('cosine sim=%.3f: %s' % (c, (embed.itos[i])))
```

已创建的预训练词向量实例`glove_6b50d`的词典中含40万个词和1个特殊的未知词。除去输入词和未知词，我们从中搜索与“chip”语义最相近的3个词。

```{.python .input  n=46}
get_similar_tokens('chip', 3, glove_6b50d)
```

```{.json .output n=46}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "cosine sim=0.856: chips\ncosine sim=0.749: intel\ncosine sim=0.749: electronics\n"
 }
]
```

接下来查找“baby”和“beautiful”的近义词。

```{.python .input  n=47}
get_similar_tokens('baby', 3, glove_6b50d)
```

```{.json .output n=47}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "cosine sim=0.839: babies\ncosine sim=0.800: boy\ncosine sim=0.792: girl\n"
 }
]
```

```{.python .input  n=48}
get_similar_tokens('beautiful', 3, glove_6b50d)
```

```{.json .output n=48}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "cosine sim=0.921: lovely\ncosine sim=0.893: gorgeous\ncosine sim=0.830: wonderful\n"
 }
]
```

### 求类比词

除了求近义词以外，我们还可以使用预训练词向量求词与词之间的类比关系。例如，“man”（男人）: “woman”（女人）:: “son”（儿子） : “daughter”（女儿）是一个类比例子：“man”之于“woman”相当于“son”之于“daughter”。求类比词问题可以定义为：对于类比关系中的4个词 $a : b :: c : d$，给定前3个词$a$、$b$和$c$，求$d$。设词$w$的词向量为$\text{vec}(w)$。求类比词的思路是，搜索与$\text{vec}(c)+\text{vec}(b)-\text{vec}(a)$的结果向量最相似的词向量。

```{.python .input  n=49}
def get_analogy(token_a, token_b, token_c, embed):
    vecs = embed.get_vecs_by_tokens([token_a, token_b, token_c])
    x = vecs[1] - vecs[0] + vecs[2]
    topk, cos = knn(embed.vectors, x, 1)
    return embed.itos[topk[0]]
```

验证一下“男-女”类比。

```{.python .input  n=50}
get_analogy('man', 'woman', 'son', glove_6b50d)
```

```{.json .output n=50}
[
 {
  "data": {
   "text/plain": "'daughter'"
  },
  "execution_count": 50,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

“首都-国家”类比：“beijing”（北京）之于“china”（中国）相当于“tokyo”（东京）之于什么？答案应该是“japan”（日本）。

```{.python .input  n=51}
get_analogy('beijing', 'china', 'tokyo', glove_6b50d)
```

```{.json .output n=51}
[
 {
  "data": {
   "text/plain": "'japan'"
  },
  "execution_count": 51,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

“形容词-形容词最高级”类比：“bad”（坏的）之于“worst”（最坏的）相当于“big”（大的）之于什么？答案应该是“biggest”（最大的）。

```{.python .input  n=52}
get_analogy('bad', 'worst', 'big', glove_6b50d)
```

```{.json .output n=52}
[
 {
  "data": {
   "text/plain": "'biggest'"
  },
  "execution_count": 52,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

“动词一般时-动词过去时”类比：“do”（做）之于“did”（做过）相当于“go”（去）之于什么？答案应该是“went”（去过）。

```{.python .input  n=53}
get_analogy('do', 'did', 'go', glove_6b50d)
```

```{.json .output n=53}
[
 {
  "data": {
   "text/plain": "'went'"
  },
  "execution_count": 53,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## 小结

* 在大规模语料上预训练的词向量常常可以应用于下游自然语言处理任务中。
* 可以应用预训练的词向量求近义词和类比词。


## 练习

* 测试一下fastText的结果。值得一提的是，fastText有预训练的中文词向量（`pretrained_file_name='wiki.zh.vec'`）。
* 如果词典特别大，如何提升近义词或类比词的搜索速度？




## 参考文献

[1] GluonNLP工具包。 https://gluon-nlp.mxnet.io/

[2] GloVe项目网站。 https://nlp.stanford.edu/projects/glove/

[3] fastText项目网站。 https://fasttext.cc/

## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/4373)

![](../img/qr_similarity-analogy.svg)
