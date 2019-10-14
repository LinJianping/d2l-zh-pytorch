# 实战Kaggle比赛：房价预测

作为深度学习基础篇章的总结，我们将对本章内容学以致用。下面，让我们动手实战一个Kaggle比赛：房价预测。本节将提供未经调优的数据的预处理、模型的设计和超参数的选择。我们希望读者通过动手操作、仔细观察实验现象、认真分析实验结果并不断调整方法，得到令自己满意的结果。

## Kaggle比赛

[Kaggle](https://www.kaggle.com)是一个著名的供机器学习爱好者交流的平台。图3.7展示了Kaggle网站的首页。为了便于提交结果，需要注册Kaggle账号。

![Kaggle网站首页](../img/kaggle.png)

我们可以在房价预测比赛的网页上了解比赛信息和参赛者成绩，也可以下载数据集并提交自己的预测结果。该比赛的网页地址是 https://www.kaggle.com/c/house-prices-advanced-regression-techniques 。


图3.8展示了房价预测比赛的网页信息。

![房价预测比赛的网页信息。比赛数据集可通过点击“Data”标签获取](../img/house_pricing.png)

## 获取和读取数据集

比赛数据分为训练数据集和测试数据集。两个数据集都包括每栋房子的特征，如街道类型、建造年份、房顶类型、地下室状况等特征值。这些特征值有连续的数字、离散的标签甚至是缺失值“na”。只有训练数据集包括了每栋房子的价格，也就是标签。我们可以访问比赛网页，点击图3.8中的“Data”标签，并下载这些数据集。

我们将通过`pandas`库读入并处理数据。在导入本节需要的包前请确保已安装`pandas`库，否则请参考下面的代码注释。

```{.python .input  n=36}
# 如果没有安装pandas，则反注释下面一行
# !pip install pandas

%matplotlib inline
import d2lzh as d2l
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
```

解压后的数据位于`../data`目录，它包括两个csv文件。下面使用`pandas`读取这两个文件。

```{.python .input  n=37}
train_data = pd.read_csv('../data/house-prices-advanced-regression-techniques/train.csv')
test_data = pd.read_csv('../data/house-prices-advanced-regression-techniques/test.csv')
```

训练数据集包括1460个样本、80个特征和1个标签。

```{.python .input  n=38}
train_data.shape
```

```{.json .output n=38}
[
 {
  "data": {
   "text/plain": "(1460, 81)"
  },
  "execution_count": 38,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

测试数据集包括1459个样本和80个特征。我们需要将测试数据集中每个样本的标签预测出来。

```{.python .input  n=39}
test_data.shape
```

```{.json .output n=39}
[
 {
  "data": {
   "text/plain": "(1459, 80)"
  },
  "execution_count": 39,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

让我们来查看前4个样本的前4个特征、后2个特征和标签（SalePrice）：

```{.python .input  n=40}
train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]]
```

```{.json .output n=40}
[
 {
  "data": {
   "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>Id</th>\n      <th>MSSubClass</th>\n      <th>MSZoning</th>\n      <th>LotFrontage</th>\n      <th>SaleType</th>\n      <th>SaleCondition</th>\n      <th>SalePrice</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>0</th>\n      <td>1</td>\n      <td>60</td>\n      <td>RL</td>\n      <td>65.0</td>\n      <td>WD</td>\n      <td>Normal</td>\n      <td>208500</td>\n    </tr>\n    <tr>\n      <th>1</th>\n      <td>2</td>\n      <td>20</td>\n      <td>RL</td>\n      <td>80.0</td>\n      <td>WD</td>\n      <td>Normal</td>\n      <td>181500</td>\n    </tr>\n    <tr>\n      <th>2</th>\n      <td>3</td>\n      <td>60</td>\n      <td>RL</td>\n      <td>68.0</td>\n      <td>WD</td>\n      <td>Normal</td>\n      <td>223500</td>\n    </tr>\n    <tr>\n      <th>3</th>\n      <td>4</td>\n      <td>70</td>\n      <td>RL</td>\n      <td>60.0</td>\n      <td>WD</td>\n      <td>Abnorml</td>\n      <td>140000</td>\n    </tr>\n  </tbody>\n</table>\n</div>",
   "text/plain": "   Id  MSSubClass MSZoning  LotFrontage SaleType SaleCondition  SalePrice\n0   1          60       RL         65.0       WD        Normal     208500\n1   2          20       RL         80.0       WD        Normal     181500\n2   3          60       RL         68.0       WD        Normal     223500\n3   4          70       RL         60.0       WD       Abnorml     140000"
  },
  "execution_count": 40,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

可以看到第一个特征是Id，它能帮助模型记住每个训练样本，但难以推广到测试样本，所以我们不使用它来训练。我们将所有的训练数据和测试数据的79个特征按样本连结。

```{.python .input  n=41}
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
```

## 预处理数据

我们对连续数值的特征做标准化（standardization）：设该特征在整个数据集上的均值为$\mu$，标准差为$\sigma$。那么，我们可以将该特征的每个值先减去$\mu$再除以$\sigma$得到标准化后的每个特征值。对于缺失的特征值，我们将其替换成该特征的均值。

```{.python .input  n=42}
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
# 标准化后，每个特征的均值变为0，所以可以直接用0来替换缺失值
all_features[numeric_features] = all_features[numeric_features].fillna(0)
```

接下来将离散数值转成指示特征。举个例子，假设特征MSZoning里面有两个不同的离散值RL和RM，那么这一步转换将去掉MSZoning特征，并新加两个特征MSZoning\_RL和MSZoning\_RM，其值为0或1。如果一个样本原来在MSZoning里的值为RL，那么有MSZoning\_RL=1且MSZoning\_RM=0。

```{.python .input  n=43}
# dummy_na=True将缺失值也当作合法的特征值并为其创建指示特征
all_features = pd.get_dummies(all_features, dummy_na=True)
all_features.shape
```

```{.json .output n=43}
[
 {
  "data": {
   "text/plain": "(2919, 331)"
  },
  "execution_count": 43,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

可以看到这一步转换将特征数从79增加到了331。

最后，通过`values`属性得到NumPy格式的数据，并转成`Tensor`方便后面的训练。

```{.python .input  n=44}
n_train = train_data.shape[0]
train_features = torch.FloatTensor(all_features[:n_train].values.astype(np.float32))
test_features = torch.FloatTensor(all_features[n_train:].values.astype(np.float32))
train_labels = torch.FloatTensor(train_data.SalePrice.values.astype(np.float32)).reshape((-1, 1))
```

## 训练模型

我们使用一个基本的线性回归模型和平方损失函数来训练模型。

```{.python .input  n=45}
loss = nn.MSELoss()
in_features = train_features.shape[1]
def get_net():
    net = nn.Sequential(nn.Linear(in_features, 1))
    return net
```

下面定义比赛用来评价模型的对数均方根误差。给定预测值$\hat y_1, \ldots, \hat y_n$和对应的真实标签$y_1,\ldots, y_n$，它的定义为

$$\sqrt{\frac{1}{n}\sum_{i=1}^n\left(\log(y_i)-\log(\hat y_i)\right)^2}.$$

对数均方根误差的实现如下。

```{.python .input  n=64}
def log_rmse(net,features,labels):
    # To further stabilize the value when the logarithm is taken, set the
    # value less than 1 as 1
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt((loss(torch.log(clipped_preds),torch.log(labels))))
    return rmse.item()
```

下面的训练函数跟本章中前几节的不同在于使用了Adam优化算法。相对之前使用的小批量随机梯度下降，它对学习率相对不那么敏感。我们将在之后的“优化算法”一章里详细介绍它。

```{.python .input  n=65}
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_dl = DataLoader(TensorDataset(train_features, train_labels), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_dl:
            pred_y = net(X)
            l = loss(pred_y, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls
```

## $K$折交叉验证

我们在[“模型选择、欠拟合和过拟合”](underfit-overfit.md)一节中介绍了$K$折交叉验证。它将被用来选择模型设计并调节超参数。下面实现了一个函数，它返回第`i`折交叉验证时所需要的训练和验证数据。

```{.python .input  n=66}
def get_k_fold_data(k, i, X, y):
    assert k>1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j* fold_size, (j+1)*fold_size)
        X_part, y_part = X[idx,:], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat((X_train, X_part), dim=0)
            y_train = torch.cat((y_train, y_part), dim=0)
    return X_train, y_train, X_valid, y_valid
```

在$K$折交叉验证中我们训练$K$次并返回训练和验证的平均误差。

```{.python .input  n=69}
def k_fold(k, X_train, y_train, num_epochs,
           learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                           weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse',
                         range(1, num_epochs + 1), valid_ls,
                         ['train', 'valid'])
        print('fold %d, train rmse %f, valid rmse %f'
              % (i, train_ls[-1], valid_ls[-1]))
    return train_l_sum / k, valid_l_sum / k
```

## 模型选择

我们使用一组未经调优的超参数并计算交叉验证误差。可以改动这些超参数来尽可能减小平均测试误差。

```{.python .input  n=70}
k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
                          weight_decay, batch_size)
print('%d-fold validation: avg train rmse %f, avg valid rmse %f'
      % (k, train_l, valid_l))
```

```{.json .output n=70}
[
 {
  "data": {
   "image/svg+xml": "<?xml version=\"1.0\" encoding=\"utf-8\" standalone=\"no\"?>\n<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\"\n  \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n<!-- Created with matplotlib (http://matplotlib.org/) -->\n<svg height=\"184.15625pt\" version=\"1.1\" viewBox=\"0 0 251.478125 184.15625\" width=\"251.478125pt\" xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\">\n <defs>\n  <style type=\"text/css\">\n*{stroke-linecap:butt;stroke-linejoin:round;}\n  </style>\n </defs>\n <g id=\"figure_1\">\n  <g id=\"patch_1\">\n   <path d=\"M 0 184.15625 \nL 251.478125 184.15625 \nL 251.478125 -0 \nL 0 -0 \nz\n\" style=\"fill:none;\"/>\n  </g>\n  <g id=\"axes_1\">\n   <g id=\"patch_2\">\n    <path d=\"M 45.478125 146.6 \nL 240.778125 146.6 \nL 240.778125 10.7 \nL 45.478125 10.7 \nz\n\" style=\"fill:#ffffff;\"/>\n   </g>\n   <g id=\"matplotlib.axis_1\">\n    <g id=\"xtick_1\">\n     <g id=\"line2d_1\">\n      <defs>\n       <path d=\"M 0 0 \nL 0 3.5 \n\" id=\"m4d47c79f27\" style=\"stroke:#000000;stroke-width:0.8;\"/>\n      </defs>\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"52.562009\" xlink:href=\"#m4d47c79f27\" y=\"146.6\"/>\n      </g>\n     </g>\n     <g id=\"text_1\">\n      <!-- 0 -->\n      <defs>\n       <path d=\"M 31.78125 66.40625 \nQ 24.171875 66.40625 20.328125 58.90625 \nQ 16.5 51.421875 16.5 36.375 \nQ 16.5 21.390625 20.328125 13.890625 \nQ 24.171875 6.390625 31.78125 6.390625 \nQ 39.453125 6.390625 43.28125 13.890625 \nQ 47.125 21.390625 47.125 36.375 \nQ 47.125 51.421875 43.28125 58.90625 \nQ 39.453125 66.40625 31.78125 66.40625 \nz\nM 31.78125 74.21875 \nQ 44.046875 74.21875 50.515625 64.515625 \nQ 56.984375 54.828125 56.984375 36.375 \nQ 56.984375 17.96875 50.515625 8.265625 \nQ 44.046875 -1.421875 31.78125 -1.421875 \nQ 19.53125 -1.421875 13.0625 8.265625 \nQ 6.59375 17.96875 6.59375 36.375 \nQ 6.59375 54.828125 13.0625 64.515625 \nQ 19.53125 74.21875 31.78125 74.21875 \nz\n\" id=\"DejaVuSans-30\"/>\n      </defs>\n      <g transform=\"translate(49.380759 161.198437)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-30\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_2\">\n     <g id=\"line2d_2\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"88.429778\" xlink:href=\"#m4d47c79f27\" y=\"146.6\"/>\n      </g>\n     </g>\n     <g id=\"text_2\">\n      <!-- 20 -->\n      <defs>\n       <path d=\"M 19.1875 8.296875 \nL 53.609375 8.296875 \nL 53.609375 0 \nL 7.328125 0 \nL 7.328125 8.296875 \nQ 12.9375 14.109375 22.625 23.890625 \nQ 32.328125 33.6875 34.8125 36.53125 \nQ 39.546875 41.84375 41.421875 45.53125 \nQ 43.3125 49.21875 43.3125 52.78125 \nQ 43.3125 58.59375 39.234375 62.25 \nQ 35.15625 65.921875 28.609375 65.921875 \nQ 23.96875 65.921875 18.8125 64.3125 \nQ 13.671875 62.703125 7.8125 59.421875 \nL 7.8125 69.390625 \nQ 13.765625 71.78125 18.9375 73 \nQ 24.125 74.21875 28.421875 74.21875 \nQ 39.75 74.21875 46.484375 68.546875 \nQ 53.21875 62.890625 53.21875 53.421875 \nQ 53.21875 48.921875 51.53125 44.890625 \nQ 49.859375 40.875 45.40625 35.40625 \nQ 44.1875 33.984375 37.640625 27.21875 \nQ 31.109375 20.453125 19.1875 8.296875 \nz\n\" id=\"DejaVuSans-32\"/>\n      </defs>\n      <g transform=\"translate(82.067278 161.198437)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-32\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-30\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_3\">\n     <g id=\"line2d_3\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"124.297546\" xlink:href=\"#m4d47c79f27\" y=\"146.6\"/>\n      </g>\n     </g>\n     <g id=\"text_3\">\n      <!-- 40 -->\n      <defs>\n       <path d=\"M 37.796875 64.3125 \nL 12.890625 25.390625 \nL 37.796875 25.390625 \nz\nM 35.203125 72.90625 \nL 47.609375 72.90625 \nL 47.609375 25.390625 \nL 58.015625 25.390625 \nL 58.015625 17.1875 \nL 47.609375 17.1875 \nL 47.609375 0 \nL 37.796875 0 \nL 37.796875 17.1875 \nL 4.890625 17.1875 \nL 4.890625 26.703125 \nz\n\" id=\"DejaVuSans-34\"/>\n      </defs>\n      <g transform=\"translate(117.935046 161.198437)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-34\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-30\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_4\">\n     <g id=\"line2d_4\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"160.165315\" xlink:href=\"#m4d47c79f27\" y=\"146.6\"/>\n      </g>\n     </g>\n     <g id=\"text_4\">\n      <!-- 60 -->\n      <defs>\n       <path d=\"M 33.015625 40.375 \nQ 26.375 40.375 22.484375 35.828125 \nQ 18.609375 31.296875 18.609375 23.390625 \nQ 18.609375 15.53125 22.484375 10.953125 \nQ 26.375 6.390625 33.015625 6.390625 \nQ 39.65625 6.390625 43.53125 10.953125 \nQ 47.40625 15.53125 47.40625 23.390625 \nQ 47.40625 31.296875 43.53125 35.828125 \nQ 39.65625 40.375 33.015625 40.375 \nz\nM 52.59375 71.296875 \nL 52.59375 62.3125 \nQ 48.875 64.0625 45.09375 64.984375 \nQ 41.3125 65.921875 37.59375 65.921875 \nQ 27.828125 65.921875 22.671875 59.328125 \nQ 17.53125 52.734375 16.796875 39.40625 \nQ 19.671875 43.65625 24.015625 45.921875 \nQ 28.375 48.1875 33.59375 48.1875 \nQ 44.578125 48.1875 50.953125 41.515625 \nQ 57.328125 34.859375 57.328125 23.390625 \nQ 57.328125 12.15625 50.6875 5.359375 \nQ 44.046875 -1.421875 33.015625 -1.421875 \nQ 20.359375 -1.421875 13.671875 8.265625 \nQ 6.984375 17.96875 6.984375 36.375 \nQ 6.984375 53.65625 15.1875 63.9375 \nQ 23.390625 74.21875 37.203125 74.21875 \nQ 40.921875 74.21875 44.703125 73.484375 \nQ 48.484375 72.75 52.59375 71.296875 \nz\n\" id=\"DejaVuSans-36\"/>\n      </defs>\n      <g transform=\"translate(153.802815 161.198437)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-36\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-30\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_5\">\n     <g id=\"line2d_5\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"196.033084\" xlink:href=\"#m4d47c79f27\" y=\"146.6\"/>\n      </g>\n     </g>\n     <g id=\"text_5\">\n      <!-- 80 -->\n      <defs>\n       <path d=\"M 31.78125 34.625 \nQ 24.75 34.625 20.71875 30.859375 \nQ 16.703125 27.09375 16.703125 20.515625 \nQ 16.703125 13.921875 20.71875 10.15625 \nQ 24.75 6.390625 31.78125 6.390625 \nQ 38.8125 6.390625 42.859375 10.171875 \nQ 46.921875 13.96875 46.921875 20.515625 \nQ 46.921875 27.09375 42.890625 30.859375 \nQ 38.875 34.625 31.78125 34.625 \nz\nM 21.921875 38.8125 \nQ 15.578125 40.375 12.03125 44.71875 \nQ 8.5 49.078125 8.5 55.328125 \nQ 8.5 64.0625 14.71875 69.140625 \nQ 20.953125 74.21875 31.78125 74.21875 \nQ 42.671875 74.21875 48.875 69.140625 \nQ 55.078125 64.0625 55.078125 55.328125 \nQ 55.078125 49.078125 51.53125 44.71875 \nQ 48 40.375 41.703125 38.8125 \nQ 48.828125 37.15625 52.796875 32.3125 \nQ 56.78125 27.484375 56.78125 20.515625 \nQ 56.78125 9.90625 50.3125 4.234375 \nQ 43.84375 -1.421875 31.78125 -1.421875 \nQ 19.734375 -1.421875 13.25 4.234375 \nQ 6.78125 9.90625 6.78125 20.515625 \nQ 6.78125 27.484375 10.78125 32.3125 \nQ 14.796875 37.15625 21.921875 38.8125 \nz\nM 18.3125 54.390625 \nQ 18.3125 48.734375 21.84375 45.5625 \nQ 25.390625 42.390625 31.78125 42.390625 \nQ 38.140625 42.390625 41.71875 45.5625 \nQ 45.3125 48.734375 45.3125 54.390625 \nQ 45.3125 60.0625 41.71875 63.234375 \nQ 38.140625 66.40625 31.78125 66.40625 \nQ 25.390625 66.40625 21.84375 63.234375 \nQ 18.3125 60.0625 18.3125 54.390625 \nz\n\" id=\"DejaVuSans-38\"/>\n      </defs>\n      <g transform=\"translate(189.670584 161.198437)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-38\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-30\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_6\">\n     <g id=\"line2d_6\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"231.900852\" xlink:href=\"#m4d47c79f27\" y=\"146.6\"/>\n      </g>\n     </g>\n     <g id=\"text_6\">\n      <!-- 100 -->\n      <defs>\n       <path d=\"M 12.40625 8.296875 \nL 28.515625 8.296875 \nL 28.515625 63.921875 \nL 10.984375 60.40625 \nL 10.984375 69.390625 \nL 28.421875 72.90625 \nL 38.28125 72.90625 \nL 38.28125 8.296875 \nL 54.390625 8.296875 \nL 54.390625 0 \nL 12.40625 0 \nz\n\" id=\"DejaVuSans-31\"/>\n      </defs>\n      <g transform=\"translate(222.357102 161.198437)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-31\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-30\"/>\n       <use x=\"127.246094\" xlink:href=\"#DejaVuSans-30\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"text_7\">\n     <!-- epochs -->\n     <defs>\n      <path d=\"M 56.203125 29.59375 \nL 56.203125 25.203125 \nL 14.890625 25.203125 \nQ 15.484375 15.921875 20.484375 11.0625 \nQ 25.484375 6.203125 34.421875 6.203125 \nQ 39.59375 6.203125 44.453125 7.46875 \nQ 49.3125 8.734375 54.109375 11.28125 \nL 54.109375 2.78125 \nQ 49.265625 0.734375 44.1875 -0.34375 \nQ 39.109375 -1.421875 33.890625 -1.421875 \nQ 20.796875 -1.421875 13.15625 6.1875 \nQ 5.515625 13.8125 5.515625 26.8125 \nQ 5.515625 40.234375 12.765625 48.109375 \nQ 20.015625 56 32.328125 56 \nQ 43.359375 56 49.78125 48.890625 \nQ 56.203125 41.796875 56.203125 29.59375 \nz\nM 47.21875 32.234375 \nQ 47.125 39.59375 43.09375 43.984375 \nQ 39.0625 48.390625 32.421875 48.390625 \nQ 24.90625 48.390625 20.390625 44.140625 \nQ 15.875 39.890625 15.1875 32.171875 \nz\n\" id=\"DejaVuSans-65\"/>\n      <path d=\"M 18.109375 8.203125 \nL 18.109375 -20.796875 \nL 9.078125 -20.796875 \nL 9.078125 54.6875 \nL 18.109375 54.6875 \nL 18.109375 46.390625 \nQ 20.953125 51.265625 25.265625 53.625 \nQ 29.59375 56 35.59375 56 \nQ 45.5625 56 51.78125 48.09375 \nQ 58.015625 40.1875 58.015625 27.296875 \nQ 58.015625 14.40625 51.78125 6.484375 \nQ 45.5625 -1.421875 35.59375 -1.421875 \nQ 29.59375 -1.421875 25.265625 0.953125 \nQ 20.953125 3.328125 18.109375 8.203125 \nz\nM 48.6875 27.296875 \nQ 48.6875 37.203125 44.609375 42.84375 \nQ 40.53125 48.484375 33.40625 48.484375 \nQ 26.265625 48.484375 22.1875 42.84375 \nQ 18.109375 37.203125 18.109375 27.296875 \nQ 18.109375 17.390625 22.1875 11.75 \nQ 26.265625 6.109375 33.40625 6.109375 \nQ 40.53125 6.109375 44.609375 11.75 \nQ 48.6875 17.390625 48.6875 27.296875 \nz\n\" id=\"DejaVuSans-70\"/>\n      <path d=\"M 30.609375 48.390625 \nQ 23.390625 48.390625 19.1875 42.75 \nQ 14.984375 37.109375 14.984375 27.296875 \nQ 14.984375 17.484375 19.15625 11.84375 \nQ 23.34375 6.203125 30.609375 6.203125 \nQ 37.796875 6.203125 41.984375 11.859375 \nQ 46.1875 17.53125 46.1875 27.296875 \nQ 46.1875 37.015625 41.984375 42.703125 \nQ 37.796875 48.390625 30.609375 48.390625 \nz\nM 30.609375 56 \nQ 42.328125 56 49.015625 48.375 \nQ 55.71875 40.765625 55.71875 27.296875 \nQ 55.71875 13.875 49.015625 6.21875 \nQ 42.328125 -1.421875 30.609375 -1.421875 \nQ 18.84375 -1.421875 12.171875 6.21875 \nQ 5.515625 13.875 5.515625 27.296875 \nQ 5.515625 40.765625 12.171875 48.375 \nQ 18.84375 56 30.609375 56 \nz\n\" id=\"DejaVuSans-6f\"/>\n      <path d=\"M 48.78125 52.59375 \nL 48.78125 44.1875 \nQ 44.96875 46.296875 41.140625 47.34375 \nQ 37.3125 48.390625 33.40625 48.390625 \nQ 24.65625 48.390625 19.8125 42.84375 \nQ 14.984375 37.3125 14.984375 27.296875 \nQ 14.984375 17.28125 19.8125 11.734375 \nQ 24.65625 6.203125 33.40625 6.203125 \nQ 37.3125 6.203125 41.140625 7.25 \nQ 44.96875 8.296875 48.78125 10.40625 \nL 48.78125 2.09375 \nQ 45.015625 0.34375 40.984375 -0.53125 \nQ 36.96875 -1.421875 32.421875 -1.421875 \nQ 20.0625 -1.421875 12.78125 6.34375 \nQ 5.515625 14.109375 5.515625 27.296875 \nQ 5.515625 40.671875 12.859375 48.328125 \nQ 20.21875 56 33.015625 56 \nQ 37.15625 56 41.109375 55.140625 \nQ 45.0625 54.296875 48.78125 52.59375 \nz\n\" id=\"DejaVuSans-63\"/>\n      <path d=\"M 54.890625 33.015625 \nL 54.890625 0 \nL 45.90625 0 \nL 45.90625 32.71875 \nQ 45.90625 40.484375 42.875 44.328125 \nQ 39.84375 48.1875 33.796875 48.1875 \nQ 26.515625 48.1875 22.3125 43.546875 \nQ 18.109375 38.921875 18.109375 30.90625 \nL 18.109375 0 \nL 9.078125 0 \nL 9.078125 75.984375 \nL 18.109375 75.984375 \nL 18.109375 46.1875 \nQ 21.34375 51.125 25.703125 53.5625 \nQ 30.078125 56 35.796875 56 \nQ 45.21875 56 50.046875 50.171875 \nQ 54.890625 44.34375 54.890625 33.015625 \nz\n\" id=\"DejaVuSans-68\"/>\n      <path d=\"M 44.28125 53.078125 \nL 44.28125 44.578125 \nQ 40.484375 46.53125 36.375 47.5 \nQ 32.28125 48.484375 27.875 48.484375 \nQ 21.1875 48.484375 17.84375 46.4375 \nQ 14.5 44.390625 14.5 40.28125 \nQ 14.5 37.15625 16.890625 35.375 \nQ 19.28125 33.59375 26.515625 31.984375 \nL 29.59375 31.296875 \nQ 39.15625 29.25 43.1875 25.515625 \nQ 47.21875 21.78125 47.21875 15.09375 \nQ 47.21875 7.46875 41.1875 3.015625 \nQ 35.15625 -1.421875 24.609375 -1.421875 \nQ 20.21875 -1.421875 15.453125 -0.5625 \nQ 10.6875 0.296875 5.421875 2 \nL 5.421875 11.28125 \nQ 10.40625 8.6875 15.234375 7.390625 \nQ 20.0625 6.109375 24.8125 6.109375 \nQ 31.15625 6.109375 34.5625 8.28125 \nQ 37.984375 10.453125 37.984375 14.40625 \nQ 37.984375 18.0625 35.515625 20.015625 \nQ 33.0625 21.96875 24.703125 23.78125 \nL 21.578125 24.515625 \nQ 13.234375 26.265625 9.515625 29.90625 \nQ 5.8125 33.546875 5.8125 39.890625 \nQ 5.8125 47.609375 11.28125 51.796875 \nQ 16.75 56 26.8125 56 \nQ 31.78125 56 36.171875 55.265625 \nQ 40.578125 54.546875 44.28125 53.078125 \nz\n\" id=\"DejaVuSans-73\"/>\n     </defs>\n     <g transform=\"translate(125.295312 174.876562)scale(0.1 -0.1)\">\n      <use xlink:href=\"#DejaVuSans-65\"/>\n      <use x=\"61.523438\" xlink:href=\"#DejaVuSans-70\"/>\n      <use x=\"125\" xlink:href=\"#DejaVuSans-6f\"/>\n      <use x=\"186.181641\" xlink:href=\"#DejaVuSans-63\"/>\n      <use x=\"241.162109\" xlink:href=\"#DejaVuSans-68\"/>\n      <use x=\"304.541016\" xlink:href=\"#DejaVuSans-73\"/>\n     </g>\n    </g>\n   </g>\n   <g id=\"matplotlib.axis_2\">\n    <g id=\"ytick_1\">\n     <g id=\"line2d_7\">\n      <defs>\n       <path d=\"M 0 0 \nL -3.5 0 \n\" id=\"m5ff04dc685\" style=\"stroke:#000000;stroke-width:0.8;\"/>\n      </defs>\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"45.478125\" xlink:href=\"#m5ff04dc685\" y=\"68.329869\"/>\n      </g>\n     </g>\n     <g id=\"text_8\">\n      <!-- $\\mathdefault{10^{0}}$ -->\n      <g transform=\"translate(20.878125 72.129088)scale(0.1 -0.1)\">\n       <use transform=\"translate(0 0.765625)\" xlink:href=\"#DejaVuSans-31\"/>\n       <use transform=\"translate(63.623047 0.765625)\" xlink:href=\"#DejaVuSans-30\"/>\n       <use transform=\"translate(128.203125 39.046875)scale(0.7)\" xlink:href=\"#DejaVuSans-30\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_2\">\n     <g id=\"line2d_8\">\n      <defs>\n       <path d=\"M 0 0 \nL -2 0 \n\" id=\"mdb287bceb6\" style=\"stroke:#000000;stroke-width:0.6;\"/>\n      </defs>\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.6;\" x=\"45.478125\" xlink:href=\"#mdb287bceb6\" y=\"130.914059\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_3\">\n     <g id=\"line2d_9\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.6;\" x=\"45.478125\" xlink:href=\"#mdb287bceb6\" y=\"115.147247\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_4\">\n     <g id=\"line2d_10\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.6;\" x=\"45.478125\" xlink:href=\"#mdb287bceb6\" y=\"103.960515\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_5\">\n     <g id=\"line2d_11\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.6;\" x=\"45.478125\" xlink:href=\"#mdb287bceb6\" y=\"95.283412\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_6\">\n     <g id=\"line2d_12\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.6;\" x=\"45.478125\" xlink:href=\"#mdb287bceb6\" y=\"88.193703\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_7\">\n     <g id=\"line2d_13\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.6;\" x=\"45.478125\" xlink:href=\"#mdb287bceb6\" y=\"82.199439\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_8\">\n     <g id=\"line2d_14\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.6;\" x=\"45.478125\" xlink:href=\"#mdb287bceb6\" y=\"77.006972\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_9\">\n     <g id=\"line2d_15\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.6;\" x=\"45.478125\" xlink:href=\"#mdb287bceb6\" y=\"72.426891\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_10\">\n     <g id=\"line2d_16\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.6;\" x=\"45.478125\" xlink:href=\"#mdb287bceb6\" y=\"41.376325\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_11\">\n     <g id=\"line2d_17\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.6;\" x=\"45.478125\" xlink:href=\"#mdb287bceb6\" y=\"25.609513\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_12\">\n     <g id=\"line2d_18\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.6;\" x=\"45.478125\" xlink:href=\"#mdb287bceb6\" y=\"14.422782\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"text_9\">\n     <!-- rmse -->\n     <defs>\n      <path d=\"M 41.109375 46.296875 \nQ 39.59375 47.171875 37.8125 47.578125 \nQ 36.03125 48 33.890625 48 \nQ 26.265625 48 22.1875 43.046875 \nQ 18.109375 38.09375 18.109375 28.8125 \nL 18.109375 0 \nL 9.078125 0 \nL 9.078125 54.6875 \nL 18.109375 54.6875 \nL 18.109375 46.1875 \nQ 20.953125 51.171875 25.484375 53.578125 \nQ 30.03125 56 36.53125 56 \nQ 37.453125 56 38.578125 55.875 \nQ 39.703125 55.765625 41.0625 55.515625 \nz\n\" id=\"DejaVuSans-72\"/>\n      <path d=\"M 52 44.1875 \nQ 55.375 50.25 60.0625 53.125 \nQ 64.75 56 71.09375 56 \nQ 79.640625 56 84.28125 50.015625 \nQ 88.921875 44.046875 88.921875 33.015625 \nL 88.921875 0 \nL 79.890625 0 \nL 79.890625 32.71875 \nQ 79.890625 40.578125 77.09375 44.375 \nQ 74.3125 48.1875 68.609375 48.1875 \nQ 61.625 48.1875 57.5625 43.546875 \nQ 53.515625 38.921875 53.515625 30.90625 \nL 53.515625 0 \nL 44.484375 0 \nL 44.484375 32.71875 \nQ 44.484375 40.625 41.703125 44.40625 \nQ 38.921875 48.1875 33.109375 48.1875 \nQ 26.21875 48.1875 22.15625 43.53125 \nQ 18.109375 38.875 18.109375 30.90625 \nL 18.109375 0 \nL 9.078125 0 \nL 9.078125 54.6875 \nL 18.109375 54.6875 \nL 18.109375 46.1875 \nQ 21.1875 51.21875 25.484375 53.609375 \nQ 29.78125 56 35.6875 56 \nQ 41.65625 56 45.828125 52.96875 \nQ 50 49.953125 52 44.1875 \nz\n\" id=\"DejaVuSans-6d\"/>\n     </defs>\n     <g transform=\"translate(14.798437 91.25625)rotate(-90)scale(0.1 -0.1)\">\n      <use xlink:href=\"#DejaVuSans-72\"/>\n      <use x=\"41.097656\" xlink:href=\"#DejaVuSans-6d\"/>\n      <use x=\"138.509766\" xlink:href=\"#DejaVuSans-73\"/>\n      <use x=\"190.609375\" xlink:href=\"#DejaVuSans-65\"/>\n     </g>\n    </g>\n   </g>\n   <g id=\"line2d_19\">\n    <path clip-path=\"url(#p02dc8cc17a)\" d=\"M 54.355398 17.072535 \nL 56.148786 24.967446 \nL 57.942175 30.41478 \nL 59.735563 34.800374 \nL 61.528951 38.510621 \nL 63.32234 41.81248 \nL 65.115728 44.807685 \nL 66.909117 47.597178 \nL 68.702505 50.217952 \nL 70.495894 52.674664 \nL 72.289282 55.038951 \nL 74.08267 57.289193 \nL 75.876059 59.474274 \nL 77.669447 61.586873 \nL 79.462836 63.643144 \nL 81.256224 65.662086 \nL 83.049613 67.629284 \nL 84.843001 69.566527 \nL 86.636389 71.455599 \nL 88.429778 73.314226 \nL 90.223166 75.138626 \nL 92.016555 76.958615 \nL 93.809943 78.744149 \nL 95.603332 80.506143 \nL 97.39672 82.25069 \nL 99.190108 83.971482 \nL 100.983497 85.676659 \nL 102.776885 87.354537 \nL 104.570274 89.042764 \nL 106.363662 90.713799 \nL 108.157051 92.341503 \nL 109.950439 93.974601 \nL 111.743827 95.564958 \nL 113.537216 97.149905 \nL 115.330604 98.735843 \nL 117.123993 100.31195 \nL 118.917381 101.846984 \nL 120.71077 103.381133 \nL 122.504158 104.871508 \nL 124.297546 106.351858 \nL 126.090935 107.860706 \nL 127.884323 109.27405 \nL 129.677712 110.706163 \nL 131.4711 112.102744 \nL 133.264489 113.45367 \nL 135.057877 114.780727 \nL 136.851265 116.096959 \nL 138.644654 117.367923 \nL 140.438042 118.568432 \nL 142.231431 119.788576 \nL 144.024819 120.935007 \nL 145.818208 122.00277 \nL 147.611596 123.060601 \nL 149.404985 124.087606 \nL 151.198373 125.061919 \nL 152.991761 125.981319 \nL 154.78515 126.877961 \nL 156.578538 127.727265 \nL 158.371927 128.520734 \nL 160.165315 129.243241 \nL 161.958704 129.955658 \nL 163.752092 130.596623 \nL 165.54548 131.17888 \nL 167.338869 131.781045 \nL 169.132257 132.318763 \nL 170.925646 132.79079 \nL 172.719034 133.237922 \nL 174.512423 133.631705 \nL 176.305811 134.019203 \nL 178.099199 134.333513 \nL 179.892588 134.681735 \nL 181.685976 134.976393 \nL 183.479365 135.2238 \nL 185.272753 135.458517 \nL 187.066142 135.671268 \nL 188.85953 135.856852 \nL 190.652918 136.025949 \nL 192.446307 136.167572 \nL 194.239695 136.306554 \nL 196.033084 136.42423 \nL 197.826472 136.532584 \nL 199.619861 136.617263 \nL 201.413249 136.700903 \nL 203.206637 136.763751 \nL 205.000026 136.830006 \nL 206.793414 136.878659 \nL 208.586803 136.92878 \nL 210.380191 136.963314 \nL 212.17358 136.987503 \nL 213.966968 137.015184 \nL 215.760356 137.0259 \nL 217.553745 137.04933 \nL 219.347133 137.065098 \nL 221.140522 137.081126 \nL 222.93391 137.085537 \nL 224.727299 137.094736 \nL 226.520687 137.057697 \nL 228.314075 137.074822 \nL 230.107464 137.071305 \nL 231.900852 137.084871 \n\" style=\"fill:none;stroke:#1f77b4;stroke-linecap:square;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_20\">\n    <path clip-path=\"url(#p02dc8cc17a)\" d=\"M 54.355398 16.877273 \nL 56.148786 24.76377 \nL 57.942175 30.19534 \nL 59.735563 34.55893 \nL 61.528951 38.253531 \nL 63.32234 41.536079 \nL 65.115728 44.519264 \nL 66.909117 47.292595 \nL 68.702505 49.895615 \nL 70.495894 52.335673 \nL 72.289282 54.685465 \nL 74.08267 56.920711 \nL 75.876059 59.087162 \nL 77.669447 61.190192 \nL 79.462836 63.23545 \nL 81.256224 65.239861 \nL 83.049613 67.19186 \nL 84.843001 69.114295 \nL 86.636389 70.99374 \nL 88.429778 72.841193 \nL 90.223166 74.655272 \nL 92.016555 76.462309 \nL 93.809943 78.24115 \nL 95.603332 79.999094 \nL 97.39672 81.73545 \nL 99.190108 83.45496 \nL 100.983497 85.155206 \nL 102.776885 86.832733 \nL 104.570274 88.520124 \nL 106.363662 90.193332 \nL 108.157051 91.826047 \nL 109.950439 93.464132 \nL 111.743827 95.061527 \nL 113.537216 96.657187 \nL 115.330604 98.254461 \nL 117.123993 99.847765 \nL 118.917381 101.405862 \nL 120.71077 102.965066 \nL 122.504158 104.48489 \nL 124.297546 105.998952 \nL 126.090935 107.542956 \nL 127.884323 108.999532 \nL 129.677712 110.476671 \nL 131.4711 111.928263 \nL 133.264489 113.339113 \nL 135.057877 114.731149 \nL 136.851265 116.120175 \nL 138.644654 117.465325 \nL 140.438042 118.744464 \nL 142.231431 120.056919 \nL 144.024819 121.303182 \nL 145.818208 122.47101 \nL 147.611596 123.632373 \nL 149.404985 124.766865 \nL 151.198373 125.852611 \nL 152.991761 126.890849 \nL 154.78515 127.911009 \nL 156.578538 128.883565 \nL 158.371927 129.799042 \nL 160.165315 130.642192 \nL 161.958704 131.476169 \nL 163.752092 132.238878 \nL 165.54548 132.941347 \nL 167.338869 133.660532 \nL 169.132257 134.31477 \nL 170.925646 134.894618 \nL 172.719034 135.451429 \nL 174.512423 135.948027 \nL 176.305811 136.428877 \nL 178.099199 136.818765 \nL 179.892588 137.257468 \nL 181.685976 137.637552 \nL 183.479365 137.950256 \nL 185.272753 138.26314 \nL 187.066142 138.552729 \nL 188.85953 138.792652 \nL 190.652918 139.018714 \nL 192.446307 139.199526 \nL 194.239695 139.38047 \nL 196.033084 139.534147 \nL 197.826472 139.686994 \nL 199.619861 139.787069 \nL 201.413249 139.894321 \nL 203.206637 139.969461 \nL 205.000026 140.058605 \nL 206.793414 140.123275 \nL 208.586803 140.191818 \nL 210.380191 140.230179 \nL 212.17358 140.272882 \nL 213.966968 140.30928 \nL 215.760356 140.330715 \nL 217.553745 140.36362 \nL 219.347133 140.380765 \nL 221.140522 140.409392 \nL 222.93391 140.417378 \nL 224.727299 140.422727 \nL 226.520687 140.379253 \nL 228.314075 140.393835 \nL 230.107464 140.380014 \nL 231.900852 140.381271 \n\" style=\"fill:none;stroke:#ff7f0e;stroke-dasharray:1.5,2.475;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"patch_3\">\n    <path d=\"M 45.478125 146.6 \nL 45.478125 10.7 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_4\">\n    <path d=\"M 240.778125 146.6 \nL 240.778125 10.7 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_5\">\n    <path d=\"M 45.478125 146.6 \nL 240.778125 146.6 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_6\">\n    <path d=\"M 45.478125 10.7 \nL 240.778125 10.7 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"legend_1\">\n    <g id=\"patch_7\">\n     <path d=\"M 177.826562 48.05625 \nL 233.778125 48.05625 \nQ 235.778125 48.05625 235.778125 46.05625 \nL 235.778125 17.7 \nQ 235.778125 15.7 233.778125 15.7 \nL 177.826562 15.7 \nQ 175.826562 15.7 175.826562 17.7 \nL 175.826562 46.05625 \nQ 175.826562 48.05625 177.826562 48.05625 \nz\n\" style=\"fill:#ffffff;opacity:0.8;stroke:#cccccc;stroke-linejoin:miter;\"/>\n    </g>\n    <g id=\"line2d_21\">\n     <path d=\"M 179.826562 23.798437 \nL 199.826562 23.798437 \n\" style=\"fill:none;stroke:#1f77b4;stroke-linecap:square;stroke-width:1.5;\"/>\n    </g>\n    <g id=\"line2d_22\"/>\n    <g id=\"text_10\">\n     <!-- train -->\n     <defs>\n      <path d=\"M 18.3125 70.21875 \nL 18.3125 54.6875 \nL 36.8125 54.6875 \nL 36.8125 47.703125 \nL 18.3125 47.703125 \nL 18.3125 18.015625 \nQ 18.3125 11.328125 20.140625 9.421875 \nQ 21.96875 7.515625 27.59375 7.515625 \nL 36.8125 7.515625 \nL 36.8125 0 \nL 27.59375 0 \nQ 17.1875 0 13.234375 3.875 \nQ 9.28125 7.765625 9.28125 18.015625 \nL 9.28125 47.703125 \nL 2.6875 47.703125 \nL 2.6875 54.6875 \nL 9.28125 54.6875 \nL 9.28125 70.21875 \nz\n\" id=\"DejaVuSans-74\"/>\n      <path d=\"M 34.28125 27.484375 \nQ 23.390625 27.484375 19.1875 25 \nQ 14.984375 22.515625 14.984375 16.5 \nQ 14.984375 11.71875 18.140625 8.90625 \nQ 21.296875 6.109375 26.703125 6.109375 \nQ 34.1875 6.109375 38.703125 11.40625 \nQ 43.21875 16.703125 43.21875 25.484375 \nL 43.21875 27.484375 \nz\nM 52.203125 31.203125 \nL 52.203125 0 \nL 43.21875 0 \nL 43.21875 8.296875 \nQ 40.140625 3.328125 35.546875 0.953125 \nQ 30.953125 -1.421875 24.3125 -1.421875 \nQ 15.921875 -1.421875 10.953125 3.296875 \nQ 6 8.015625 6 15.921875 \nQ 6 25.140625 12.171875 29.828125 \nQ 18.359375 34.515625 30.609375 34.515625 \nL 43.21875 34.515625 \nL 43.21875 35.40625 \nQ 43.21875 41.609375 39.140625 45 \nQ 35.0625 48.390625 27.6875 48.390625 \nQ 23 48.390625 18.546875 47.265625 \nQ 14.109375 46.140625 10.015625 43.890625 \nL 10.015625 52.203125 \nQ 14.9375 54.109375 19.578125 55.046875 \nQ 24.21875 56 28.609375 56 \nQ 40.484375 56 46.34375 49.84375 \nQ 52.203125 43.703125 52.203125 31.203125 \nz\n\" id=\"DejaVuSans-61\"/>\n      <path d=\"M 9.421875 54.6875 \nL 18.40625 54.6875 \nL 18.40625 0 \nL 9.421875 0 \nz\nM 9.421875 75.984375 \nL 18.40625 75.984375 \nL 18.40625 64.59375 \nL 9.421875 64.59375 \nz\n\" id=\"DejaVuSans-69\"/>\n      <path d=\"M 54.890625 33.015625 \nL 54.890625 0 \nL 45.90625 0 \nL 45.90625 32.71875 \nQ 45.90625 40.484375 42.875 44.328125 \nQ 39.84375 48.1875 33.796875 48.1875 \nQ 26.515625 48.1875 22.3125 43.546875 \nQ 18.109375 38.921875 18.109375 30.90625 \nL 18.109375 0 \nL 9.078125 0 \nL 9.078125 54.6875 \nL 18.109375 54.6875 \nL 18.109375 46.1875 \nQ 21.34375 51.125 25.703125 53.5625 \nQ 30.078125 56 35.796875 56 \nQ 45.21875 56 50.046875 50.171875 \nQ 54.890625 44.34375 54.890625 33.015625 \nz\n\" id=\"DejaVuSans-6e\"/>\n     </defs>\n     <g transform=\"translate(207.826562 27.298437)scale(0.1 -0.1)\">\n      <use xlink:href=\"#DejaVuSans-74\"/>\n      <use x=\"39.208984\" xlink:href=\"#DejaVuSans-72\"/>\n      <use x=\"80.322266\" xlink:href=\"#DejaVuSans-61\"/>\n      <use x=\"141.601562\" xlink:href=\"#DejaVuSans-69\"/>\n      <use x=\"169.384766\" xlink:href=\"#DejaVuSans-6e\"/>\n     </g>\n    </g>\n    <g id=\"line2d_23\">\n     <path d=\"M 179.826562 38.476562 \nL 199.826562 38.476562 \n\" style=\"fill:none;stroke:#ff7f0e;stroke-dasharray:1.5,2.475;stroke-dashoffset:0;stroke-width:1.5;\"/>\n    </g>\n    <g id=\"line2d_24\"/>\n    <g id=\"text_11\">\n     <!-- valid -->\n     <defs>\n      <path d=\"M 2.984375 54.6875 \nL 12.5 54.6875 \nL 29.59375 8.796875 \nL 46.6875 54.6875 \nL 56.203125 54.6875 \nL 35.6875 0 \nL 23.484375 0 \nz\n\" id=\"DejaVuSans-76\"/>\n      <path d=\"M 9.421875 75.984375 \nL 18.40625 75.984375 \nL 18.40625 0 \nL 9.421875 0 \nz\n\" id=\"DejaVuSans-6c\"/>\n      <path d=\"M 45.40625 46.390625 \nL 45.40625 75.984375 \nL 54.390625 75.984375 \nL 54.390625 0 \nL 45.40625 0 \nL 45.40625 8.203125 \nQ 42.578125 3.328125 38.25 0.953125 \nQ 33.9375 -1.421875 27.875 -1.421875 \nQ 17.96875 -1.421875 11.734375 6.484375 \nQ 5.515625 14.40625 5.515625 27.296875 \nQ 5.515625 40.1875 11.734375 48.09375 \nQ 17.96875 56 27.875 56 \nQ 33.9375 56 38.25 53.625 \nQ 42.578125 51.265625 45.40625 46.390625 \nz\nM 14.796875 27.296875 \nQ 14.796875 17.390625 18.875 11.75 \nQ 22.953125 6.109375 30.078125 6.109375 \nQ 37.203125 6.109375 41.296875 11.75 \nQ 45.40625 17.390625 45.40625 27.296875 \nQ 45.40625 37.203125 41.296875 42.84375 \nQ 37.203125 48.484375 30.078125 48.484375 \nQ 22.953125 48.484375 18.875 42.84375 \nQ 14.796875 37.203125 14.796875 27.296875 \nz\n\" id=\"DejaVuSans-64\"/>\n     </defs>\n     <g transform=\"translate(207.826562 41.976562)scale(0.1 -0.1)\">\n      <use xlink:href=\"#DejaVuSans-76\"/>\n      <use x=\"59.179688\" xlink:href=\"#DejaVuSans-61\"/>\n      <use x=\"120.458984\" xlink:href=\"#DejaVuSans-6c\"/>\n      <use x=\"148.242188\" xlink:href=\"#DejaVuSans-69\"/>\n      <use x=\"176.025391\" xlink:href=\"#DejaVuSans-64\"/>\n     </g>\n    </g>\n   </g>\n  </g>\n </g>\n <defs>\n  <clipPath id=\"p02dc8cc17a\">\n   <rect height=\"135.9\" width=\"195.3\" x=\"45.478125\" y=\"10.7\"/>\n  </clipPath>\n </defs>\n</svg>\n",
   "text/plain": "<Figure size 252x180 with 1 Axes>"
  },
  "metadata": {
   "needs_background": "light"
  },
  "output_type": "display_data"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "fold 0, train rmse 0.170652, valid rmse 0.156782\nfold 1, train rmse 0.161863, valid rmse 0.186399\nfold 2, train rmse 0.164117, valid rmse 0.168819\nfold 3, train rmse 0.168278, valid rmse 0.154722\nfold 4, train rmse 0.162772, valid rmse 0.182688\n5-fold validation: avg train rmse 0.165536, avg valid rmse 0.169882\n"
 }
]
```

有时候你会发现一组参数的训练误差可以达到很低，但是在$K$折交叉验证上的误差可能反而较高。这种现象很可能是由过拟合造成的。因此，当训练误差降低时，我们要观察$K$折交叉验证上的误差是否也相应降低。

## 预测并在Kaggle提交结果

下面定义预测函数。在预测之前，我们会使用完整的训练数据集来重新训练模型，并将预测结果存成提交所需要的格式。

```{.python .input  n=71}
def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse')
    print('train rmse %f' % train_ls[-1])
    preds = net(test_features).detach().numpy()
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)
```

设计好模型并调好超参数之后，下一步就是对测试数据集上的房屋样本做价格预测。如果我们得到与交叉验证时差不多的训练误差，那么这个结果很可能是理想的，可以在Kaggle上提交结果。

```{.python .input  n=72}
train_and_pred(train_features, test_features, train_labels, test_data,
               num_epochs, lr, weight_decay, batch_size)
```

```{.json .output n=72}
[
 {
  "data": {
   "image/svg+xml": "<?xml version=\"1.0\" encoding=\"utf-8\" standalone=\"no\"?>\n<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\"\n  \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n<!-- Created with matplotlib (http://matplotlib.org/) -->\n<svg height=\"184.15625pt\" version=\"1.1\" viewBox=\"0 0 251.478125 184.15625\" width=\"251.478125pt\" xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\">\n <defs>\n  <style type=\"text/css\">\n*{stroke-linecap:butt;stroke-linejoin:round;}\n  </style>\n </defs>\n <g id=\"figure_1\">\n  <g id=\"patch_1\">\n   <path d=\"M 0 184.15625 \nL 251.478125 184.15625 \nL 251.478125 -0 \nL 0 -0 \nz\n\" style=\"fill:none;\"/>\n  </g>\n  <g id=\"axes_1\">\n   <g id=\"patch_2\">\n    <path d=\"M 45.478125 146.6 \nL 240.778125 146.6 \nL 240.778125 10.7 \nL 45.478125 10.7 \nz\n\" style=\"fill:#ffffff;\"/>\n   </g>\n   <g id=\"matplotlib.axis_1\">\n    <g id=\"xtick_1\">\n     <g id=\"line2d_1\">\n      <defs>\n       <path d=\"M 0 0 \nL 0 3.5 \n\" id=\"m655ae8d27d\" style=\"stroke:#000000;stroke-width:0.8;\"/>\n      </defs>\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"52.562009\" xlink:href=\"#m655ae8d27d\" y=\"146.6\"/>\n      </g>\n     </g>\n     <g id=\"text_1\">\n      <!-- 0 -->\n      <defs>\n       <path d=\"M 31.78125 66.40625 \nQ 24.171875 66.40625 20.328125 58.90625 \nQ 16.5 51.421875 16.5 36.375 \nQ 16.5 21.390625 20.328125 13.890625 \nQ 24.171875 6.390625 31.78125 6.390625 \nQ 39.453125 6.390625 43.28125 13.890625 \nQ 47.125 21.390625 47.125 36.375 \nQ 47.125 51.421875 43.28125 58.90625 \nQ 39.453125 66.40625 31.78125 66.40625 \nz\nM 31.78125 74.21875 \nQ 44.046875 74.21875 50.515625 64.515625 \nQ 56.984375 54.828125 56.984375 36.375 \nQ 56.984375 17.96875 50.515625 8.265625 \nQ 44.046875 -1.421875 31.78125 -1.421875 \nQ 19.53125 -1.421875 13.0625 8.265625 \nQ 6.59375 17.96875 6.59375 36.375 \nQ 6.59375 54.828125 13.0625 64.515625 \nQ 19.53125 74.21875 31.78125 74.21875 \nz\n\" id=\"DejaVuSans-30\"/>\n      </defs>\n      <g transform=\"translate(49.380759 161.198437)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-30\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_2\">\n     <g id=\"line2d_2\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"88.429778\" xlink:href=\"#m655ae8d27d\" y=\"146.6\"/>\n      </g>\n     </g>\n     <g id=\"text_2\">\n      <!-- 20 -->\n      <defs>\n       <path d=\"M 19.1875 8.296875 \nL 53.609375 8.296875 \nL 53.609375 0 \nL 7.328125 0 \nL 7.328125 8.296875 \nQ 12.9375 14.109375 22.625 23.890625 \nQ 32.328125 33.6875 34.8125 36.53125 \nQ 39.546875 41.84375 41.421875 45.53125 \nQ 43.3125 49.21875 43.3125 52.78125 \nQ 43.3125 58.59375 39.234375 62.25 \nQ 35.15625 65.921875 28.609375 65.921875 \nQ 23.96875 65.921875 18.8125 64.3125 \nQ 13.671875 62.703125 7.8125 59.421875 \nL 7.8125 69.390625 \nQ 13.765625 71.78125 18.9375 73 \nQ 24.125 74.21875 28.421875 74.21875 \nQ 39.75 74.21875 46.484375 68.546875 \nQ 53.21875 62.890625 53.21875 53.421875 \nQ 53.21875 48.921875 51.53125 44.890625 \nQ 49.859375 40.875 45.40625 35.40625 \nQ 44.1875 33.984375 37.640625 27.21875 \nQ 31.109375 20.453125 19.1875 8.296875 \nz\n\" id=\"DejaVuSans-32\"/>\n      </defs>\n      <g transform=\"translate(82.067278 161.198437)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-32\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-30\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_3\">\n     <g id=\"line2d_3\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"124.297546\" xlink:href=\"#m655ae8d27d\" y=\"146.6\"/>\n      </g>\n     </g>\n     <g id=\"text_3\">\n      <!-- 40 -->\n      <defs>\n       <path d=\"M 37.796875 64.3125 \nL 12.890625 25.390625 \nL 37.796875 25.390625 \nz\nM 35.203125 72.90625 \nL 47.609375 72.90625 \nL 47.609375 25.390625 \nL 58.015625 25.390625 \nL 58.015625 17.1875 \nL 47.609375 17.1875 \nL 47.609375 0 \nL 37.796875 0 \nL 37.796875 17.1875 \nL 4.890625 17.1875 \nL 4.890625 26.703125 \nz\n\" id=\"DejaVuSans-34\"/>\n      </defs>\n      <g transform=\"translate(117.935046 161.198437)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-34\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-30\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_4\">\n     <g id=\"line2d_4\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"160.165315\" xlink:href=\"#m655ae8d27d\" y=\"146.6\"/>\n      </g>\n     </g>\n     <g id=\"text_4\">\n      <!-- 60 -->\n      <defs>\n       <path d=\"M 33.015625 40.375 \nQ 26.375 40.375 22.484375 35.828125 \nQ 18.609375 31.296875 18.609375 23.390625 \nQ 18.609375 15.53125 22.484375 10.953125 \nQ 26.375 6.390625 33.015625 6.390625 \nQ 39.65625 6.390625 43.53125 10.953125 \nQ 47.40625 15.53125 47.40625 23.390625 \nQ 47.40625 31.296875 43.53125 35.828125 \nQ 39.65625 40.375 33.015625 40.375 \nz\nM 52.59375 71.296875 \nL 52.59375 62.3125 \nQ 48.875 64.0625 45.09375 64.984375 \nQ 41.3125 65.921875 37.59375 65.921875 \nQ 27.828125 65.921875 22.671875 59.328125 \nQ 17.53125 52.734375 16.796875 39.40625 \nQ 19.671875 43.65625 24.015625 45.921875 \nQ 28.375 48.1875 33.59375 48.1875 \nQ 44.578125 48.1875 50.953125 41.515625 \nQ 57.328125 34.859375 57.328125 23.390625 \nQ 57.328125 12.15625 50.6875 5.359375 \nQ 44.046875 -1.421875 33.015625 -1.421875 \nQ 20.359375 -1.421875 13.671875 8.265625 \nQ 6.984375 17.96875 6.984375 36.375 \nQ 6.984375 53.65625 15.1875 63.9375 \nQ 23.390625 74.21875 37.203125 74.21875 \nQ 40.921875 74.21875 44.703125 73.484375 \nQ 48.484375 72.75 52.59375 71.296875 \nz\n\" id=\"DejaVuSans-36\"/>\n      </defs>\n      <g transform=\"translate(153.802815 161.198437)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-36\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-30\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_5\">\n     <g id=\"line2d_5\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"196.033084\" xlink:href=\"#m655ae8d27d\" y=\"146.6\"/>\n      </g>\n     </g>\n     <g id=\"text_5\">\n      <!-- 80 -->\n      <defs>\n       <path d=\"M 31.78125 34.625 \nQ 24.75 34.625 20.71875 30.859375 \nQ 16.703125 27.09375 16.703125 20.515625 \nQ 16.703125 13.921875 20.71875 10.15625 \nQ 24.75 6.390625 31.78125 6.390625 \nQ 38.8125 6.390625 42.859375 10.171875 \nQ 46.921875 13.96875 46.921875 20.515625 \nQ 46.921875 27.09375 42.890625 30.859375 \nQ 38.875 34.625 31.78125 34.625 \nz\nM 21.921875 38.8125 \nQ 15.578125 40.375 12.03125 44.71875 \nQ 8.5 49.078125 8.5 55.328125 \nQ 8.5 64.0625 14.71875 69.140625 \nQ 20.953125 74.21875 31.78125 74.21875 \nQ 42.671875 74.21875 48.875 69.140625 \nQ 55.078125 64.0625 55.078125 55.328125 \nQ 55.078125 49.078125 51.53125 44.71875 \nQ 48 40.375 41.703125 38.8125 \nQ 48.828125 37.15625 52.796875 32.3125 \nQ 56.78125 27.484375 56.78125 20.515625 \nQ 56.78125 9.90625 50.3125 4.234375 \nQ 43.84375 -1.421875 31.78125 -1.421875 \nQ 19.734375 -1.421875 13.25 4.234375 \nQ 6.78125 9.90625 6.78125 20.515625 \nQ 6.78125 27.484375 10.78125 32.3125 \nQ 14.796875 37.15625 21.921875 38.8125 \nz\nM 18.3125 54.390625 \nQ 18.3125 48.734375 21.84375 45.5625 \nQ 25.390625 42.390625 31.78125 42.390625 \nQ 38.140625 42.390625 41.71875 45.5625 \nQ 45.3125 48.734375 45.3125 54.390625 \nQ 45.3125 60.0625 41.71875 63.234375 \nQ 38.140625 66.40625 31.78125 66.40625 \nQ 25.390625 66.40625 21.84375 63.234375 \nQ 18.3125 60.0625 18.3125 54.390625 \nz\n\" id=\"DejaVuSans-38\"/>\n      </defs>\n      <g transform=\"translate(189.670584 161.198437)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-38\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-30\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_6\">\n     <g id=\"line2d_6\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"231.900852\" xlink:href=\"#m655ae8d27d\" y=\"146.6\"/>\n      </g>\n     </g>\n     <g id=\"text_6\">\n      <!-- 100 -->\n      <defs>\n       <path d=\"M 12.40625 8.296875 \nL 28.515625 8.296875 \nL 28.515625 63.921875 \nL 10.984375 60.40625 \nL 10.984375 69.390625 \nL 28.421875 72.90625 \nL 38.28125 72.90625 \nL 38.28125 8.296875 \nL 54.390625 8.296875 \nL 54.390625 0 \nL 12.40625 0 \nz\n\" id=\"DejaVuSans-31\"/>\n      </defs>\n      <g transform=\"translate(222.357102 161.198437)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-31\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-30\"/>\n       <use x=\"127.246094\" xlink:href=\"#DejaVuSans-30\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"text_7\">\n     <!-- epochs -->\n     <defs>\n      <path d=\"M 56.203125 29.59375 \nL 56.203125 25.203125 \nL 14.890625 25.203125 \nQ 15.484375 15.921875 20.484375 11.0625 \nQ 25.484375 6.203125 34.421875 6.203125 \nQ 39.59375 6.203125 44.453125 7.46875 \nQ 49.3125 8.734375 54.109375 11.28125 \nL 54.109375 2.78125 \nQ 49.265625 0.734375 44.1875 -0.34375 \nQ 39.109375 -1.421875 33.890625 -1.421875 \nQ 20.796875 -1.421875 13.15625 6.1875 \nQ 5.515625 13.8125 5.515625 26.8125 \nQ 5.515625 40.234375 12.765625 48.109375 \nQ 20.015625 56 32.328125 56 \nQ 43.359375 56 49.78125 48.890625 \nQ 56.203125 41.796875 56.203125 29.59375 \nz\nM 47.21875 32.234375 \nQ 47.125 39.59375 43.09375 43.984375 \nQ 39.0625 48.390625 32.421875 48.390625 \nQ 24.90625 48.390625 20.390625 44.140625 \nQ 15.875 39.890625 15.1875 32.171875 \nz\n\" id=\"DejaVuSans-65\"/>\n      <path d=\"M 18.109375 8.203125 \nL 18.109375 -20.796875 \nL 9.078125 -20.796875 \nL 9.078125 54.6875 \nL 18.109375 54.6875 \nL 18.109375 46.390625 \nQ 20.953125 51.265625 25.265625 53.625 \nQ 29.59375 56 35.59375 56 \nQ 45.5625 56 51.78125 48.09375 \nQ 58.015625 40.1875 58.015625 27.296875 \nQ 58.015625 14.40625 51.78125 6.484375 \nQ 45.5625 -1.421875 35.59375 -1.421875 \nQ 29.59375 -1.421875 25.265625 0.953125 \nQ 20.953125 3.328125 18.109375 8.203125 \nz\nM 48.6875 27.296875 \nQ 48.6875 37.203125 44.609375 42.84375 \nQ 40.53125 48.484375 33.40625 48.484375 \nQ 26.265625 48.484375 22.1875 42.84375 \nQ 18.109375 37.203125 18.109375 27.296875 \nQ 18.109375 17.390625 22.1875 11.75 \nQ 26.265625 6.109375 33.40625 6.109375 \nQ 40.53125 6.109375 44.609375 11.75 \nQ 48.6875 17.390625 48.6875 27.296875 \nz\n\" id=\"DejaVuSans-70\"/>\n      <path d=\"M 30.609375 48.390625 \nQ 23.390625 48.390625 19.1875 42.75 \nQ 14.984375 37.109375 14.984375 27.296875 \nQ 14.984375 17.484375 19.15625 11.84375 \nQ 23.34375 6.203125 30.609375 6.203125 \nQ 37.796875 6.203125 41.984375 11.859375 \nQ 46.1875 17.53125 46.1875 27.296875 \nQ 46.1875 37.015625 41.984375 42.703125 \nQ 37.796875 48.390625 30.609375 48.390625 \nz\nM 30.609375 56 \nQ 42.328125 56 49.015625 48.375 \nQ 55.71875 40.765625 55.71875 27.296875 \nQ 55.71875 13.875 49.015625 6.21875 \nQ 42.328125 -1.421875 30.609375 -1.421875 \nQ 18.84375 -1.421875 12.171875 6.21875 \nQ 5.515625 13.875 5.515625 27.296875 \nQ 5.515625 40.765625 12.171875 48.375 \nQ 18.84375 56 30.609375 56 \nz\n\" id=\"DejaVuSans-6f\"/>\n      <path d=\"M 48.78125 52.59375 \nL 48.78125 44.1875 \nQ 44.96875 46.296875 41.140625 47.34375 \nQ 37.3125 48.390625 33.40625 48.390625 \nQ 24.65625 48.390625 19.8125 42.84375 \nQ 14.984375 37.3125 14.984375 27.296875 \nQ 14.984375 17.28125 19.8125 11.734375 \nQ 24.65625 6.203125 33.40625 6.203125 \nQ 37.3125 6.203125 41.140625 7.25 \nQ 44.96875 8.296875 48.78125 10.40625 \nL 48.78125 2.09375 \nQ 45.015625 0.34375 40.984375 -0.53125 \nQ 36.96875 -1.421875 32.421875 -1.421875 \nQ 20.0625 -1.421875 12.78125 6.34375 \nQ 5.515625 14.109375 5.515625 27.296875 \nQ 5.515625 40.671875 12.859375 48.328125 \nQ 20.21875 56 33.015625 56 \nQ 37.15625 56 41.109375 55.140625 \nQ 45.0625 54.296875 48.78125 52.59375 \nz\n\" id=\"DejaVuSans-63\"/>\n      <path d=\"M 54.890625 33.015625 \nL 54.890625 0 \nL 45.90625 0 \nL 45.90625 32.71875 \nQ 45.90625 40.484375 42.875 44.328125 \nQ 39.84375 48.1875 33.796875 48.1875 \nQ 26.515625 48.1875 22.3125 43.546875 \nQ 18.109375 38.921875 18.109375 30.90625 \nL 18.109375 0 \nL 9.078125 0 \nL 9.078125 75.984375 \nL 18.109375 75.984375 \nL 18.109375 46.1875 \nQ 21.34375 51.125 25.703125 53.5625 \nQ 30.078125 56 35.796875 56 \nQ 45.21875 56 50.046875 50.171875 \nQ 54.890625 44.34375 54.890625 33.015625 \nz\n\" id=\"DejaVuSans-68\"/>\n      <path d=\"M 44.28125 53.078125 \nL 44.28125 44.578125 \nQ 40.484375 46.53125 36.375 47.5 \nQ 32.28125 48.484375 27.875 48.484375 \nQ 21.1875 48.484375 17.84375 46.4375 \nQ 14.5 44.390625 14.5 40.28125 \nQ 14.5 37.15625 16.890625 35.375 \nQ 19.28125 33.59375 26.515625 31.984375 \nL 29.59375 31.296875 \nQ 39.15625 29.25 43.1875 25.515625 \nQ 47.21875 21.78125 47.21875 15.09375 \nQ 47.21875 7.46875 41.1875 3.015625 \nQ 35.15625 -1.421875 24.609375 -1.421875 \nQ 20.21875 -1.421875 15.453125 -0.5625 \nQ 10.6875 0.296875 5.421875 2 \nL 5.421875 11.28125 \nQ 10.40625 8.6875 15.234375 7.390625 \nQ 20.0625 6.109375 24.8125 6.109375 \nQ 31.15625 6.109375 34.5625 8.28125 \nQ 37.984375 10.453125 37.984375 14.40625 \nQ 37.984375 18.0625 35.515625 20.015625 \nQ 33.0625 21.96875 24.703125 23.78125 \nL 21.578125 24.515625 \nQ 13.234375 26.265625 9.515625 29.90625 \nQ 5.8125 33.546875 5.8125 39.890625 \nQ 5.8125 47.609375 11.28125 51.796875 \nQ 16.75 56 26.8125 56 \nQ 31.78125 56 36.171875 55.265625 \nQ 40.578125 54.546875 44.28125 53.078125 \nz\n\" id=\"DejaVuSans-73\"/>\n     </defs>\n     <g transform=\"translate(125.295312 174.876562)scale(0.1 -0.1)\">\n      <use xlink:href=\"#DejaVuSans-65\"/>\n      <use x=\"61.523438\" xlink:href=\"#DejaVuSans-70\"/>\n      <use x=\"125\" xlink:href=\"#DejaVuSans-6f\"/>\n      <use x=\"186.181641\" xlink:href=\"#DejaVuSans-63\"/>\n      <use x=\"241.162109\" xlink:href=\"#DejaVuSans-68\"/>\n      <use x=\"304.541016\" xlink:href=\"#DejaVuSans-73\"/>\n     </g>\n    </g>\n   </g>\n   <g id=\"matplotlib.axis_2\">\n    <g id=\"ytick_1\">\n     <g id=\"line2d_7\">\n      <defs>\n       <path d=\"M 0 0 \nL -3.5 0 \n\" id=\"mc00e681b5e\" style=\"stroke:#000000;stroke-width:0.8;\"/>\n      </defs>\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"45.478125\" xlink:href=\"#mc00e681b5e\" y=\"67.612877\"/>\n      </g>\n     </g>\n     <g id=\"text_8\">\n      <!-- $\\mathdefault{10^{0}}$ -->\n      <g transform=\"translate(20.878125 71.412096)scale(0.1 -0.1)\">\n       <use transform=\"translate(0 0.765625)\" xlink:href=\"#DejaVuSans-31\"/>\n       <use transform=\"translate(63.623047 0.765625)\" xlink:href=\"#DejaVuSans-30\"/>\n       <use transform=\"translate(128.203125 39.046875)scale(0.7)\" xlink:href=\"#DejaVuSans-30\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_2\">\n     <g id=\"line2d_8\">\n      <defs>\n       <path d=\"M 0 0 \nL -2 0 \n\" id=\"macd3912657\" style=\"stroke:#000000;stroke-width:0.6;\"/>\n      </defs>\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.6;\" x=\"45.478125\" xlink:href=\"#macd3912657\" y=\"132.079928\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_3\">\n     <g id=\"line2d_9\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.6;\" x=\"45.478125\" xlink:href=\"#macd3912657\" y=\"115.838767\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_4\">\n     <g id=\"line2d_10\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.6;\" x=\"45.478125\" xlink:href=\"#macd3912657\" y=\"104.315481\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_5\">\n     <g id=\"line2d_11\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.6;\" x=\"45.478125\" xlink:href=\"#macd3912657\" y=\"95.377325\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_6\">\n     <g id=\"line2d_12\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.6;\" x=\"45.478125\" xlink:href=\"#macd3912657\" y=\"88.07432\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_7\">\n     <g id=\"line2d_13\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.6;\" x=\"45.478125\" xlink:href=\"#macd3912657\" y=\"81.899717\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_8\">\n     <g id=\"line2d_14\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.6;\" x=\"45.478125\" xlink:href=\"#macd3912657\" y=\"76.551033\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_9\">\n     <g id=\"line2d_15\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.6;\" x=\"45.478125\" xlink:href=\"#macd3912657\" y=\"71.833159\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_10\">\n     <g id=\"line2d_16\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.6;\" x=\"45.478125\" xlink:href=\"#macd3912657\" y=\"39.84843\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_11\">\n     <g id=\"line2d_17\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.6;\" x=\"45.478125\" xlink:href=\"#macd3912657\" y=\"23.607269\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_12\">\n     <g id=\"line2d_18\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.6;\" x=\"45.478125\" xlink:href=\"#macd3912657\" y=\"12.083982\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"text_9\">\n     <!-- rmse -->\n     <defs>\n      <path d=\"M 41.109375 46.296875 \nQ 39.59375 47.171875 37.8125 47.578125 \nQ 36.03125 48 33.890625 48 \nQ 26.265625 48 22.1875 43.046875 \nQ 18.109375 38.09375 18.109375 28.8125 \nL 18.109375 0 \nL 9.078125 0 \nL 9.078125 54.6875 \nL 18.109375 54.6875 \nL 18.109375 46.1875 \nQ 20.953125 51.171875 25.484375 53.578125 \nQ 30.03125 56 36.53125 56 \nQ 37.453125 56 38.578125 55.875 \nQ 39.703125 55.765625 41.0625 55.515625 \nz\n\" id=\"DejaVuSans-72\"/>\n      <path d=\"M 52 44.1875 \nQ 55.375 50.25 60.0625 53.125 \nQ 64.75 56 71.09375 56 \nQ 79.640625 56 84.28125 50.015625 \nQ 88.921875 44.046875 88.921875 33.015625 \nL 88.921875 0 \nL 79.890625 0 \nL 79.890625 32.71875 \nQ 79.890625 40.578125 77.09375 44.375 \nQ 74.3125 48.1875 68.609375 48.1875 \nQ 61.625 48.1875 57.5625 43.546875 \nQ 53.515625 38.921875 53.515625 30.90625 \nL 53.515625 0 \nL 44.484375 0 \nL 44.484375 32.71875 \nQ 44.484375 40.625 41.703125 44.40625 \nQ 38.921875 48.1875 33.109375 48.1875 \nQ 26.21875 48.1875 22.15625 43.53125 \nQ 18.109375 38.875 18.109375 30.90625 \nL 18.109375 0 \nL 9.078125 0 \nL 9.078125 54.6875 \nL 18.109375 54.6875 \nL 18.109375 46.1875 \nQ 21.1875 51.21875 25.484375 53.609375 \nQ 29.78125 56 35.6875 56 \nQ 41.65625 56 45.828125 52.96875 \nQ 50 49.953125 52 44.1875 \nz\n\" id=\"DejaVuSans-6d\"/>\n     </defs>\n     <g transform=\"translate(14.798437 91.25625)rotate(-90)scale(0.1 -0.1)\">\n      <use xlink:href=\"#DejaVuSans-72\"/>\n      <use x=\"41.097656\" xlink:href=\"#DejaVuSans-6d\"/>\n      <use x=\"138.509766\" xlink:href=\"#DejaVuSans-73\"/>\n      <use x=\"190.609375\" xlink:href=\"#DejaVuSans-65\"/>\n     </g>\n    </g>\n   </g>\n   <g id=\"line2d_19\">\n    <path clip-path=\"url(#p68187e5238)\" d=\"M 54.355398 16.877273 \nL 56.148786 25.47152 \nL 57.942175 31.482456 \nL 59.735563 36.302936 \nL 61.528951 40.449395 \nL 63.32234 44.145911 \nL 65.115728 47.538424 \nL 66.909117 50.681314 \nL 68.702505 53.651717 \nL 70.495894 56.48543 \nL 72.289282 59.175629 \nL 74.08267 61.785748 \nL 75.876059 64.309709 \nL 77.669447 66.771417 \nL 79.462836 69.17089 \nL 81.256224 71.510398 \nL 83.049613 73.826902 \nL 84.843001 76.080104 \nL 86.636389 78.316243 \nL 88.429778 80.520507 \nL 90.223166 82.684059 \nL 92.016555 84.830966 \nL 93.809943 86.962018 \nL 95.603332 89.043725 \nL 97.39672 91.128537 \nL 99.190108 93.181479 \nL 100.983497 95.213168 \nL 102.776885 97.235977 \nL 104.570274 99.218805 \nL 106.363662 101.160318 \nL 108.157051 103.121185 \nL 109.950439 105.029522 \nL 111.743827 106.912634 \nL 113.537216 108.740661 \nL 115.330604 110.563842 \nL 117.123993 112.330438 \nL 118.917381 114.099295 \nL 120.71077 115.774184 \nL 122.504158 117.3908 \nL 124.297546 119.014116 \nL 126.090935 120.531117 \nL 127.884323 122.026884 \nL 129.677712 123.446687 \nL 131.4711 124.7716 \nL 133.264489 126.036052 \nL 135.057877 127.251232 \nL 136.851265 128.424828 \nL 138.644654 129.488898 \nL 140.438042 130.483357 \nL 142.231431 131.411928 \nL 144.024819 132.266956 \nL 145.818208 133.075916 \nL 147.611596 133.802224 \nL 149.404985 134.469381 \nL 151.198373 135.073989 \nL 152.991761 135.61517 \nL 154.78515 136.114425 \nL 156.578538 136.572314 \nL 158.371927 136.955301 \nL 160.165315 137.303217 \nL 161.958704 137.620039 \nL 163.752092 137.88532 \nL 165.54548 138.130062 \nL 167.338869 138.347507 \nL 169.132257 138.532575 \nL 170.925646 138.695886 \nL 172.719034 138.839339 \nL 174.512423 138.96518 \nL 176.305811 139.070983 \nL 178.099199 139.167751 \nL 179.892588 139.243667 \nL 181.685976 139.309029 \nL 183.479365 139.357266 \nL 185.272753 139.414169 \nL 187.066142 139.441639 \nL 188.85953 139.483663 \nL 190.652918 139.525447 \nL 192.446307 139.541321 \nL 194.239695 139.574564 \nL 196.033084 139.601831 \nL 197.826472 139.604856 \nL 199.619861 139.645227 \nL 201.413249 139.66684 \nL 203.206637 139.690821 \nL 205.000026 139.715665 \nL 206.793414 139.715095 \nL 208.586803 139.745385 \nL 210.380191 139.791965 \nL 212.17358 139.805568 \nL 213.966968 139.847395 \nL 215.760356 139.88246 \nL 217.553745 139.953415 \nL 219.347133 139.991354 \nL 221.140522 140.057912 \nL 222.93391 140.07718 \nL 224.727299 140.09579 \nL 226.520687 140.231715 \nL 228.314075 140.30559 \nL 230.107464 140.345757 \nL 231.900852 140.422727 \n\" style=\"fill:none;stroke:#1f77b4;stroke-linecap:square;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"patch_3\">\n    <path d=\"M 45.478125 146.6 \nL 45.478125 10.7 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_4\">\n    <path d=\"M 240.778125 146.6 \nL 240.778125 10.7 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_5\">\n    <path d=\"M 45.478125 146.6 \nL 240.778125 146.6 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_6\">\n    <path d=\"M 45.478125 10.7 \nL 240.778125 10.7 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n  </g>\n </g>\n <defs>\n  <clipPath id=\"p68187e5238\">\n   <rect height=\"135.9\" width=\"195.3\" x=\"45.478125\" y=\"10.7\"/>\n  </clipPath>\n </defs>\n</svg>\n",
   "text/plain": "<Figure size 252x180 with 1 Axes>"
  },
  "metadata": {
   "needs_background": "light"
  },
  "output_type": "display_data"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "train rmse 0.162396\n"
 }
]
```

上述代码执行完之后会生成一个submission.csv文件。这个文件是符合Kaggle比赛要求的提交格式的。这时，我们可以在Kaggle上提交我们预测得出的结果，并且查看与测试数据集上真实房价（标签）的误差。具体来说有以下几个步骤：登录Kaggle网站，访问房价预测比赛网页，并点击右侧“Submit Predictions”或“Late Submission”按钮；然后，点击页面下方“Upload Submission File”图标所在的虚线框选择需要提交的预测结果文件；最后，点击页面最下方的“Make Submission”按钮就可以查看结果了，如图3.9所示。

![Kaggle预测房价比赛的预测结果提交页面](../img/kaggle_submit2.png)


## 小结

* 通常需要对真实数据做预处理。
* 可以使用$K$折交叉验证来选择模型并调节超参数。


## 练习

* 在Kaggle提交本节的预测结果。观察一下，这个结果在Kaggle上能拿到什么样的分数？
* 对照$K$折交叉验证结果，不断修改模型（例如添加隐藏层）和调参，能提高Kaggle上的分数吗？
* 如果不使用本节中对连续数值特征的标准化处理，结果会有什么变化？
* 扫码直达讨论区，在社区交流方法和结果。你能发掘出其他更好的技巧吗？ 


## 扫码直达[讨论区](https://discuss.gluon.ai/t/topic/1039)

![](../img/qr_kaggle-house-price.svg)
