# 基于注意力机制的图像字幕生成

##  一、项目任务
图像字幕对于人们理解图像是一项具有挑战性的工作。图像字幕的任务是用自然语言描述来描述输入图像。它有许多实际应用，如帮助盲人理解图像的内容，拍照片就可以马上生成合适的描述文字，省去了用户手动编辑。图像字幕是一个具有挑战性的领域，不仅需要对图像中的物体进行描述，还需要用类似人的句子来表达信息。本项目的主要任务就是使机器识别出与人相似的图像和真实文字。

## 二、技术方案

![图片信息描述](https://github.com/ytWu1314/VGG16-RNN-LSTM.py/blob/master/image/image1.png)

本项目拟采用CNN卷积神经网络提取图片特征+RNN循环网络对提取的特征进行建模从而实现单词的表示与生成。首先在ImageNet数据集上预训练神经卷积网络（采用VGG16卷积神经网络）作为特征提取器，从而提取图像的特征信息，然后通过循环神经网络LSTM对描述句子进行建模从而实现了对单词的表示与生成，最后通过全连接网络将卷积神经网络的结果和LSTM网络连接起来形成一个可以采用端到端训练的图像字幕生成模型。本项目拟采用Tensorflow框架实现和训练该模型，并使用Filckr8K数据集进行验证。

相关技术及数据集：

（1）Flickr8K数据集
   该数据集的来源是雅虎相册，共有大概8000张图像。这些图像的内容大多展示的是人类与动物参与到某项活动中的情景。每张图像对应的人工标注大概是5句英文字母描述。数据集也是按照标准的训练集，验证集和测试集来进行分块的。
Eg:

![image2](https://github.com/ytWu1314/VGG16-RNN-LSTM.py/blob/master/image/image2.png)

`0 A child in a pink dress is climbing up a set of stairs in an entry way .`

`1 A girl going into a wooden building .`

`2 A little girl climbing into a wooden playhouse .`

`3 A little girl climbing the stairs to her playhouse .`

`4 A little girl in a pink dress going into a wooden cabin .`

(2)VGG16预训练模型

![image3](https://github.com/ytWu1314/VGG16-RNN-LSTM.py/blob/master/image/image3.png)


24x224x3的彩色图表示3通道的长和宽都为224的图像数据，也是网络的输入层白色部分为卷积层，红色部分为池化层（使用最大池化），蓝色部分为全连接层，其中卷积层和全连接层的激活函数都使用RELU函数。总的来说，VGG16网络为13层卷积层+3层全连接层而组成。具体结构如下：


![image6](https://github.com/ytWu1314/VGG16-RNN-LSTM.py/blob/master/image/image6.png)

（3）LSTM长短期记忆网络
长短期记忆网络（LSTM，Long Short-Term Memory）是一种时间循环神经网络，是为了解决一般的RNN（循环神经网络）存在的长期依赖问题而专门设计出来的，所有的RNN都具有一种重复神经网络模块的链式形式。相比普通的RNN，LSTM能够在更长的序列中有更好的表现。

![image4](https://github.com/ytWu1314/VGG16-RNN-LSTM.py/blob/master/image/image4.png)

![image7](https://github.com/ytWu1314/VGG16-RNN-LSTM.py/blob/master/image/image7.png)

![image8](https://github.com/ytWu1314/VGG16-RNN-LSTM.py/blob/master/image/image8.png)

## 实验方案
(1) 导入所需的库

``` python
import numpy as np
import pandas as pd
from numpy import array
from pickle import load
import string

from PIL import Image
import pickle
from collections import Counter
import matplotlib.pyplot as plt
 
import sys, time, os, warnings
warnings.filterwarnings("ignore")
import re
 
import keras
import tensorflow as tf
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
 
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, BatchNormalization
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.text import Tokenizer
from keras.applications.vgg16 import VGG16, preprocess_input
 
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
```
（2）数据加载和预处理
定义图像和字幕路径，并检查数据集中总共有多少图像。

```python
image_path = "/content/gdrive/My Drive/FLICKR8K/Flicker8k_Dataset"
dir_Flickr_text = "/content/gdrive/My Drive/FLICKR8K/Flickr8k_text/Flickr8k.token.txt"
jpgs = os.listdir(image_path)
 
print("Total Images in Dataset = {}".format(len(jpgs)))
```
得出16182个图像

（3）创建一个数据框来存储图像ID和标题，可视化一些图片及其5个标题

```python
image_path = "/content/gdrive/My Drive/FLICKR8K/Flicker8k_Dataset"
dir_Flickr_text = "/content/gdrive/My Drive/FLICKR8K/Flickr8k_text/Flickr8k.token.txt"
jpgs = os.listdir(image_path)

npic = 5
npix = 224
target_size = (npix,npix,3)
count = 1
 
fig = plt.figure(figsize=(10,20))
for jpgfnm in uni_filenames[10:14]:
   filename = image_path + '/' + jpgfnm
   captions = list(data["caption"].loc[data["filename"]==jpgfnm].values)
   image_load = load_img(filename, target_size=target_size)
   ax = fig.add_subplot(npic,2,count,xticks=[],yticks=[])
   ax.imshow(image_load)
   count += 1
 
   ax = fig.add_subplot(npic,2,count)
   plt.axis('off')
   ax.plot()
   ax.set_xlim(0,1)
   ax.set_ylim(0,len(captions))
   for i, caption in enumerate(captions):
       ax.text(0,i,caption,fontsize=20)
   count += 1
plt.show()
```

输出如下:

![image5](https://github.com/ytWu1314/VGG16-RNN-LSTM.py/blob/master/image/image5.png)

（4）执行一些文本清理，例如删除标点符号，单个字符和数字值，得到词汇量

```pyhton
vocabulary = []
for txt in data.caption.values:
   vocabulary.extend(txt.split())
print('Vocabulary Size: %d' % len(set(vocabulary)))

def remove_punctuation(text_original):
   text_no_punctuation = text_original.translate(string.punctuation)
   return(text_no_punctuation)
 
def remove_single_character(text):
   text_len_more_than1 = ""
   for word in text.split():
       if len(word) > 1:
           text_len_more_than1 += " " + word
   return(text_len_more_than1)
 
def remove_numeric(text):
   text_no_numeric = ""
   for word in text.split():
       isalpha = word.isalpha()
       if isalpha:
           text_no_numeric += " " + word
   return(text_no_numeric)
 
def text_clean(text_original):
   text = remove_punctuation(text_original)
   text = remove_single_character(text)
   text = remove_numeric(text)
   return(text)
 
for i, caption in enumerate(data.caption.values):
   newcaption = text_clean(caption)
   data["caption"].iloc[i] = newcaption
```

（5）VGG模型定义
使用VGG16定义图像特征提取模型。这里不需要分类图像，只需要为图像提取图像矢量即可。因此，从模型中删除了softmax层。我们先将所有图像预处理为相同大小，即224×224，然后再将其输入模型。

```python
def load_image(image_path):
   img = tf.io.read_file(image_path)
   img = tf.image.decode_jpeg(img, channels=3)
   img = tf.image.resize(img, (224, 224))
   img = preprocess_input(img)
   return img, image_path
 
image_model = tf.keras.applications.VGG16(include_top=False, weights='imagenet')
new_input = image_model.input
hidden_layer = image_model.layers[-1].output
image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
 
image_features_extract_model.summary()
```
得到：

![image](https://github.com/ytWu1314/VGG16-RNN-LSTM.py/blob/master/image/image9.png)


（6）利用LSTM对语言序列进行处理

#### 四、参考文献

[1] 图像物体分类与检测算法综述[J]. 黄凯奇,任伟强,谭铁牛. 计算机学报. 2014(06) 
[2] 图像特征提取方法的综述[J]. 王志瑞,闫彩良. 吉首大学学报(自然科学版). 2011(05) 
[3] 机器学习及其相关算法综述[J]. 陈凯,朱钰. 统计与信息论坛. 2007(05) 
[4] 大数据下的机器学习算法综述[J]. 何清,李宁,罗文娟,史忠植. 模式识别与人工智能. 2014(04) 
[5] 王剑锋.基于深度学习的图像字幕生成方法研究[D].上海:上海师范大学,2018 
[6] 王许.基于场景的图像语义描述生成技术研究[D].贵阳:贵州大学,2020
[7] 于东飞.基于注意力机制与高层语义的视觉问答研究[D].安徽.中国科学技术大学，2019 



