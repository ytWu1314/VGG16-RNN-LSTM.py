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

（3）LSTM长短期记忆网络
长短期记忆网络（LSTM，Long Short-Term Memory）是一种时间循环神经网络，是为了解决一般的RNN（循环神经网络）存在的长期依赖问题而专门设计出来的，所有的RNN都具有一种重复神经网络模块的链式形式。相比普通的RNN，LSTM能够在更长的序列中有更好的表现。

