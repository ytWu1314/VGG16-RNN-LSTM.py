import re
import keras
import pickle
import string
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tqdm import tqdm
from numpy import array
from pickle import load
import sys, time, os, warnings
warnings.filterwarnings("ignore")
from collections import Counter
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, BatchNormalization
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.text import Tokenizer
from keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


# 使用GPU版TensorFlow，并且在显卡高占用率的情况下训练模型，要注意在初始化 Session 的时候为其分配固定数量的显存
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)


# 定义图像和字幕路径，并检查数据集中总共有多少图像
image_path = "E:\BaiduNetdiskDownload\Flickr8k and Flickr8kCN\\Flicker8k_Dataset"  #图片文件路径
dir_Flickr_text = "E:\BaiduNetdiskDownload\Flickr8k and Flickr8kCN\\Flickr8k.token.txt"  #标注文件路径
jpgs = os.listdir(image_path)
#print("Total Images in Dataset = {}".format(len(jpgs)))  #


# 创建一个数据框来存储图像ID和标题，以便于使用
file = open(dir_Flickr_text,'r')
text = file.read()
file.close()

datatxt = []
for line in text.split('\n'):
    col = line.split('\t')
    if len(col) == 1:
        continue
    w = col[0].split("#")    #['1000268201_693b08cb0e.jpg', 0]
    datatxt.append(w + [col[1].lower()])  #['1000268201_693b08cb0e.jpg', 0, 'A child in a pink dress is climbing up a set of stairs in an entry way .']

data = pd.DataFrame(datatxt, columns=["filename","index","caption"])
data = data.reindex(columns =['index','filename','caption'])  #列重排
data = data[data.filename != '2258277193_586949ec62.jpg.1']
uni_filenames = np.unique(data.filename.values)  #不重复的文件名列表
#print(uni_filenames)
#print(data)

"""
npic = 5
npix = 224
count = 1
target_size = (npix, npix, 3)
fig = plt.figure(figsize = (15,18))
for jpgfnm in uni_filenames[10:14]:
    filename = image_path + '/' + jpgfnm  #具体的图像文件名称
    captions = list(data["caption"].loc[data["filename"]==jpgfnm].values)
    image_load = load_img(filename, target_size = target_size)
    ax = fig.add_subplot(npic, 2, count, xticks=[], yticks=[])
    ax.imshow(image_load)
    count += 1

    ax = fig.add_subplot(npic, 2, count)
    plt.axis('off')
    ax.plot()
    ax.set_xlim(0,1)
    ax.set_ylim(0,len(captions))
    for i, caption in enumerate(captions):
        ax.text(0,i,caption,fontsize = 10)
    count += 1
#plt.show()
"""


# 当前的词汇量
vocabulary = []
for txt in data.caption.values:
    vocabulary.extend(txt.split())  #分割每一句标注并加入字典
#print('Vocabulary Size: %d' % len(set(vocabulary)))  #输出现有词汇量

# 文本清理(删除标点符号，单个字符和数字值)
def remove_punctuation(text_original):  #移除标点符号
    text_no_punctuation = text_original.translate(string.punctuation)
    return(text_no_punctuation)

def remove_single_character(text):  #移除单个字符
    text_len_more_than1 = ""
    for word in text.split():
        if len(word) > 1:  #判断长度
            text_len_more_than1 += " " + word
    return(text_len_more_than1)

def remove_numeric(text):  #移除数字值
     text_no_numeric = ""
     for word in text.split():
           isalpha = word.isalpha()  #判断字符是否为英文字母
           if isalpha:
                 text_no_numeric += " " + word
     return(text_no_numeric)
 
def text_clean(text_original):  #调用以上三个函数，执行文本清理
     text = remove_punctuation(text_original)
     text = remove_single_character(text)
     text = remove_numeric(text)
     return(text)
 
for i, caption in enumerate(data.caption.values):
     newcaption = text_clean(caption)
     data["caption"].iloc[i] = newcaption  #替换成清理后的文本
     
     
# 清理后的词汇量
clean_vocabulary = []
for txt in data.caption.values:
   clean_vocabulary.extend(txt.split())
#print('Clean Vocabulary Size: %d' % len(set(clean_vocabulary)))


PATH = "E:\BaiduNetdiskDownload\Flickr8k and Flickr8kCN\Flicker8k_Dataset\\"  #将所有标题和图像路径保存在两个列表中

all_captions = []  #保存全部字幕
for caption in data["caption"].astype(str):
   caption = '<start> ' + caption+ ' <end>'  #向每个字幕添加“ <开始>”和“ <结束>”标签,以便模型可以理解每个字幕的开始和结束
   all_captions.append(caption)
#print(all_captions[:10])

all_img_name_vector = []  #保存全部图片
for annot in data["filename"]:
   full_image_path = PATH + annot
   all_img_name_vector.append(full_image_path)
#print(all_img_name_vector[:10])


#  现在可以看到有40455个图像路径和标题
print(f"len(all_captions) : {len(all_captions)}")  #40455个标题
print(f"len(all_img_name_vector) : {len(all_img_name_vector)}")  #40455个图像路径


#  定义将数据集限制为40000个图像和标题的函数
def data_limiter(num,total_captions,all_img_name_vector):
    train_captions, img_name_vector = shuffle(total_captions,all_img_name_vector,random_state=1)
    train_captions = train_captions[:num]
    img_name_vector = img_name_vector[:num]
    return train_captions, img_name_vector

train_captions,img_name_vector = data_limiter(40000,all_captions,all_img_name_vector)


#  图片预处理
def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)  #将表示一张图像的三维矩阵重新按照jpeg格式编码并存入文件中
    img = tf.image.resize(img, (224, 224))  #所有图像预处理为相同大小，即224×224
    img = preprocess_input(img)  #归一化，对每个通道减均值
    return img, image_path


#  使用迁移学习构建新模型
# 最后一层卷积输入shape(8*8*2048),并将结果向量保存为dict
image_model = tf.keras.applications.VGG16(include_top=False, weights='imagenet')  #不使用最后全连接层,VGG16模型的训练集是imagenet
new_input = image_model.input  #shape:(batch_size,224,244,3)
hidden_layer = image_model.layers[-1].output  #hidden_layer shape:(batch_size,7,7,512)
image_features_extract_model = tf.keras.Model(new_input, hidden_layer)  #创建新模型
image_features_extract_model.summary()  #模型视图


#  保存通过使用VGG16获得的特征,将每个图片名称映射到要加载的图片
encode_train = sorted(set(img_name_vector))
image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)  #切分传入Tensor的第一个维度，生成相应的dataset
#map:可以并行处理数据，默认读取的文件具有确定性顺序
image_dataset = image_dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(64)   #从图像文件名中解析像素值。使用多线程提升预处理的速度

for img, path in tqdm(image_dataset):  #VGG16得到的feature
    batch_features = image_features_extract_model(img)
    batch_features = tf.reshape(batch_features,(batch_features.shape[0], -1, batch_features.shape[3]))
    #shape:(batch_size,7,7,512) reshape (batch_size,49,512)

    for bf, p in zip(batch_features, path):
        path_of_feature = p.numpy().decode("utf-8")
        np.save(path_of_feature, bf.numpy())  #保存，后缀为.npy,包括dtype和shape信息


# 标记标题，并为数据中所有唯一的单词建立词汇表
top_k = 5000
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,  #将词汇量限制在前5000个单词以节省内存
                                                  oov_token="<unk>",  #字典中没有的字符用<unk>代替
                                                  filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')  #需要过滤掉的特殊字符
 
tokenizer.fit_on_texts(train_captions)  #要用以训练的文本列表(分词器)
train_seqs = tokenizer.texts_to_sequences(train_captions)  #转为序列列表向量
tokenizer.word_index['<pad>'] = 0
tokenizer.index_word[0] = '<pad>'

train_seqs = tokenizer.texts_to_sequences(train_captions)
cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')  #如果没有指定最大长度，pad_sequences会自动计算最大长度


#print(train_captions[:3])
#print(train_seqs[:3])


# 计算所有字幕的最大和最小长度
def calc_max_length(tensor):
    return max(len(t) for t in tensor)
max_length = calc_max_length(train_seqs)
 
def calc_min_length(tensor):
    return min(len(t) for t in tensor)
min_length = calc_min_length(train_seqs)
 
print('Max Length of any caption : Min Length of any caption = '+ str(max_length) +" : "+str(min_length))  #33:2


# 使用80-20拆分创建训练和验证集
img_name_train, img_name_val, cap_train, cap_val = train_test_split(img_name_vector,cap_vector,
                                                                    test_size=0.2,   #验证数据集占20%
                                                                    random_state=0)  #确保每次数据一致


# 定义训练参数
BATCH_SIZE = 64
BUFFER_SIZE = 1000  #shuffle缓冲区大小
embedding_dim = 256  #词嵌入维度
units = 512
vocab_size = len(tokenizer.word_index) + 1
num_steps = len(img_name_train) // BATCH_SIZE
features_shape = 512
attention_features_shape = 49  #后面会将(7,7,512)转为(49,512)


# 加载保存的之前feature文件
def map_func(img_name, cap):
    img_tensor = np.load(img_name.decode('utf-8')+'.npy')
    return img_tensor, cap
dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))
 
# Use map to load the numpy files in parallel
dataset = dataset.map(lambda item1, item2: tf.numpy_function(map_func, [item1, item2], [tf.float32, tf.int32]),num_parallel_calls=tf.data.experimental.AUTOTUNE)
 
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)  #prefetch 可以合理利用CPU准备数据，GPU计算数据之间的空闲时间，加快数据读取


# 定义VGG-16编码器,一层使用relu的全连接层
class VGG16_Encoder(tf.keras.Model):
    # This encoder passes the features through a Fully connected layer
    def __init__(self, embedding_dim):
        super(VGG16_Encoder, self).__init__()
        # shape after fc == (batch_size, 49, embedding_dim)
        self.fc = tf.keras.layers.Dense(embedding_dim)
        self.dropout = tf.keras.layers.Dropout(0.5, noise_shape=None, seed=None)

    def call(self, x):
        #x= self.dropout(x)
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x


# 基于GPU/CPU功能定义RNN
def rnn_type(units):
    if tf.test.is_gpu_available():
        return tf.compat.v1.keras.layers.CuDNNLSTM(units,return_sequences=True,return_state=True,recurrent_initializer='glorot_uniform')
    else:
        return tf.keras.layers.GRU(units,return_sequences=True,return_state=True,recurrent_activation='sigmoid',recurrent_initializer='glorot_uniform')


'''The encoder output(i.e. 'features'), hidden state(initialized to 0)(i.e. 'hidden') and
the decoder input (which is the start token)(i.e. 'x') is passed to the decoder.'''


# 使用注意力模型
class Rnn_Local_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(Rnn_Local_Decoder, self).__init__()
        #词嵌入将高维离散数据转为低维连续数据，并表现出数据之间的相似性（向量空间）
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.units,
                                  return_sequences=True,
                                  return_state=True,
                                  recurrent_initializer='glorot_uniform')
  
        self.fc1 = tf.keras.layers.Dense(self.units)
 
        self.dropout = tf.keras.layers.Dropout(0.5, noise_shape=None, seed=None)
        self.batchnormalization = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)
 
        self.fc2 = tf.keras.layers.Dense(vocab_size)
 
        # Implementing Attention Mechanism
        self.Uattn = tf.keras.layers.Dense(units)
        self.Wattn = tf.keras.layers.Dense(units)
        self.Vattn = tf.keras.layers.Dense(1)
 
    def call(self, x, features, hidden):   # 获取注意力模型输出
        # features shape ==> (64,49,256) ==> Output from ENCODER
        # hidden shape == (batch_size, hidden_size) ==>(64,512)
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size) ==> (64,1,512)
 
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
 
        # score shape == (64, 49, 1)
        # Attention Function
        '''e(ij) = f(s(t-1),h(j))'''
        ''' e(ij) = Vattn(T)*tanh(Uattn * h(j) + Wattn * s(t))'''
 
        score = self.Vattn(tf.nn.tanh(self.Uattn(features) + self.Wattn(hidden_with_time_axis)))
 
        # self.Uattn(features) : (64,49,512)
        # self.Wattn(hidden_with_time_axis) : (64,1,512)
        # tf.nn.tanh(self.Uattn(features) + self.Wattn(hidden_with_time_axis)) : (64,49,512)
        # self.Vattn(tf.nn.tanh(self.Uattn(features) + self.Wattn(hidden_with_time_axis))) : (64,49,1) ==> score
 
        # you get 1 at the last axis because you are applying score to self.Vattn
        # Then find Probability using Softmax
        '''attention_weights(alpha(ij)) = softmax(e(ij))'''
 
        attention_weights = tf.nn.softmax(score, axis=1)
 
        # attention_weights shape == (64, 49, 1)
        # Give weights to the different pixels in the image
        ''' C(t) = Summation(j=1 to T) (attention_weights * VGG-16 features) '''

        context_vector = attention_weights * features  #获取注意力模型输出
        context_vector = tf.reduce_sum(context_vector, axis=1)
 
        # Context Vector(64,256) = AttentionWeights(64,49,1) * features(64,49,256)
        # context_vector shape after sum == (64, 256)
        # x shape after passing through embedding == (64, 1, 256)
 
        x = self.embedding(x)
        # x shape after concatenation == (64, 1,  512)
 
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        # passing the concatenated vector to the GRU
 
        output, state = self.gru(x)
        # shape == (batch_size, max_length, hidden_size)
 
        x = self.fc1(output)
        # x shape == (batch_size * max_length, hidden_size)
 
        x = tf.reshape(x, (-1, x.shape[2]))
 
        # Adding Dropout and BatchNorm Layers
        x= self.dropout(x)
        x= self.batchnormalization(x)
 
        # output shape == (64 * 512)
        x = self.fc2(x)
 
        # shape : (64 * 8329(vocab))
        return x, state, attention_weights
 
    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))
 
# 实例化模型
encoder = VGG16_Encoder(embedding_dim)
decoder = Rnn_Local_Decoder(embedding_dim, units, vocab_size)


# 损失函数，优化器设置
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
 
def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)


# 使用一种称为教师强制的技术，该技术将目标单词作为下一个输入传递给解码器。此技术有助于快速了解正确的序列或序列的正确统计属性
loss_plot = []
 
@tf.function
def train_step(img_tensor, target):
    loss = 0
    # initializing the hidden state for each batch
    # because the captions are not related from image to image
 
    hidden = decoder.reset_state(batch_size=target.shape[0])  #每迭代一次batch后重置 hidden_state
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * BATCH_SIZE, 1)  #input维度是3维
 
    with tf.GradientTape() as tape:  #eager模式下记录梯度
        features = encoder(img_tensor)  #VGG模式提取的特征
        for i in range(1, target.shape[1]):  #每张照片不止一个caption
            # passing the features through the decoder
            predictions, hidden, _ = decoder(dec_input, features, hidden)
            loss += loss_function(target[:, i], predictions)
 
            # using teacher forcing
            dec_input = tf.expand_dims(target[:, i], 1)
 
    total_loss = (loss / int(target.shape[1]))
    trainable_variables = encoder.trainable_variables + decoder.trainable_variables  #总训练参数
    gradients = tape.gradient(loss, trainable_variables)  #梯度计算及应用
    optimizer.apply_gradients(zip(gradients, trainable_variables))
 
    return loss, total_loss


# 训练模型
EPOCHS = 5
start_epoch = 0
for epoch in range(start_epoch, EPOCHS):
    start = time.time()
    total_loss = 0
 
    for (batch, (img_tensor, target)) in enumerate(dataset):
        batch_loss, t_loss = train_step(img_tensor, target)
        total_loss += t_loss
 
        if batch % 100 == 0:
            print ('Epoch {} Batch {} Loss {:.4f}'.format(
                epoch + 1, batch, batch_loss.numpy() / int(target.shape[1])))
    # storing the epoch end loss value to plot later
    loss_plot.append(total_loss / num_steps)
 
    print ('Epoch {} Loss {:.6f}'.format(epoch + 1, total_loss/num_steps))
 
    print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))




# 绘制损失函数
plt.plot(loss_plot)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Plot')
plt.show()
plt.savefig('testblueline.jpg')


#  定义字幕的贪婪方法
def evaluate(image):
    attention_plot = np.zeros((max_length, attention_features_shape))
    hidden = decoder.reset_state(batch_size=1)  #初始化hidden-state
    temp_input = tf.expand_dims(load_image(image)[0], 0)  #shape：(1,224,224,3)
    img_tensor_val = image_features_extract_model(temp_input)  #特征提取
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))  #shape:(1,7,7,512) reshape:(1,49,512)

    features = encoder(img_tensor_val)  #shape:(1,49,256)
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)
        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()
        predicted_id = tf.argmax(predictions[0]).numpy()   #使用softmax归一化结果，使用argmax查询最大值
        result.append(tokenizer.index_word[predicted_id])  #ID转字符，获取文本结果
        if tokenizer.index_word[predicted_id] == '<end>':  #判断是否是预设的结束标记
            return result, attention_plot
        dec_input = tf.expand_dims([predicted_id], 0)  #将预测值作为输入，预测下一个结果（teacher-forcing在这里使用数据标签作为输入）

    attention_plot = attention_plot[:len(result), :]

    return result, attention_plot


# 定义函数来绘制生成的每个单词的注意力图
def plot_attention(image, result, attention_plot):
    temp_image = np.array(Image.open(image))
    fig = plt.figure(figsize=(10, 10))
    len_result = len(result)
    for l in range(len_result):
        temp_att = np.resize(attention_plot[l], (8, 8))
        ax = fig.add_subplot(len_result//2, len_result//2, l+1)
        ax.set_title(result[l])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())
 
    plt.tight_layout()
    plt.show()


# 测试准确性
rid = np.random.randint(0, len(img_name_val))
image = 'E:\BaiduNetdiskDownload\Flickr8k and Flickr8kCN/Flicker8k_Dataset/2319175397_3e586cfaf8.jpg'
 
# real_caption = ' '.join([tokenizer.index_word[i] for i in cap_val[rid] if i not in [0]])
result, attention_plot = evaluate(image)
 
# remove <start> and <end> from the real_caption
real_caption = 'Two white dogs are playing in the snow'
first = real_caption.split(' ', 1)[1]

#remove "<unk>" in result
for i in result:
   if i=="<unk>":
       result.remove(i)
 
for i in real_caption:
   if i=="<unk>":
       real_caption.remove(i)
 
#remove <end> from result        
result_join = ' '.join(result)
result_final = result_join.rsplit(' ', 1)[0]
 
real_appn = []
real_appn.append(real_caption.split())
reference = real_appn
candidate = result
 
score = sentence_bleu(reference, candidate)
print(f"BELU score: {score*100}")
 
print ('Real Caption:', real_caption)
print ('Prediction Caption:', result_final)
plot_attention(image, result, attention_plot)
