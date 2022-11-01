import keras.models
import tensorflow as tf
import numpy as np

from keras.layers.core import Lambda
from keras import backend as K
from tensorflow.keras import layers, models, Input
from keras.layers import Conv2D, BatchNormalization, Activation, ZeroPadding2D, AveragePooling2D, MaxPooling2D, Dense

input_shape = (2, 4, 5, 3)
x = tf.random.normal(input_shape)
y = tf.keras.layers.GlobalAveragePooling2D()(x)
print(y.shape)


#SENet代码
def SE_Block(input_hyj):
    x = layers.GlobalAveragePooling2D()(input_hyj) #GlobalAveragePooling这个函数可以使正常的输入(batchsize,w,h,channel)变成输出(batchsize,channel)这样
    channel = input_hyj.shape[-1]
    x = x[:, None, None, :] #这个作用 输入经过GlobalAveragePooling以后从(1,224,224,32)变成了(1,32) 需要加一个这个才能变成了(1,1,1,32)
    x = Conv2D(filters=channel//16, kernel_size=(1, 1))(x) #通道//16这个是论文里SE这个论文里得出来的 效果最好
    x = Activation('relu')(x)
    x = Conv2D(filters=channel, kernel_size=(1, 1))(x)
    x = Activation('sigmoid')(x)
    x = layers.Multiply()([input_hyj, x])

    return x

input_shape = Input([224,224,64])
k = SE_Block(input_shape)
model = models.Model(input_shape, k)
model.summary()


#CBAM代码
#CAM模块代码
def channelattention(input_hyj2):

    channel = input_hyj2.shape[-1] #输入特征图的通道数
    x = input_hyj2
    x_max = layers.GlobalMaxPooling2D()(x) #两个池化分支 一个最大池化 一个平均池化
    x_avg = layers.GlobalAveragePooling2D()(x)
    x_max = x_max[:, None, None, :] #原因和上面SENet一样
    x_avg = x_avg[:, None, None, :]

    x_max = Conv2D(filters=channel//16, kernel_size=(1, 1))(x_max) #先对最大池化的进行拉长变成1*1*通道数的一个形状 这里用flatten 或者 1*1 卷积都可以拉伸特征图
    x_max = Activation('relu')(x_max)
    x_max = Conv2D(filters=channel, kernel_size=(1, 1))(x_max)

    x_avg = Conv2D(filters=channel//16, kernel_size=(1, 1))(x_avg)#同理上面
    x_avg = Activation('relu')(x_avg)
    x_avg = Conv2D(filters=channel, kernel_size=(1, 1))(x_avg)

    x = layers.Add()([x_max, x_avg]) #对两个处理后的特征通道进行相加 然后sigmoid函数处理
    x = Activation('sigmoid')(x)

    x = layers.Multiply()([input_hyj2, x]) #跟输入进行融合
    return x



#定义的函数 没啥用因为会报错下面用
def channel_max(x):
    x = K.max(x, axis = 3, keepdims=True)
    return x

def channel_avg(x):
    x = K.mean(x, axis = 3, keepdims=True)
    return x


#SAM 模块
def spatialattention(inputs):
    x_max = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(inputs) #lambda 是一个简便的函数过程比如 x = x*2 lambda x:x*2 这样
    x_avg = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(inputs) #这一行和上一行主要是把特征图比如5*5*30 通过一个最大池化和平均池化变成5*5*1 5*5*1 两个这样的特征图
    #x_max = Lambda(channel_max)(inputs) 有问题视频里可以这样 我这样写不行
    #x_avg = Lambda(channel_avg)(inputs)
    x     = layers.Concatenate()([x_max, x_avg])
    x     = Conv2D(filters=1, kernel_size=(7, 7), padding='same')(x) #7*7卷积核大小也是论文里提出来的
    x     = Activation('sigmoid')(x)
    x     = layers.Multiply()([inputs, x])
    return x


inputs = Input([26, 26, 512])
y = channelattention(inputs) #CAM模块
y = spatialattention(y)  #SAM模块
model = models.Model(inputs, y)
model.summary()