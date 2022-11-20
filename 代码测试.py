# import os # 本文详细介绍了与操作系统交互的os模块中一些常用的属性和函数，基本可以覆盖初阶的学习和使用。有了这些功能，我们已经可以写出一些比较实用的脚本了。https://zhuanlan.zhihu.com/p/150835193
# # 详细可见git clone https://github.com/JustDoPython/python-100-day.git
# print(os.name)
# print(os.environ['HOMEPATH'])

# import keras
# print(keras.__version__)
# import keras.callbacks
# # print(keras.callbacks.Callback.__doc__)

# import tensorflow as tf
# from keras.layers import Conv2D, Dense, DepthwiseConv2D

# print(Dense(12).get_config())
# # 卷积神经网络中对于channels的理解 通道指特殊的维度 1乘1的卷积核就是一个通道

# # 层的实例是可调用的，它以张量为参数，并且返回一个张量
# from keras.layers import Dense, Input
# from keras.models import Model

# inputs = Input(shape=(784,))
# x = Dense(64, activation='relu')(inputs)
# predictions = Dense(10, activation='softmax')(x)
# # 函数式API返回的是张量 输入也是张量  可以轻易实现模型的多输入多输出和复用

# import numpp as np
# import keras.layers
# from keras.models import Model

# # 自定义网路块
# def vgg12(input_tensor):
#     x = keras.layers.Conv2D(2, (3, 3), padding='same')(input_tensor)
#     x = keras.layers.BatchNormalization()(x)
#     x = keras.layers.Activation('relu')(x)
#     x = keras.layers.Conv2D(2, (3, 3), activation='relu',padding='same')(x)
#     x1 = x

#     x = keras.layers.MaxPool2D((2,2),strides=(2,2))(x)
#     x = keras.layers.Conv2D(4, (3,3), padding='same', activation='relu')(x)
#     x = keras.layers.Conv2D(4, (3,3), padding='same', activation='relu')(x)
#     x2 = x
#     x = keras.layers.MaxPool2D((2,2),strides=(2,2))(x)

#     x = keras.layers.Flatten()(x)
#     x = keras.layers.Dense(4, activation="softmax")(x)

#     return x

# a = keras.layers.Input((512, 512, 3))
# b = vgg12(a)
# model = Model(a,b)
# model.compile('adam','mean_squared_error')
# # 关于输入输出的疑问 可以参考这篇文章 https://www.cnblogs.com/panchuangai/p/12567970.html#:~:text=%E7%90%86%E8%A7%A3%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%AD%E7%9A%84%E8%BE%93%E5%85%A5%E4%B8%8E%E8%BE%93%E5%87%BA%E5%BD%A2%E7%8A%B6%20%28Keras%E5%AE%9E%E7%8E%B0%29%201%20%E8%BE%93%E5%85%A5%E7%9A%84%E5%BD%A2%E7%8A%B6%20%E4%BD%A0%E5%A7%8B%E7%BB%88%E5%BF%85%E9%A1%BB%E5%B0%864D%E6%95%B0%E7%BB%84%E4%BD%9C%E4%B8%BACNN%E7%9A%84%E8%BE%93%E5%85%A5%E3%80%82%20%E5%9B%A0%E6%AD%A4%EF%BC%8C%E8%BE%93%E5%85%A5%E6%95%B0%E6%8D%AE%E7%9A%84%E5%BD%A2%E7%8A%B6%E4%B8%BA%20%28batch_size%EF%BC%8Cheight%EF%BC%8Cwidth%EF%BC%8Cdepth%29%EF%BC%8C%E5%85%B6%E4%B8%AD%E7%AC%AC%E4%B8%80%E7%BB%B4%E8%A1%A8%E7%A4%BA%E5%9B%BE%E5%83%8F%E7%9A%84batch%E5%A4%A7%E5%B0%8F%EF%BC%8C%E5%85%B6%E4%BB%96%E4%B8%89%E4%B8%AA%E7%BB%B4%E8%A1%A8%E7%A4%BA%E5%9B%BE%E5%83%8F%E7%9A%84%E5%90%84%E4%B8%AA%E5%B1%9E%E6%80%A7%EF%BC%8C%E5%8D%B3%E9%AB%98%E5%BA%A6%EF%BC%8C%E5%AE%BD%E5%BA%A6%E5%92%8C%E6%B7%B1%E5%BA%A6%E3%80%82%20%E6%B7%B1%E5%BA%A6%E5%B0%B1%E6%98%AF%E8%89%B2%E5%BD%A9%E9%80%9A%E9%81%93%E7%9A%84%E6%95%B0%E9%87%8F%E3%80%82,4%20%E6%B1%87%E6%80%BB%20%E4%BD%A0%E5%A7%8B%E7%BB%88%E5%BF%85%E9%A1%BB%E5%B0%86%E5%BD%A2%E7%8A%B6%E4%B8%BA%20%28batch_size%2C%20height%2C%20width%2C%20depth%29%E7%9A%844D%E6%95%B0%E7%BB%84%E8%BE%93%E5%85%A5CNN%E3%80%82%20
# model.summary()
# model.fit(np.random.random([10, 512, 512 ,3]),np.array([0.1,0.3,0.3,0.3]))

# # 关于raise的异常和自定义异常的使用
# def test(valueerror):
#     if valueerror != [1,2,3]:
#         raise ValueError('valueerror')

# print(np.random.random([10, 512, 512 ,3]).shape)

# try:
#     test(1)
# except ValueError as e:
#     print(e)

import numpy as np
from keras import layers
from keras.models import Model

#注意自己玩的时候要注意用np.random.random([10, 512, 512 ,3])  或者用reshape重新裁剪一下

x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
# 这里是三维度空间中的二维
inputs = layers.Input(shape=(2, 2, 1))
out = layers.UpSampling2D(size=(2, 2))(inputs)
model = Model(inputs, out)
model.summary()
y = model.predict (np.reshape(x, (2, 2, 2, 1)))
y = np.reshape(y, (8, 4))
print('input:')
print(x)
print(np.reshape(x, (2, 2, 2, 1)))
print('output:')
print(y)
# 以下为日志结果：
# input:
# [[[1 2]
#   [3 4]]

#  [[5 6]
#   [7 8]]]
# output:
# [[1. 1. 2. 2.]
#  [1. 1. 2. 2.]
#  [3. 3. 4. 4.]
#  [3. 3. 4. 4.]
#  [5. 5. 6. 6.]
#  [5. 5. 6. 6.]
#  [7. 7. 8. 8.]
#  [7. 7. 8. 8.]]

import numpy as np

342342342

阿斯蒂芬撒地方
哦加哦
OK破