# 参考文献 https://blog.csdn.net/weixin_43496706/article/details/101210981 
# 用于构建经典的卷积神经网络 从上而下提取特征 但因为是从上而下 所以不适合用于分割任务(因为在卷积池化过程会损失很多信息)

from keras import layers
from keras.initializers import RandomNormal


def VGG16(img_input):
    # 不一定是512*512 这里只是举个例子
    # Block 1
    # 512,512,3 -> 512,512,64
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer = RandomNormal(stddev=0.02), 
                      name='block1_conv1')(img_input)
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer = RandomNormal(stddev=0.02), 
                      name='block1_conv2')(x)
    feat1 = x
    # 512,512,64 -> 256,256,64
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    # 256,256,64 -> 256,256,128
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer = RandomNormal(stddev=0.02), 
                      name='block2_conv1')(x)
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer = RandomNormal(stddev=0.02), 
                      name='block2_conv2')(x)
    feat2 = x
    # 256,256,128 -> 128,128,128
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)


    # Block 3
    # 128,128,128 -> 128,128,256
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer = RandomNormal(stddev=0.02), 
                      name='block3_conv1')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer = RandomNormal(stddev=0.02), 
                      name='block3_conv2')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer = RandomNormal(stddev=0.02), 
                      name='block3_conv3')(x)
    feat3 = x
    # 128,128,256 -> 64,64,256
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    # 64,64,256 -> 64,64,512
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer = RandomNormal(stddev=0.02), 
                      name='block4_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer = RandomNormal(stddev=0.02), 
                      name='block4_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer = RandomNormal(stddev=0.02), 
                      name='block4_conv3')(x)
    feat4 = x
    # 64,64,512 -> 32,32,512
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    # 32,32,512 -> 32,32,512
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer = RandomNormal(stddev=0.02), 
                      name='block5_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer = RandomNormal(stddev=0.02), 
                      name='block5_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_initializer = RandomNormal(stddev=0.02), 
                      name='block5_conv3')(x)
    feat5 = x
    return feat1, feat2, feat3, feat4, feat5

# # 测试代码
# if __name__ == '__main__':
#     import numpy as np
#     a = VGG16(np.random.rand(1,512,512,3))
#     while True:
#         pass