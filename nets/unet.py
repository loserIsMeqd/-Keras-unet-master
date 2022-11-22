from keras.initializers import random_normal
from keras.layers import *
from keras.models import *

# 在运行是出现ModuleNotFoundError: No module named的错误 可能是路径出现了问题 了解关于import的本质
# 路径问题 如果当前路径是nets 那么就不需要加下面两行代码
import os, sys
sys.path.append(os.getcwd())

from nets.vgg16 import VGG16
from nets.resnet50 import ResNet50


#定义unet网络层 Input声明输入张量的形状 以及有了input层和Model（inputs, outputs）就不是自定义的网络模块
def Unet(input_shape=(512, 512, 3), num_classes=21, backbone='vgg'):
    inputs = Input(input_shape)
    #-------------------------------#
    #   获得五个有效特征层
    #   feat1   512,512,64
    #   feat2   256,256,128
    #   feat3   128,128,256
    #   feat4   64,64,512
    #   feat5   32,32,512
    #-------------------------------#
    if backbone == 'vgg':
        feat1, feat2, feat3, feat4, feat5 = VGG16(inputs)
    elif backbone == 'resnet':
        feat1, feat2, feat3, feat4, feat5 = ResNet50(inputs)
    else:
        raise ValueError('Unsupported backbone - `{}`, Use resnet or vgg.'.format(backbone))

    channels = [64, 128, 256, 512]
    # 32, 32, 512 -> 64, 64, 512

    # 32, 32, 512 -> 64, 64, 512
    P5_up = UpSampling2D(size=(2, 2))(feat5)
    # 64, 64, 512 + 64, 64, 512 -> 64, 64, 1024
    P4 = Concatenate(axis=3)([feat4, P5_up])
    # 64, 64, 1024 -> 64, 64, 512
    P4 = Conv2D(channels[3], 3, activation='relu', padding='same', kernel_initializer = random_normal(stddev=0.02))(P4)
    P4 = Conv2D(channels[3], 3, activation='relu', padding='same', kernel_initializer = random_normal(stddev=0.02))(P4)

    # 64, 64, 512 -> 128, 128, 256
    P4_up = UpSampling2D(size=(2, 2))(P4)
    # 128, 128, 256 + 128, 128, 256 -> 128, 128, 512 
    P3 = Concatenate(axis=3)([feat3, P4_up])
    # 128, 128, 512 -> 128, 128, 256
    P3 = Conv2D(channels[2], 3, activation='relu', padding='same', kernel_initializer = random_normal(stddev=0.02))(P3)
    P3 = Conv2D(channels[2], 3, activation='relu', padding='same', kernel_initializer = random_normal(stddev=0.02))(P3)

    # 128, 128, 256 -> 256, 256, 128
    P3_up = UpSampling2D(size=(2, 2))(P3)
    # 256, 256, 128 + 256, 256, 128 -> 256, 256, 256
    P2 = Concatenate(axis=3)([feat2, P3_up])
    # 256, 256, 256 -> 256, 256, 128
    P2 = Conv2D(channels[1], 3, activation='relu', padding='same', kernel_initializer = random_normal(stddev=0.02))(P2)
    P2 = Conv2D(channels[1], 3, activation='relu', padding='same', kernel_initializer = random_normal(stddev=0.02))(P2)

    # 256, 256, 128 -> 512, 512, 64
    P2_up = UpSampling2D(size=(2, 2))(P2)
    # 512, 512, 64 + 512, 512, 64 -> 512, 512, 128
    P1 = Concatenate(axis=3)([feat1, P2_up])
    # 512, 512, 128 -> 512, 512, 64
    P1 = Conv2D(channels[0], 3, activation='relu', padding='same', kernel_initializer = random_normal(stddev=0.02))(P1)
    P1 = Conv2D(channels[0], 3, activation='relu', padding='same', kernel_initializer = random_normal(stddev=0.02))(P1)

    if backbone == "vgg":
        # 512, 512, 64 -> 512, 512, num_classes
        P1 = Conv2D(num_classes, 1, activation="softmax")(P1)
    elif backbone == "resnet50":
        ResNet50_up = UpSampling2D(size=(2, 2))(P1)
        # 512, 512, 192 -> 512, 512, 64
        ResNet50_up = Conv2D(channels[0], 3, activation='relu', padding='same', kernel_initializer = random_normal(stddev=0.02))(ResNet50_up)
        ResNet50_up = Conv2D(channels[0], 3, activation='relu', padding='same', kernel_initializer = random_normal(stddev=0.02))(ResNet50_up)

        P1 = Conv2D(num_classes, 1, activation="softmax")(ResNet50_up)
    else:
        raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        
    model = Model(inputs=inputs, outputs=P1)
    return model


if __name__ == "__main__":
    model = Unet()
    model.summary()
    