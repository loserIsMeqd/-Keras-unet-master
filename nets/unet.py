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
def Unet(input_shape=(512, 512, 3), num_class=21, backbone='vgg'):
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

    channel = [64, 128, 256, 512]
    # 32, 32, 512 -> 64, 64, 512

    