import math
from functools import partial

import numpy as np
import tensorflow as tf
from keras import backend as K

def dice_loss_with_CE(cls_weights, beta=1, smooth = 1e-5):
    cls_weights = np.reshape(cls_weights, [1, 1, 1, -1])
    def _dice_loss_with_CE(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())

        CE_loss = - y_true[...,:-1] * K.log(y_pred) * cls_weights
        CE_loss = K.mean(K.sum(CE_loss, axis = -1))

        tp = K.sum(y_true[...,:-1] * y_pred, axis=[0,1,2])
        fp = K.sum(y_pred         , axis=[0,1,2]) - tp
        fn = K.sum(y_true[...,:-1], axis=[0,1,2]) - tp

        score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
        score = tf.reduce_mean(score)
        dice_loss = 1 - score
        # dice_loss = tf.Print(dice_loss, [dice_loss, CE_loss])
        return CE_loss + dice_loss
    return _dice_loss_with_CE