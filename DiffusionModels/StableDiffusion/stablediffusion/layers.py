import tensorflow as tf
from tensorflow import keras
from keras.layers import Layer, ZeroPadding2D, Conv2D, Dense
from keras.activations import tanh
from keras.backend import batch_dot

class PaddedConv2D(Layer):
    def __init__(self, channels, kernel_size, padding=0, stride=1):
        super().__init__()
        self.padding2d = ZeroPadding2D((padding, padding))
        self.conv2d = Conv2D(
            channels, kernel_size, strides=(stride, stride)
        )

    def call(self, x):
        x = self.padding2d(x)

        return self.conv2d(x)
        
# https://arxiv.org/pdf/2002.05202
class GeGLU(Layer):
    def __init__(self, dim_out):
        super().__init__()
        self.proj = Dense(dim_out * 2)
        self.dim_out = dim_out
    
    def call(self, x):
        xp = self.proj(x)
        x, gate = xp[..., :self.dim_out], xp[..., self.dim_out:]

        return x * gelu(gate)
    
def gelu(x):
    tanh_res = tanh(x * 0.7978845608 * (1 + 0.044715 * (x**2)))

    return 0.5 * x * (1 + tanh_res)


def quick_gelu(x):
    return x * tf.sigmoid(x * 1.702)


def apply_seq(x, layers):
    for l in layers:
        x = l(x)
    return x


def td_dot(a, b):
    aa = tf.reshape(a, (-1, a.shape[2], a.shape[3]))
    bb = tf.reshape(b, (-1, b.shape[2], b.shape[3]))
    cc = batch_dot(aa, bb)
    
    return tf.reshape(cc, (-1, a.shape[1], cc.shape[1], cc.shape[2]))
