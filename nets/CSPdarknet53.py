from functools import wraps

from keras import backend as K
from keras.initializers import random_normal
from keras.layers import Add, Concatenate, Conv2D, Layer, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from utils.utils import compose


class Mish(Layer):
    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return inputs * K.tanh(K.softplus(inputs))

    def get_config(self):
        config = super(Mish, self).get_config()
        return config

    def compute_output_shape(self, input_shape):
        return input_shape

'''
# single convolution with DarknetConv2D
'''
@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    # darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs = {'kernel_initializer' : random_normal(stddev=0.02)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)

'''
# convolution blocks -> convolution + standardization + activation function
# DarknetConv2D + BatchNormalization + Mish
'''
def DarknetConv2D_BN_Mish(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        Mish())

'''
# Structure blocks of CSPdarknet
'''
def resblock_body(x, num_filters, num_blocks, all_narrow=True):
    '''
    # Height and width compression using ZeroPadding2D and a convolution block with step size 2x2
    '''
    preconv1 = ZeroPadding2D(((1,0),(1,0)))(x)
    preconv1 = DarknetConv2D_BN_Mish(num_filters, (3,3), strides=(2,2))(preconv1)

    '''
    # create a large residual edge shortconv which bypasses much of the residual structure
    '''
    shortconv = DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, (1,1))(preconv1)

    '''
    # backbone loops num_blocks in residual structure
    '''
    mainconv = DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, (1,1))(preconv1)
    for i in range(num_blocks):
        y = compose(
                DarknetConv2D_BN_Mish(num_filters//2, (1,1)),
                DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, (3,3)))(mainconv)
        mainconv = Add()([mainconv,y])
    postconv = DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, (1,1))(mainconv)

    '''
    # concatenate the large residual edges
    '''
    route = Concatenate()([postconv, shortconv])

    # integrate channel number
    return DarknetConv2D_BN_Mish(num_filters, (1,1))(route)

'''
# main part of CSPdarknet53
# input a 416x416x3 image
# output three valid feature layers
'''
def darknet_body(x):
    x = DarknetConv2D_BN_Mish(32, (3,3))(x)
    x = resblock_body(x, 64, 1, False)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 8)
    feat1 = x
    x = resblock_body(x, 512, 8)
    feat2 = x
    x = resblock_body(x, 1024, 4)
    feat3 = x
    return feat1,feat2,feat3
