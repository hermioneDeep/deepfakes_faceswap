import tensorflow as tf
from tensorflow.keras.layers import (add, Add, BatchNormalization, concatenate, Lambda,
                          Permute, Reshape, SeparableConv2D, Softmax, UpSampling2D, LeakyReLU, Conv2D, Activation)
from tensorflow.keras import regularizers
#from tensorflow.keras.layers.core import Activation
from tensorflow.keras.initializers import he_uniform, VarianceScaling
from pixel_shuffler import PixelShuffler
names = dict()
first_run = 1
use_subpixel = 0
use_icnr_init = 0
use_convaware_init = 0
use_reflect_padding = 0

def get_name(name):
    """ Return unique layer name for requested block """
    names[name] = names.setdefault(name, -1) + 1
    name = "{}_{}".format(name, names[name])
    return name

def set_default_initializer(kwargs):
    if "kernel_initializer" in kwargs:
        return kwargs
    default = he_uniform()
    if kwargs.get("kernel_initializer", None) != default:
        kwargs["kernel_initializer"] = default
    return kwargs


def conv2d(inp, filters, kernel_size, strides=(1, 1), padding="same", **kwargs):
    """ A standard conv2D layer with correct initialization """
    if kwargs.get("name", None) is None:
        kwargs["name"] = get_name("conv2d_{}".format(inp.shape[1]))
    kwargs = set_default_initializer(kwargs)
    var_x = Conv2D(filters, kernel_size,
                    strides=strides,
                    padding=padding,
                    **kwargs)(inp)
    return var_x

# <<< Original Model Blocks >>> #
def conv(inp, filters, kernel_size=5, strides=2, padding="same",
            use_instance_norm=False, res_block_follows=False, **kwargs):
    """ Convolution Layer"""
    name = get_name("conv_{}".format(inp.shape[1]))
#        if use_reflect_padding:
#            inp = ReflectionPadding2D(stride=strides,
#                                      kernel_size=kernel_size,
#                                      name="{}_reflectionpadding2d".format(name))(inp) //TODO
#            padding = "valid"
    var_x = conv2d(inp, filters,
                        kernel_size=kernel_size,
                        strides=strides,
                        padding=padding,
                        name="{}_conv2d".format(name),
                        **kwargs)
    #if use_instance_norm:
    #    var_x = InstanceNormalization(name="{}_instancenorm".format(name))(var_x)
    #if not res_block_follows:
    #    var_x = LeakyReLU(0.1, name="{}_leakyrelu".format(name))(var_x) //TODO
    return var_x

def upscale(inp, filters, kernel_size=3, padding="same",
            use_instance_norm=False, res_block_follows=False, scale_factor=2, **kwargs):
    """ Upscale Layer """
    name = get_name("upscale_{}".format(inp.shape[1]))
#        if use_reflect_padding:
#            inp = ReflectionPadding2D(stride=1,
#                                      kernel_size=kernel_size,
#                                      name="{}_reflectionpadding2d".format(name))(inp)
#            padding = "valid"
    kwargs = set_default_initializer(kwargs)
#        if use_icnr_init:
#            original_init = switch_kernel_initializer(
#                kwargs,
#                ICNR(initializer=kwargs["kernel_initializer"]))
    var_x = conv2d(inp, filters * scale_factor * scale_factor,
                        kernel_size=kernel_size,
                        padding=padding,
                        name="{}_conv2d".format(name),
                        **kwargs)
#        if use_icnr_init:
#            switch_kernel_initializer(kwargs, original_init)
#        if use_instance_norm:
#            var_x = InstanceNormalization(name="{}_instancenorm".format(name))(var_x)
#        if not res_block_follows:
#            var_x = LeakyReLU(0.1, name="{}_leakyrelu".format(name))(var_x)
#        if use_subpixel:
#            var_x = SubPixelUpscaling(name="{}_subpixel".format(name),
#                                      scale_factor=scale_factor)(var_x) //TODO
#        else:
    var_x = PixelShuffler(name="{}_pixelshuffler".format(name), size=scale_factor)(var_x)
    return var_x

# <<< DFaker Model Blocks >>> #
def res_block(inp, filters, kernel_size=3, padding="same", **kwargs):
    """ Residual block """
    name = get_name("residual_{}".format(inp.shape[1]))
    var_x = LeakyReLU(alpha=0.2, name="{}_leakyrelu_0".format(name))(inp)
#        if use_reflect_padding:
#            var_x = ReflectionPadding2D(stride=1,
#                                        kernel_size=kernel_size,
#                                        name="{}_reflectionpadding2d_0".format(name))(var_x)
#            padding = "valid"
    var_x = conv2d(var_x, filters,
                        kernel_size=kernel_size,
                        padding=padding,
                        name="{}_conv2d_0".format(name),
                        **kwargs)
    var_x = LeakyReLU(alpha=0.2, name="{}_leakyrelu_1".format(name))(var_x)
#        if use_reflect_padding:
#            var_x = ReflectionPadding2D(stride=1,
#                                        kernel_size=kernel_size,
#                                        name="{}_reflectionpadding2d_1".format(name))(var_x)
#            padding = "valid"
#        if not use_convaware_init:
#            original_init = switch_kernel_initializer(kwargs, VarianceScaling(
#                scale=0.2,
#                mode="fan_in",
#                distribution="uniform"))
#        var_x = conv2d(var_x, filters,
#                            kernel_size=kernel_size,
#                            padding=padding,
#                            **kwargs)
#        if not use_convaware_init:
#            switch_kernel_initializer(kwargs, original_init)
    var_x = Add()([var_x, inp])
    var_x = LeakyReLU(alpha=0.2, name="{}_leakyrelu_3".format(name))(var_x)
    return var_x
def conv_sep(inp, filters, kernel_size=5, strides=2, **kwargs):
    """ Seperable Convolution Layer """
    name = get_name("separableconv2d_{}".format(inp.shape[1]))
    kwargs = set_default_initializer(kwargs)
    var_x = SeparableConv2D(filters,
                            kernel_size=kernel_size,
                            strides=strides,
                            padding="same",
                            name="{}_seperableconv2d".format(name),
                            **kwargs)(inp)
    var_x = Activation("relu", name="{}_relu".format(name))(var_x)
    return var_x


