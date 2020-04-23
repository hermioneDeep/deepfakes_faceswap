import tensorflow as tf
import os

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, Flatten, Reshape, 
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.keras.optimizers import Adam
from pixel_shuffler import PixelShuffler
from model_tools import NNBlocks
from tf.keras.initializers import RandomNormal

blocks = NNBlocks()

#from tensorflow.python.framework.ops import disable_eager_execution

#disable_eager_execution()

optimizer = Adam( lr=5e-5, beta_1=0.5, beta_2=0.999 )

IMAGE_SHAPE = (64,64,3)
ENCODER_DIM = 1024

#def conv( filters ):
#    def block(x):
#        x = Conv2D( filters, kernel_size=5, strides=2, padding='same' )(x)
#        x = LeakyReLU(0.1)(x)
#        return x
#    return block

#def upscale( filters ):
#    def block(x):
#        x = Conv2D( filters*4, kernel_size=3, padding='same' )(x)
#        x = LeakyReLU(0.1)(x)
#        x = PixelShuffler()(x)
#        return x
#    return block
#
#def Encoder():
#    input_ = Input( shape=IMAGE_SHAPE )
#    x = input_
#    x = conv( 128)(x)
#    x = conv( 256)(x)
#    x = conv( 512)(x)
#    x = conv(1024)(x)
#    x = Dense( ENCODER_DIM )( Flatten()(x) )
#    x = Dense(4*4*1024)(x)
#    x = Reshape((4,4,1024))(x)
#    x = upscale(512)(x)
#    return Model( input_, x )
#
#def Decoder():
#    input_ = Input( shape=(8,8,512) )
#    x = input_
#    x = upscale(256)(x)
#    x = upscale(128)(x)
#    x = upscale( 64)(x)
#    x = Conv2D( 3, kernel_size=5, padding='same', activation='sigmoid' )(x)
#    return Model( input_, x )


#kwargs["input_shape"] = (128, 128, 3)
#kwargs["encoder_dim"] = 512 if config["lowmem"] else 1024
kernel_initializer = RandomNormal(0, 0.02)
input_shape = (128, 128, 3)
encoder_dim = 1024
def encoder():
    """ Encoder Network """
    global kernel_initializer
    global input_shape
    kwargs = dict(kernel_initializer=kernel_initializer)
    input_ = Input(shape=input_shape)
    in_conv_filters = input_shape[0]
    dense_shape = input_shape[0] // 16

    var_x = blocks.conv(input_, in_conv_filters, res_block_follows=True, **kwargs)
    tmp_x = var_x
    res_cycles = 16
    for _ in range(res_cycles):
        nn_x = blocks.res_block(var_x, in_conv_filters, **kwargs)
        var_x = nn_x
    # consider adding scale before this layer to scale the residual chain
    var_x = add([var_x, tmp_x])
    var_x = blocks.conv(var_x, 128, **kwargs)
    var_x = PixelShuffler()(var_x)
    var_x = blocks.conv(var_x, 128, **kwargs)
    var_x = PixelShuffler()(var_x)
    var_x = blocks.conv(var_x, 128, **kwargs)
    var_x = blocks.conv_sep(var_x, 256, **kwargs)
    var_x = blocks.conv(var_x, 512, **kwargs)
    var_x = blocks.conv_sep(var_x, 1024, **kwargs)
    var_x = Dense(encoder_dim, **kwargs)(Flatten()(var_x))
    var_x = Dense(dense_shape * dense_shape * 1024, **kwargs)(var_x)
    var_x = Reshape((dense_shape, dense_shape, 1024))(var_x)
    var_x = blocks.upscale(var_x, 512, **kwargs)
    return Model(input_, var_x)

def decoder_A():
    """ Decoder Network """
    global kernel_initializer
    global input_shape
    kwargs = dict(kernel_initializer=kernel_initializer)
    decoder_shape = input_shape[0] // 8
    input_ = Input(shape=(decoder_shape, decoder_shape, 512))

    var_x = input_
    var_x = blocks.upscale(var_x, 512, res_block_follows=True, **kwargs)
    var_x = blocks.res_block(var_x, 512, **kwargs)
    var_x = blocks.upscale(var_x, 256, res_block_follows=True, **kwargs)
    var_x = blocks.res_block(var_x, 256, **kwargs)
    var_x = blocks.upscale(var_x, input_shape[0], res_block_follows=True, **kwargs)
    var_x = blocks.res_block(var_x, input_shape[0], **kwargs)
    var_x = blocks.conv2d(var_x, 3,
                                kernel_size=5,
                                padding="same",
                                activation="sigmoid",
                                name="face_out")
    outputs = [var_x]
    return Model(input_, outputs=outputs)

def decoder_B():
    """ Decoder Network """
    global kernel_initializer
    global input_shape
    kwargs = dict(kernel_initializer=kernel_initializer)
    decoder_shape = input_shape[0] // 8
    input_ = Input(shape=(decoder_shape, decoder_shape, 512))

    var_x = input_
    var_x = blocks.upscale(var_x, 512, res_block_follows=True, **kwargs)
    var_x = blocks.res_block(var_x, 512, **kwargs)
    var_x = blocks.upscale(var_x, 256, res_block_follows=True, **kwargs)
    var_x = blocks.res_block(var_x, 256, **kwargs)
    var_x = blocks.upscale(var_x, input_shape[0], res_block_follows=True, **kwargs)
    var_x = blocks.res_block(var_x, input_shape[0], **kwargs)
    var_x = blocks.conv2d(var_x, 3,
                                kernel_size=5,
                                padding="same",
                                activation="sigmoid",
                                name="face_out")
    outputs = [var_x]
    return Model(input_, outputs=outputs)


#resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
#tf.config.experimental_connect_to_cluster(resolver)
## This is the TPU initialization code that has to be at the beginning.
#tf.tpu.experimental.initialize_tpu_system(resolver)
#strategy = tf.distribute.experimental.TPUStrategy(resolver)
#try:
    #with strategy.scope():
encoder = Encoder()
decoder_A = Decoder()
decoder_B = Decoder()

var_x = Input( shape=IMAGE_SHAPE )
var_y = Input( shape=IMAGE_SHAPE )
#autoencoder_A = Model( x, decoder_A( encoder(x) ) )
#autoencoder_B = Model( x, decoder_B( encoder(x) ) )
dual_model = Model(inputs=[x, y],
                    outputs=[decoder_A(encoder(var_x)), decoder_B(encoder(var_y))])
dual_model.compile(optimizer=optimizer, loss='mean_absolute_error')
    print("compiled")
#except:
#    print("error")
#autoencoder_A.compile( optimizer=optimizer, loss='mean_absolute_error' )
#autoencoder_B.compile( optimizer=optimizer, loss='mean_absolute_error' )

