import tensorflow as tf
import os

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, Flatten, Reshape
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.keras.optimizers import Adam
from pixel_shuffler import PixelShuffler
from tensorflow.python.framework.ops import disable_eager_execution

#disable_eager_execution()

optimizer = Adam( lr=5e-5, beta_1=0.5, beta_2=0.999 )

IMAGE_SHAPE = tf.constant((64,64,3))
ENCODER_DIM = tf.constant(1024)

def conv( filters ):
    def block(x):
        x = Conv2D( filters, kernel_size=5, strides=2, padding='same' )(x)
        x = LeakyReLU(0.1)(x)
        return x
    return block

def upscale( filters ):
    def block(x):
        x = Conv2D( filters*4, kernel_size=3, padding='same' )(x)
        x = LeakyReLU(0.1)(x)
        x = PixelShuffler()(x)
        return x
    return block

def Encoder():
    input_ = Input( shape=IMAGE_SHAPE )
    x = input_
    x = conv( 128)(x)
    x = conv( 256)(x)
    x = conv( 512)(x)
    x = conv(1024)(x)
    x = Dense( ENCODER_DIM )( Flatten()(x) )
    x = Dense(4*4*1024)(x)
    x = Reshape((4,4,1024))(x)
    x = upscale(512)(x)
    return Model( input_, x )

def Decoder():
    input_ = Input( shape=tf.constant((8,8,512)) )
    x = input_
    x = upscale(256)(x)
    x = upscale(128)(x)
    x = upscale( 64)(x)
    x = Conv2D( 3, kernel_size=5, padding='same', activation='sigmoid' )(x)
    return Model( input_, x )

resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
tf.config.experimental_connect_to_cluster(resolver)
# This is the TPU initialization code that has to be at the beginning.
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.experimental.TPUStrategy(resolver)
try:
    with strategy.scope():
        encoder = Encoder()
        decoder_A = Decoder()
        decoder_B = Decoder()

        x = Input( shape=IMAGE_SHAPE )
        autoencoder_A = Model( x, decoder_A( encoder(x) ) )
        autoencoder_B = Model( x, decoder_B( encoder(x) ) )
    print("compiled")
except:
    print("error")
autoencoder_A.compile( optimizer=optimizer, loss='mean_absolute_error' )
autoencoder_B.compile( optimizer=optimizer, loss='mean_absolute_error' )

