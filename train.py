import cv2
import numpy

from utils import get_image_paths, load_images, stack_images
from training_data import get_training_data

#from model import autoencoder_A
#from model import autoencoder_B
from model import dual_model
from model import encoder, decoder_A, decoder_B
import matplotlib.pyplot as plt
import time
import tensorflow as tf
import random
import numpy as np
import pathlib
data_root = pathlib.Path("/content/deepfakes_faceswap/")


def preprocess_image(image):
  image = tf.image.decode_jpeg(image, channels=3)
  #image = tf.image.central_crop(image, central_fraction=0.5)

  image = tf.image.resize_with_crop_or_pad(image, 128, 128)
  #image /= 255.0  # normalize to [0,1] range

  return image

def load_and_preprocess(path):
  image = tf.io.read_file(path)
  return preprocess_image(image)

def data_generator(src = 'workspace/data_src/*', dst = 'workspace/data_dst/*', batch_size = 56, target_res = 128):
  batch_src = np.zeros((batch_size, target_res, target_res, 3))
  batch_dst = np.zeros((batch_size, target_res, target_res, 3))
#  batch_src_inp = np.zeros((batch_size, target_res, target_res, 3))
#  batch_src_out = np.zeros((batch_size, target_res, target_res, 3))
#  batch_dst_inp = np.zeros((batch_size, target_res, target_res, 3))
#  batch_dst_out = np.zeros((batch_size, target_res, target_res, 3))
  paths_src = list(data_root.glob(src))
  paths_dst = list(data_root.glob(dst))
  print(paths_src)
  paths_src = [str(path) for path in paths_src]
  paths_dst = [str(path) for path in paths_dst]
  random.shuffle(paths_src)
  random.shuffle(paths_dst)
  print(paths_src)
  assert len(paths_src) >= batch_size
  assert len(paths_dst) >= batch_size
  for i in range(batch_size):
    batch_src[i] = load_and_preprocess(paths_src[i])
    batch_dst[i] = load_and_preprocess(paths_dst[i])
    inputs = [batch_src, batch_dst]
    targets = [batch_src, batch_dst]
  yield inputs, targets

#batcher = data_generator()

try:
    encoder  .load_weights( "models/encoder.h5"   )
    decoder_A.load_weights( "models/decoder_A.h5" )
    decoder_B.load_weights( "models/decoder_B.h5" )
except:
    pass

def save_model_weights():
    encoder  .save_weights( "models/encoder.h5"   )
    decoder_A.save_weights( "models/decoder_A.h5" )
    decoder_B.save_weights( "models/decoder_B.h5" )
    print( "save model weights" )

#images_A = get_image_paths( "workspace/data_src" )
#images_B = get_image_paths( "workspace/data_dst" )
#images_A = load_images( images_A ) / 255.0
#images_B = load_images( images_B ) / 255.0

#images_A += images_B.mean( axis=(0,1,2) ) - images_A.mean( axis=(0,1,2) )

print( "press 'q' to stop training and save model" )

for epoch in range(1000000):
    batch_size = 512
    #warped_A, target_A = get_training_data( images_A, batch_size )
    #warped_B, target_B = get_training_data( images_B, batch_size )
    #loss_A = autoencoder_A.train_on_batch( warped_A, target_A )
    #loss_B = autoencoder_B.train_on_batch( warped_B, target_B )
    dual_loss = dual_model.fit(data_generator(), epochs = 10)
    #dual_loss = dual_model.fit([warped_A, warped_B], [target_A, target_B], epochs = 10)
    #print( dual_loss )
    if epoch % 10 == 0:
        save_model_weights()
        test_A = target_A[0:14]
        test_B = target_B[0:14]
        pred_AA, pred_BA = dual_model.predict([test_A, test_A])
        figure_A = numpy.stack([
            test_A,
            pred_AA,
            pred_BA,
            ], axis=1 )
        pred_BA, pred_BB = dual_model.predict([test_B, test_B])
        figure_B = numpy.stack([
            test_B,
            pred_BB,
            pred_BA,
            ], axis=1 )
        figure = numpy.concatenate( [ figure_A, figure_B ], axis=0 )
        figure = figure.reshape( (4,7) + figure.shape[1:] )
        figure = stack_images( figure )
        figure = numpy.clip( figure * 255, 0, 255 ).astype('uint8')
        cv2.imwrite("preview.png", figure)

    key = cv2.waitKey(1)
    if key == ord('q'):
        save_model_weights()
        exit()