import numpy
from image_augmentation import random_transform
from image_augmentation import random_warp
from utils import get_images

random_transform_args = {
    'rotation_range': 10,
    'zoom_range': 0.05,
    'shift_range': 0.05,
    'random_flip': 0.4,
    }


class ArtificialDataset(tf.data.Dataset):
    def _generator(num_samples):
    image_provider = get_images()
    for image in image_provider:
        image = random_transform( image, **random_transform_args )
        warped_img, target_img = random_warp( image )
        yield warped_img, target_img    
    def __new__(cls, num_samples=3):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=tf.dtypes.int64,
            output_shapes=(1,),
            args=(num_samples,)
        )


def get_train_images():
    image_provider = get_images()
    for image in image_provider:
        image = random_transform( image, **random_transform_args )
        warped_img, target_img = random_warp( image )
        yield warped_img, target_img

def get_training_data( images, batch_size ):
    indices = numpy.random.randint( len(images), size=batch_size )
    for i,index in enumerate(indices):
        image = images[index]
        image = random_transform( image, **random_transform_args )
        warped_img, target_img = random_warp( image )

        if i == 0:
            warped_images = numpy.empty( (batch_size,) + warped_img.shape, warped_img.dtype )
            target_images = numpy.empty( (batch_size,) + target_img.shape, warped_img.dtype )

        warped_images[i] = warped_img
        target_images[i] = target_img

    return warped_images, target_images
