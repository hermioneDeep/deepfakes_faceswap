from lib.alignments import Alignments
from lib.faces_detect import DetectedFace
from lib.image import read_image_hash_batch
from training_data import TrainingDataGenerator
from lib.utils import FaceswapError
from plugins.train._config import Config
from model import input_shape
class Batcher():
    """ Handles the processing of a Batch for a single side.

    Parameters
    ----------
    side: {"a" or "b"}
        The side that this :class:`Batcher` belongs to
    images: list
        The list of full paths to the training images for this :class:`Batcher`
    model: plugin from :mod:`plugins.train.model`
        The selected model that will be running this trainer
    use_mask: bool
        ``True`` if a mask is required for training otherwise ``False``
    batch_size: int
        The size of the batch to be processed at each iteration
    config: :class:`lib.config.FaceswapConfig`
        The configuration for this trainer
    """
    def __init__(self, side, images, model, use_mask, batch_size, config):
        logger.debug("Initializing %s: side: '%s', num_images: %s, use_mask: %s, batch_size: %s, "
                     "config: %s)",
                     self.__class__.__name__, side, len(images), use_mask, batch_size, config)
        self._model = model
        self._use_mask = use_mask
        self._side = side
        self._images = images
        self._config = config
        self._target = None
        self._samples = None
        self._masks = None

        generator = self._load_generator()
        self._feed = generator.minibatch_ab(images, batch_size, self._side)

        self._preview_feed = None
        self._timelapse_feed = None
        self._set_preview_feed()

    def load_generator(self):
        """ Load the :class:`lib.training_data.TrainingDataGenerator` for this batcher """
        print("Loading generator...")
        input_size = input_shape[0]
        output_shapes = input_shape
        generator = TrainingDataGenerator(input_size,
                                          output_shapes,
                                          0.625)
        return generator

    def get_next(self):
        """ Return the next batch from the :class:`lib.training_data.TrainingDataGenerator` for
        this batcher ready for feeding into the model.

        Returns
        -------
        model_inputs: list
            A list of :class:`numpy.ndarray` for feeding into the model
        model_targets: list
            A list of :class:`numpy.ndarray` for comparing the output of the model
        """
        logger.trace("Generating targets")
        batch = next(self._feed)
        targets_use_mask = self._model.training_opts["learn_mask"]
        model_inputs = batch["feed"] + batch["masks"] if self._use_mask else batch["feed"]
        model_targets = batch["targets"] + batch["masks"] if targets_use_mask else batch["targets"]
        return model_inputs, model_targets
