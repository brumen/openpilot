# model 5 working on a single image
import logging

logging.basicConfig(level=logging.INFO)

from typing  import Tuple

from keras.layers               import Dense

from openpilot.models.lane_detect.lane_models.model_base import model_from_layers
from openpilot.models.lane_detect.lane_models.encoder_4 import encoder_4, decoder_4

logger = logging.getLogger(__name__)


def lane_model_5(input_shape : Tuple[int, int ,int], pool_size : Tuple[int, int] = (1, 1) ):
    """ First attempt at a lane model.

    :param input_shape: input shape of the image, e.g. (590, 1640, 3)
    :param pool_size: pool size for the layer

    """
    encoder = encoder_4(input_shape, pool_size=pool_size)
    decoder = decoder_4(pool_size)

    # layers in between
    between_layers = [ Dense(31, name='Between_1')
                     , Dense(30, name='Between_2')
                     , ]

    return model_from_layers(encoder + between_layers + decoder)
