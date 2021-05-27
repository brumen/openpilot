import logging

logging.basicConfig(level=logging.INFO)

from typing  import Tuple, List, Optional

from keras.layers               import Dropout, UpSampling2D, Conv2DTranspose, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

logger = logging.getLogger(__name__)


def encoder_4( input_shape         : Optional[Tuple[int, int, int]]
             , pool_size           : Tuple[int, int] = (1, 1) ) -> List:
    """ Encoder of the lane model.

    :param input_shape: input shape of the image, e.g. height, width, nb_channels: (590, 1640, 3), or None
    :param pool_size: pool size for the layer
    :returns: list of kernel layers which encode the
    """

    return [ BatchNormalization(input_shape=input_shape) if input_shape is not None else BatchNormalization()
             , Conv2D(8, (3, 3), padding='valid', strides=(1, 1), activation = 'relu', name = 'Conv1')
             , Conv2D(16, (3, 3), padding='valid', strides=(1, 1), activation = 'relu', name = 'Conv2')
             , MaxPooling2D(pool_size=pool_size)
             , Conv2D(16, (3, 3), padding='valid', strides=(1, 1), activation = 'relu', name = 'Conv3')
             , Dropout(0.2)
             , Conv2D(32, (3, 3), padding='valid', strides=(1, 1), activation = 'relu', name = 'Conv4')
             , Dropout(0.2)
             , Conv2D(32, (3, 3), padding='valid', strides=(1, 1), activation = 'relu', name = 'Conv5')
             , Dropout(0.2)
             , MaxPooling2D(pool_size=pool_size)
             , ]


def decoder_4(pool_size : Tuple[int, int] = (1, 1) ):
    return [ UpSampling2D(size=pool_size)
             , Conv2DTranspose(32, (3, 3), padding='valid', strides=(1, 1), activation = 'relu', name = 'Deconv3')
             , Dropout(0.2)
             , Conv2DTranspose(32, (3, 3), padding='valid', strides=(1, 1), activation = 'relu', name = 'Deconv4')
             , Dropout(0.2)
             , Conv2DTranspose(16, (3, 3), padding='valid', strides=(1, 1), activation = 'relu', name = 'Deconv5')
             , Dropout(0.2)
             , UpSampling2D(size=pool_size)
             , Conv2DTranspose(16, (3, 3), padding='valid', strides=(1, 1), activation = 'relu', name = 'Deconv6')
             , Conv2DTranspose(1, (3, 3), padding='valid', strides=(1, 1), activation = 'relu', name = 'Final')
             , ]


def encoder_5( input_shape         : Optional[Tuple[int, int, int]]
             , pool_size           : Tuple[int, int] = (1, 1) ) -> List:
    """ Encoder of the lane model.

    :param input_shape: input shape of the image, e.g. height, width, nb_channels: (590, 1640, 3), or None
    :param pool_size: pool size for the layer
    :returns: list of kernel layers which encode the
    """

    return [ BatchNormalization(input_shape=input_shape) if input_shape is not None else BatchNormalization()
             , Conv2D(8, (3, 3), padding='valid', strides=(1, 1), activation = 'relu', name = 'Conv1')
             , Conv2D(16, (3, 3), padding='valid', strides=(1, 1), activation = 'relu', name = 'Conv2')
             , MaxPooling2D(pool_size=pool_size)
             , Conv2D(16, (3, 3), padding='valid', strides=(1, 1), activation = 'relu', name = 'Conv3')
             , Dropout(0.2)
             , MaxPooling2D(pool_size=pool_size)
             , ]


def decoder_5(pool_size : Tuple[int, int] = (1, 1) ):
    return [ UpSampling2D(size=pool_size)
             , Conv2DTranspose(16, (3, 3), padding='valid', strides=(1, 1), activation = 'relu', name = 'Deconv5')
             , Dropout(0.2)
             , UpSampling2D(size=pool_size)
             , Conv2DTranspose(16, (3, 3), padding='valid', strides=(1, 1), activation = 'relu', name = 'Deconv6')
             , Conv2DTranspose(1, (3, 3), padding='valid', strides=(1, 1), activation = 'relu', name = 'Final')
             , ]
