from typing import Tuple, List

from tensorflow.keras.layers import TimeDistributed, ConvLSTM2D

from openpilot.models.lane_detect.lane_models.encoder_4 import encoder_4, decoder_4, encoder_5, decoder_5


def model_5_ts(input_shape : Tuple[int, int, int, int], pool_size : Tuple[int, int] = (1, 1) ) -> List:
    """ Constructs the convolutional LSTM layer w/ decoder & encoder.

    :param input_shape: input shape (nb_steps, height, width, nb_channels)
    :param pool_size: pool size for the sampling layers.
    :returns: list of layers for the model, used in the Sequential model
    """

    encoder_ts = [ TimeDistributed(layer, input_shape=input_shape)
                   for layer in encoder_5(None, pool_size=pool_size) ]

    decoder_ts = [TimeDistributed(layer) for layer in decoder_5(pool_size = pool_size)]

    return encoder_ts +\
           [ ConvLSTM2D( filters          = 32
                       , kernel_size      = (3, 3)
                       , data_format      = 'channels_last'
                       , padding          = 'same'
                       , return_sequences = True)
           , ConvLSTM2D( filters          = 32
                       , kernel_size      = (3, 3)
                       , data_format      = 'channels_last'
                       , padding          = 'same'
                       , return_sequences = True) ] +\
           decoder_ts


# from openpilot.models.lane_detect.lane_models.model_base import model_from_layers
# model = model_from_layers(model_5_ts((10, 100, 200, 3)))
# print(model.summary())
