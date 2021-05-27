import logging

logging.basicConfig(level=logging.INFO)

from keras.models import Sequential


logger = logging.getLogger(__name__)


def model_from_layers(layers):

    model = Sequential()
    for layer in layers:
        model.add(layer)

    return model
