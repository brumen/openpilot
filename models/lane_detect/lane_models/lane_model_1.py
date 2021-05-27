""" This file contains code for a fully convolutional
(i.e. contains zero fully connected layers) neural network
for detecting lanes. This version assumes the inputs
to be road images in the shape of 80 x 160 x 3 (RGB) with
the labels as 80 x 160 x 1 (just the G channel with a
re-drawn lane). Note that in order to view a returned image,
the predictions is later stacked with zero'ed R and B layers
and added back to the initial road image.
"""

# TODO: ADD THIS TO THE LIBRARY PATH FOR KERAS TO WORK W/ GPU
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/targets/x86_64-linux/lib/

import tensorflow as tf
import cv2
import logging

logging.basicConfig(level=logging.INFO)

from typing  import Tuple

from keras.models               import Sequential
from keras.layers               import Dropout, UpSampling2D, Conv2DTranspose, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

from openpilot.models.lane_detect.lane_config    import BASE_DIR
from openpilot.models.lane_detect.old.lane_generator import LaneGenerator


logger = logging.getLogger(__name__)


def model_from_layers(layers):

    model = Sequential()
    for layer in layers:
        model.add(layer)

    return model


def lane_model_1(input_shape : Tuple[int, int ,int], pool_size : Tuple[int, int] = (1, 1) ):
    """ First attempt at a lane model.

    :param input_shape: input shape of the image, e.g. (590, 1640, 3)
    :param pool_size: pool size for the layer

    """

    layers = [ BatchNormalization(input_shape=input_shape)
             , Conv2D(8, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv1')
             , Conv2D(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv2')
             , MaxPooling2D(pool_size=pool_size)
             , Conv2D(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv3')
             , Dropout(0.2)
             , Conv2D(32, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv4')
             , Dropout(0.2)
             , Conv2D(32, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv5')
             , Dropout(0.2)
             , MaxPooling2D(pool_size=pool_size)
             , Conv2D(64, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv6')
             , Dropout(0.2)
             , Conv2D(64, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv7')
             , Dropout(0.2)
             , MaxPooling2D(pool_size=pool_size)
             , UpSampling2D(size=pool_size)
             , Conv2DTranspose(64, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv1')
             , Dropout(0.2)
             , Conv2DTranspose(64, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv2')
             , Dropout(0.2)
             , UpSampling2D(size=pool_size)
             , Conv2DTranspose(32, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv3')
             , Dropout(0.2)
             , Conv2DTranspose(32, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv4')
             , Dropout(0.2)
             , Conv2DTranspose(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv5')
             , Dropout(0.2)
             , UpSampling2D(size=pool_size)
             , Conv2DTranspose(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv6')
             , Conv2DTranspose(1, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Final')
             , ]

    return model_from_layers(layers)


def lane_model_2(input_shape : Tuple[int, int ,int], pool_size : Tuple[int, int] = (1, 1) ):
    """ First attempt at a lane model.

    :param input_shape: input shape of the image, e.g. (590, 1640, 3)
    :param pool_size: pool size for the layer

    """

    layers = [ BatchNormalization(input_shape=input_shape)
             , Conv2D(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv3')
             , Dropout(0.2)
             #, MaxPooling2D(pool_size=pool_size)
             #, UpSampling2D(size=pool_size)
             #, Dropout(0.2)
             , Conv2DTranspose(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv5')
             , Dropout(0.2)
             , UpSampling2D(size=pool_size)
             , Conv2DTranspose(1, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Final')
             , ]

    return model_from_layers(layers)


def lane_model_3(input_shape : Tuple[int, int ,int], pool_size : Tuple[int, int] = (1, 1) ):
    """ First attempt at a lane model.

    :param input_shape: input shape of the image, e.g. (590, 1640, 3)
    :param pool_size: pool size for the layer

    """

    layers = [ BatchNormalization(input_shape=input_shape)
             , Conv2D(8, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv1')
             , Conv2D(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv2')
             , MaxPooling2D(pool_size=pool_size)
             , Conv2D(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv3')
             , Dropout(0.2)
             , MaxPooling2D(pool_size=pool_size)
             , UpSampling2D(size=pool_size)
             , Conv2DTranspose(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv5')
             , Dropout(0.2)
             , UpSampling2D(size=pool_size)
             , Conv2DTranspose(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv6')
             , Conv2DTranspose(1, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Final')
             , ]

    return model_from_layers(layers)


def lane_model_4(input_shape : Tuple[int, int ,int], pool_size : Tuple[int, int] = (1, 1) ):
    """ First attempt at a lane model.

    :param input_shape: input shape of the image, e.g. (590, 1640, 3)
    :param pool_size: pool size for the layer

    """

    layers = [ BatchNormalization(input_shape=input_shape)
             , Conv2D(8, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv1')
             , Conv2D(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv2')
             , MaxPooling2D(pool_size=pool_size)
             , Conv2D(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv3')
             , Dropout(0.2)
             , Conv2D(32, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv4')
             , Dropout(0.2)
             , Conv2D(32, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv5')
             , Dropout(0.2)
             , MaxPooling2D(pool_size=pool_size)
             , UpSampling2D(size=pool_size)
             , Conv2DTranspose(32, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv3')
             , Dropout(0.2)
             , Conv2DTranspose(32, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv4')
             , Dropout(0.2)
             , Conv2DTranspose(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv5')
             , Dropout(0.2)
             , UpSampling2D(size=pool_size)
             , Conv2DTranspose(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv6')
             , Conv2DTranspose(1, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Final')
             , ]

    return model_from_layers(layers)



def get_model(base_dir = BASE_DIR, scale_size = 0.2, train_percentage = 0.003, batch_size=32):
    new_image_size = (int(590 * scale_size), int(1640 * scale_size), 3)

    train_generator = LaneGenerator(base_dir
                                    , to_train = True
                                    , train_percentage  = train_percentage
                                    , scale_image=scale_size
                                    , batch_size=batch_size)
    valid_generator = LaneGenerator(base_dir
                                    , to_train = False
                                    , train_percentage = 1. - train_percentage
                                    , scale_image = scale_size
                                    , batch_size = batch_size )

    # model = lane_model_1( (590, 1640, 3) )
    model = lane_model_4( new_image_size )

    model.compile(optimizer = 'adam'
                  , loss    = tf.keras.losses.BinaryCrossentropy() # emphasize_white  # tf.keras.losses.MeanSquaredError()  # emphasize_white  #  Hinge()  # emphasize_white
                  , metrics = ['accuracy', ]
                  , )

    return model, train_generator, valid_generator


def fit(model, train_generator, valid_generator):

    model.fit(train_generator)

    return model

model, train_gen, valid_gen = get_model(base_dir = BASE_DIR, train_percentage=0.1, batch_size=16)
model.fit(train_gen)

# prediction
input_1, result_1 = valid_gen[0]
model_1 = model.predict(input_1)  # this is a collection of 32 images, since batch size is 32

# model_1 and result_1 should be close
# from openpilot.models.lane_detect.extract_coords import show_image

#from matplotlib import pyplot as plt
#plt.imshow(result_1[0])
#plt.show()

#plt.imshow(model_1[0][:,:,0])
#plt.show()

def compare_results_visual(nb_inputs, nb_images):
    """ nb_images should equal to the batch size
        nb_inputs should equal to the

    """

    for idx in range(nb_inputs):
        input_1, result_1 = valid_gen[idx]
        model_1 = model.predict(input_1)

        for image_nb in range(nb_images):
            curr_image = input_1[image_nb]
            cv2.imshow('lines', cv2.addWeighted(input_1[image_nb], 0.6, model_1[image_nb][:,:,0], 0.8, 0))

            cv2.waitKey(100)  # IMPORTANT LINE, DO NOT DELETE

compare_results_visual(10, 16)
