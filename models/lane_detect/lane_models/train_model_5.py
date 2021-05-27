import tensorflow as tf
import logging

logging.basicConfig(level=logging.INFO)

from openpilot.models.lane_detect.lane_config       import BASE_BASE, BASE_DIR3
from openpilot.models.lane_detect.lane_generator_2    import LaneGeneratorCU
from openpilot.models.lane_detect.lane_generator_ts import LaneGeneratorTS

from openpilot.models.lane_detect.lane_models.model_5    import lane_model_5
from openpilot.models.lane_detect.lane_models.model_5_ts import model_5_ts
from openpilot.models.lane_detect.lane_models.model_base import model_from_layers

logger = logging.getLogger(__name__)


def get_model(base_dir = BASE_BASE, scale_size = 0.2, train_percentage = 0.003, batch_size=32):
    new_image_size = (int(590 * scale_size), int(1640 * scale_size), 3)

    train_generator = LaneGeneratorCU(base_dir
                                    , to_train = True
                                    , train_percentage  = train_percentage
                                    , scale_image=scale_size
                                    , batch_size=batch_size)
    valid_generator = LaneGeneratorCU(base_dir
                                    , to_train = False
                                    , train_percentage = 1. - train_percentage
                                    , scale_image = scale_size
                                    , batch_size = batch_size)

    model = lane_model_5( new_image_size )

    model.compile(optimizer = 'adam'
                  , loss    = tf.keras.losses.BinaryCrossentropy()
                  , metrics = ['accuracy', ]
                  , )

    return model, train_generator, valid_generator


def get_model_ts(base_dir = BASE_BASE, scale_size = 0.2, train_percentage = 0.003, batch_size=32):
    new_image_size = (int(590 * scale_size), int(1640 * scale_size), 3)
    new_image_size = (5, int(590 * scale_size), int(1640 * scale_size), 3)

    train_generator = LaneGeneratorTS( base_dir
                                     , to_train = True
                                     , train_percentage  = train_percentage
                                     , scale_image=scale_size
                                     , batch_size=batch_size
                                     , nb_time_steps=5 )

    valid_generator = LaneGeneratorTS( base_dir
                                     , to_train = False
                                     , train_percentage = 1. - train_percentage
                                     , scale_image = scale_size
                                     , batch_size = batch_size
                                     , nb_time_steps= 5)

    # model = lane_model_1( (590, 1640, 3) )
    # model = lane_model_5( new_image_size )
    model = model_from_layers(model_5_ts(new_image_size))

    model.compile(optimizer = 'adam'
                  , loss    = tf.keras.losses.BinaryCrossentropy()
                  , metrics = ['accuracy', ]
                  , )

    return model, train_generator, valid_generator


# can be replaced w/ get_model
model, train_gen, valid_gen = get_model(base_dir = BASE_DIR3, train_percentage=0.1, batch_size=3)
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
