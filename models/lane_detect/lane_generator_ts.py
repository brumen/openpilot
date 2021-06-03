# old lane generator class.
import numpy as np
import logging
import cv2

from typing  import Union, List, Tuple

from openpilot.models.lane_detect.lane_generator_2 import LaneGeneratorCU

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LaneGeneratorTS(LaneGeneratorCU):
    """ Generates a time series of lanes.
    """

    def __init__( self
                , base_dir         : str
                , batch_size       : int                = 32
                , train_percentage : float              = .8
                , to_train         : bool               = True
                , scale_image      : Union[None, float] = None
                , nb_time_steps    : int                = 10 ):

        super().__init__(base_dir, nb_time_steps, train_percentage, to_train, scale_image)

        self.batch_size_ts = batch_size
        self.nb_time_steps = nb_time_steps

    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        """ Get the next self.batch_size images

        :returns: list of tuples of X and y image
        """

        curr_ts = 0
        X_list = []
        y_list = []

        while curr_ts < self.batch_size_ts:
            X, y = super().__next__()
            X_list.append(X)
            y_list.append(y)
            curr_ts += 1

        return np.array(X_list), np.array(y_list)

    def show_movie(self):
        """ Shows the movie from images.
        """

        for X, y in self:
            # X, y of shape (batch_size, nb_time_steps, image_x, image_y, nb_channels)
            for batch_X, batch_y in zip(X, y):
                for X_time_step, y_time_step in zip(batch_X, batch_y):
                    cv2.imshow('lines2', cv2.addWeighted(X_time_step, 0.6, y_time_step, 0.8, 0))
                    cv2.waitKey(100)  # IMPORTANT line, DO NOT DELETE


def example_1():
    # new_image_size = (590, 1640, 3)
    scale_size = 1.
    batch_size = 32
    train_percentage = 0.8

    from openpilot.models.lane_detect.lane_config import BASE_DIR

    train_generator = LaneGeneratorTS( BASE_DIR
                                     , to_train = True
                                     , train_percentage  = train_percentage
                                     , scale_image=scale_size
                                     , batch_size=batch_size )

#    for x in train_generator:
#        print(x)

    train_generator.show_movie()

# example_1()
