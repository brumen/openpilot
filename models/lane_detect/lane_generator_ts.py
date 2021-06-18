import numpy as np
import logging
import cv2

from openpilot.models.lane_detect.lane_generator import LaneGeneratorCU, LaneGeneratorTU

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LaneGeneratorTSMixin:  # (LaneGeneratorCU):
    """ Generates a time series of lanes.
    """

    def __iter__(self):
        return self

    def __next__(self):
        """ Iterator over the frames of the movie.
        """

        curr_idx_batch = 0
        curr_time_step = 0
        X_list = []
        y_list = []
        X_list_ts = []
        y_list_ts = []

        while curr_time_step < self.nb_time_steps:
            curr_filename = next(self._file_iterator())

            if curr_idx_batch < self.batch_size_ts:
                X, y = self._generate_one_Xy(curr_filename)
                X_list_ts.append(X)
                y_list_ts.append(y)
                curr_idx_batch += 1

            else:
                X_list.append(np.array(X_list_ts))
                y_list.append(np.array(y_list_ts))
                curr_idx_batch = 0
                curr_time_step += 1
                X_list_ts = []
                y_list_ts = []

        return np.array(X_list), np.array(y_list)

    def show_movie_with_lanes(self, wait_between_frames : int = 100 ):
        """ Shows the movie from images.
        """

        for X, y in self:
            # X, y of shape (batch_size, nb_time_steps, image_x, image_y, nb_channels)
            for batch_X, batch_y in zip(X, y):
                for X_time_step, y_time_step in zip(batch_X, batch_y):
                    cv2.imshow('TS Video', cv2.addWeighted(X_time_step, 0.6, y_time_step, 0.8, 0))
                    cv2.waitKey(wait_between_frames)


class LaneGeneratorCUTS(LaneGeneratorTSMixin, LaneGeneratorCU):

    def __init__( self
                , base_dir         : str
                , batch_size       : int                = 32
                , train_percentage : float              = .8
                , to_train         : bool               = True
                , nb_time_steps    : int                = 10 ):

        super().__init__(base_dir, nb_time_steps, train_percentage, to_train)

        self.batch_size_ts = batch_size
        self.nb_time_steps = nb_time_steps


class LaneGeneratorTUTS(LaneGeneratorTSMixin, LaneGeneratorTU):

    def __init__( self
                , base_dir         : str
                , batch_size       : int                = 32
                , train_percentage : float              = .8
                , to_train         : bool               = True
                , nb_time_steps    : int                = 10 ):

        super().__init__(base_dir, nb_time_steps, train_percentage, to_train)

        self.batch_size_ts = batch_size
        self.nb_time_steps = nb_time_steps


def example_1():
    # new_image_size = (590, 1640, 3)
    batch_size = 32
    train_percentage = 0.8

    from openpilot.models.lane_detect.lane_config import BASE_CU

    train_generator = LaneGeneratorCUTS( BASE_CU
                                       , to_train = True
                                       , train_percentage  = train_percentage
                                       , batch_size=batch_size )

    train_generator.show_movie()


def example_2():
    # new_image_size = (590, 1640, 3)
    batch_size = 5
    train_percentage = 0.8

    from openpilot.models.lane_detect.lane_config import BASE_TU

    train_generator = LaneGeneratorTUTS( BASE_TU
                                       , to_train = True
                                       , train_percentage  = train_percentage
                                       , batch_size=batch_size
                                       , nb_time_steps= 5)

    train_generator.show_movie_with_lanes(wait_between_frames=2)

# example_2()
