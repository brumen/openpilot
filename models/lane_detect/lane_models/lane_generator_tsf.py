# transformed version of the lane generator

import cv2
import numpy as np
import logging

from typing import Union

from openpilot.models.lane_detect.lane_generator    import LaneGeneratorCU, LaneGeneratorTU
from openpilot.models.lane_detect.lane_generator_ts import LaneGeneratorCUTS, LaneGeneratorTUTS

logging.basicConfig(level=logging.INFO)


class ImageShrinkMixin:
    """ Image shrink class.
    """

    def _process_X(self, orig_image) -> Union[None, np.ndarray]:

        first_tsf = super()._process_X(orig_image)

        if self.scale_img == 1.:  # no shrinking:
            return first_tsf

        # scale the image
        new_width = int(first_tsf.shape[1] * self.scale_img)
        new_height = int(first_tsf.shape[0] * self.scale_img)

        return  cv2.resize(first_tsf, (new_width, new_height), interpolation = cv2.INTER_AREA)

    def _process_y(self, line_image):
        first_tsf = super()._process_y(line_image)

        if self.scale_img == 1.:
            return first_tsf

        # rescale the image
        new_width = int(first_tsf.shape[1] * self.scale_img)
        new_height = int(first_tsf.shape[0] * self.scale_img)

        return cv2.resize(first_tsf, (new_width, new_height), interpolation=cv2.INTER_AREA)


class LaneGeneratorCUShrink(ImageShrinkMixin, LaneGeneratorCU):

    def __init__( self
                , base_dir         : str
                , batch_size       : int   = 32
                , train_percentage : float = .8
                , to_train         : bool  = True
                , scale_img        : float = 0.5 ):

        super().__init__(base_dir, batch_size, train_percentage, to_train)

        self.scale_img = scale_img


class LaneGeneratorTUShrink(ImageShrinkMixin, LaneGeneratorTU):

    def __init__( self
                , base_dir         : str
                , batch_size       : int   = 32
                , train_percentage : float = .8
                , to_train         : bool  = True
                , scale_img        : float = 0.5 ):

        super().__init__(base_dir, batch_size, train_percentage, to_train)

        self.scale_img = scale_img


class LaneGeneratorTUTSShrink(ImageShrinkMixin, LaneGeneratorTUTS):

    def __init__( self
                , base_dir         : str
                , batch_size       : int   = 32
                , train_percentage : float = .8
                , to_train         : bool  = True
                , nb_time_steps    : int   = 10
                , scale_img        : float = 0.2):

        super().__init__(base_dir, batch_size, train_percentage, to_train, nb_time_steps)

        self.scale_img = scale_img


class LaneGeneratorCUTSShrink(ImageShrinkMixin, LaneGeneratorCUTS):

    def __init__( self
                , base_dir         : str
                , batch_size       : int   = 32
                , train_percentage : float = .8
                , to_train         : bool  = True
                , nb_time_steps    : int   = 10
                , scale_img        : float = 0.2):

        super().__init__(base_dir, batch_size, train_percentage, to_train, nb_time_steps)

        self.scale_img = scale_img
