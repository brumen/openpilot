# transformed version of the lane generator

import cv2
import numpy as np
import logging

from typing import Union, Optional, Tuple

from openpilot.models.lane_detect.lane_generator    import LaneGeneratorCU, LaneGeneratorTU
from openpilot.models.lane_detect.lane_generator_ts import LaneGeneratorCUTS, LaneGeneratorTUTS

logging.basicConfig(level=logging.INFO)


class ImageCropMixin:
    """ Crops and scales the image from the original image.
    """

    def _process_X(self, orig_image) -> Union[None, np.ndarray]:

        first_tsf = super()._process_X(orig_image)

        if self.crop_xy is None:
            crop_img = first_tsf
        else:
            crop_x, crop_y = self.crop_xy
            crop_img = first_tsf[crop_y[0]:crop_y[1], crop_x[0]:crop_x[1], :]

        # Do the scale tsf.
        if self.scale_img == 1.:  # no shrinking:
            return crop_img

        # scale the image
        new_width = int(crop_img.shape[1] * self.scale_img)
        new_height = int(crop_img.shape[0] * self.scale_img)

        return cv2.resize(crop_img, (new_width, new_height), interpolation = cv2.INTER_AREA)

    def _process_y(self, line_image):
        first_tsf = super()._process_y(line_image)

        if self.crop_xy is None:
            crop_img = first_tsf
        else:
            # continue cropping
            crop_x, crop_y = self.crop_xy

            crop_img = first_tsf[crop_y[0]:crop_y[1], crop_x[0]:crop_x[1]]

        if self.scale_img == 1.:
            return crop_img

        # rescale the image
        new_width = int(crop_img.shape[1] * self.scale_img)
        new_height = int(crop_img.shape[0] * self.scale_img)

        return cv2.resize(crop_img, (new_width, new_height), interpolation=cv2.INTER_AREA)


TU_crop = ((0, 1280), (260, 720) )  # [(0, 460), (0, 720), (1280, 720), (1280, 460), (840, 260), (400, 260)]  # roi vertices


class LaneGeneratorCUCrop(ImageCropMixin, LaneGeneratorCU):

    def __init__( self
                , base_dir         : str
                , batch_size       : int   = 32
                , train_percentage : float = .8
                , to_train         : bool  = True
                , crop_xy          : Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = TU_crop
                , scale_img        : Optional[float] = 1. ):

        super().__init__(base_dir, batch_size, train_percentage, to_train)

        self.crop_xy   = crop_xy
        self.scale_img = scale_img


class LaneGeneratorTUCrop(ImageCropMixin, LaneGeneratorTU):

    def __init__( self
                , base_dir         : str
                , batch_size       : int   = 32
                , train_percentage : float = .8
                , to_train         : bool  = True
                , crop_xy          : Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = TU_crop
                , scale_img: Optional[float] = 1. ):

        super().__init__(base_dir, batch_size, train_percentage, to_train)

        self.crop_xy   = crop_xy
        self.scale_img = scale_img


class LaneGeneratorTUTSCrop(ImageCropMixin, LaneGeneratorTUTS):

    def __init__( self
                , base_dir         : str
                , batch_size       : int   = 32
                , train_percentage : float = .8
                , to_train         : bool  = True
                , nb_time_steps    : int   = 10
                , crop_xy          : Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = TU_crop
                , scale_img: Optional[float] = 1.):

        super().__init__(base_dir, batch_size, train_percentage, to_train, nb_time_steps)

        self.crop_xy   = crop_xy
        self.scale_img = scale_img


class LaneGeneratorCUTSCrop(ImageCropMixin, LaneGeneratorCUTS):

    def __init__( self
                , base_dir         : str
                , batch_size       : int   = 32
                , train_percentage : float = .8
                , to_train         : bool  = True
                , nb_time_steps    : int   = 10
                , crop_xy          : Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = TU_crop
                , scale_img: Optional[float] = 1. ):
        super().__init__(base_dir, batch_size, train_percentage, to_train, nb_time_steps)

        self.crop_xy   = crop_xy
        self.scale_img = scale_img
