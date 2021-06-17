import os
import numpy as np
import cv2
import logging

from typing  import Union, Tuple

from openpilot.models.lane_detect.hough_lines import HoughLanesImage
from openpilot.models.lane_detect.lane_generator import LaneGeneratorCU, LaneGeneratorTU

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LaneGeneratorCUHough(LaneGeneratorCU):

    def _process_X(self, orig_image) -> Union[None, np.ndarray]:
        """ Processes the original image.
        """

        if orig_image is None:
            return None

        cl = HoughLanesImage( orig_image, [(200, 500), (600, 250), (1000, 250), (1400, 500)])

        # adding color
        lane_image = cv2.cvtColor( cl.preprocess_image(orig_image).astype(np.uint8) * 255
                                 , cv2.COLOR_GRAY2RGB)

        scale_image = None

        if scale_image is None:
            return np.array(lane_image, dtype=np.uint8)

        # resize the image
        nb_rows, nb_cols = lane_image.shape[:2]
        return cv2.resize( lane_image, (int(nb_cols * scale_image), int(nb_rows * scale_image) ) )

    def _process_y(self, line_image):
        """ Processes the line image given.
        """

        if line_image is None:
            return None

        line_image = cv2.cvtColor(line_image.astype(np.uint8) * 255, cv2.COLOR_GRAY2RGB)

        scale_image = None

        if scale_image is None:
            return np.array(line_image, dtype=np.uint8)

        # resize the image
        nb_rows, nb_cols = line_image.shape[:2]
        return cv2.resize(line_image.astype(np.uint8), (int(nb_cols * scale_image), int(nb_rows * scale_image) ) )


class LaneGeneratorCUHough2(LaneGeneratorCUHough):

    def _process_X(self, orig_image) -> Union[None, np.ndarray]:
        """

        :param orig_image: original image of shape (image_x, image_y, nb_channels)
        :returns: image of the same shape
        """

        if orig_image is None:
            return None

        roi_vertex = [(200, 500), (600, 250), (1000, 250), (1400, 500)]
        cl = HoughLanesImage( orig_image, roi_vertex)

        image_shape = orig_image.shape[:2]
        hough_lines = cl.show_lines(image_shape).astype(np.uint8) * 255
        hough_lines = cv2.cvtColor(hough_lines, cv2.COLOR_GRAY2RGB)

        lane_image = cv2.addWeighted(orig_image, 0.6, hough_lines, 0.8, 0)
        # adding color

        scale_image = None

        if scale_image is None:
            return np.array(lane_image, dtype=np.uint8)

        # resize the image
        nb_rows, nb_cols = lane_image.shape[:2]
        return cv2.resize( lane_image, (int(nb_cols * scale_image), int(nb_rows * scale_image) ) )

# examples
def example_3():
    # new_image_size = (590, 1640, 3)
    scale_size = 1.
    batch_size = 32
    train_percentage = 0.8

    from openpilot.models.lane_detect.lane_config import BASE_BASE

    train_generator = LaneGeneratorCUHough( BASE_BASE
                                     , to_train = True
                                     , train_percentage  = train_percentage
                                     , batch_size=batch_size )

#    for x in train_generator:
#        print(x)


    train_generator.show_movie()

example_3()