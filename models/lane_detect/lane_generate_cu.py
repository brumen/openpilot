import os
import numpy as np
import logging
import cv2

from typing     import Tuple

# from openpilot.models.lane_detect.hough_lines        import HoughLanesImage
from openpilot.models.lane_detect.lane_generate_base import ImageGenerateBase
from openpilot.models.lane_detect.train_lanes        import TrainLanesCULane, TrainLanesTuSimple


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageAndLaneMarkingsCULane(ImageGenerateBase):
    """ Generates Images and lanes
    """

    def __init__(self, base_dir):
        super().__init__(base_dir)

        # cached values - the train lanes object
        self.__train_lanes_obj = None

    @property
    def train_lanes_class(self):
        return TrainLanesCULane

    @property
    def train_lanes_obj(self):
        """ class for retrieving the pre-constructed lanes.

        :returns: the object handling the rendering of the image, and the lane superimposed image.
        """

        if self.__train_lanes_obj:
            return self.__train_lanes_obj

        self.__train_lanes_obj = self.train_lanes_class(self.base_dir)
        return self.__train_lanes_obj

    def train_lanes(self, idx : Tuple[str, str]) -> np.ndarray:
        """ Constructs a marked lane image (array) name from index.

        :param idx: tuple of folder dir/filename.
        """

        folder, fname = idx
        return self.train_lanes_obj.new_image(os.path.join(folder, f'{fname}.jpg'), self._image_shape(idx))

    def show_movie_with_lanes(self):

        for dir_name_file_nb in self:  # this is the iterator, dir_name_file_nb is a tuple (directory, file_name)
            logger.info(f'Displaying file {dir_name_file_nb}')
            orig_image = self._image_from_idx(dir_name_file_nb)
            lane_image = cv2.cvtColor(self.train_lanes_obj.new_image(dir_name_file_nb, orig_image.shape[:2]) * 255, cv2.COLOR_GRAY2RGB)
            cv2.imshow('lines', cv2.addWeighted(orig_image, 0.6, lane_image, 0.8, 0))
            cv2.waitKey(100)

    # def _roi_region(self) -> List[Tuple[int, int]]:
    #     """ Generates the roi region for the image.
    #     """
    #
    #     # [[617, 448], [1080, 448], [726, 266], [885, 266]]])
    #     return [(200, 500), (600, 250), (1000, 250), (1400, 500) ]

    # def construct_lanes_hough(self, idx : Tuple[str, str]):
    #     """ Constructs the lane image using hough_lines
    #         Image size is: # 590, 1640
    #
    #     :param idx: index tuple to get the hough lines from.
    #     """
    #
    #     image = self._image_from_idx(idx)
    #     cl = HoughLanesImage(image, self._roi_region())
    #
    #     return np.array(cl.show_lines(self.image_shape(image))).astype(np.uint8)


class ImageAndLaneMarkingsTUSimple(ImageAndLaneMarkingsCULane):

    @property
    def train_lanes_class(self):
        return TrainLanesTuSimple
