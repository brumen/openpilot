# old lane generator class.
import os
import numpy as np
import cv2
import logging

from typing  import Tuple

from openpilot.models.lane_detect.construct_lanes import ConstructLanesFromImageHugh, TrainLanesCULane
from openpilot.models.lane_detect.lane_generate_base import ImageGenerateBase
from openpilot.models.lane_detect.lane_generate_cu import ImageAndLaneMarkingsCULane

import sys
sys.path.append('/home/brumen/work/')


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = '/home/brumen/data/openpilot/tusimple/clips/0313-1'


class ImageAndLaneMarkingsTUSimple(ImageAndLaneMarkingsCULane):

    def __init__(self, base_dir, lane_fname='LLL'):
        super().__init__(base_dir)

        self._lane_fname = lane_fname

    def _precomputed_lane_file(self, idx : Tuple[str, str]) -> str:
        dir_name, image_idx = idx

        return os.path.join(self.base_dir, dir_name, f'{image_idx}.lines.npy')

    def _raw_lane_file(self, idx : Tuple[str, str]) -> str:
        # TODO: CHANGE THIS HERE

        dir_name, image_idx = idx

        return os.path.join(self.base_dir, dir_name, f'{image_idx}.lines.txt')

    @property
    def _train_lanes_class(self):
        return None  # TODO: FIX THIS HERE.

    def constructed_lanes(self, idx, train_lanes_class) -> np.ndarray:
        super().constructed_lanes(idx, train_lanes_class=train_lanes_class)

    def show_movie_with_lanes(self):

        for dir_file_name in self.generate_image_names():
            curr_image = cv2.imread(self._image_name_from_idx(dir_file_name))
            lane_detected_image = self.construct_lanes(curr_image)
            lane_detected_image = cv2.cvtColor(lane_detected_image*255, cv2.COLOR_GRAY2RGB)
            cv2.imshow('lines', cv2.addWeighted(curr_image, 0.6, lane_detected_image, 0.8, 0))
            cv2.waitKey(10)
