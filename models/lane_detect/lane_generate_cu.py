import os
import numpy as np
import logging

from typing     import Tuple, List

from openpilot.models.lane_detect.construct_lanes    import HoughLanesImage
from openpilot.models.lane_detect.lane_generate_base import ImageGenerateBase
from openpilot.models.lane_detect.train_lanes        import TrainLanesCULane


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageAndLaneMarkingsCULane(ImageGenerateBase):
    """ Generates Images and lanes
    """

    def _precomputed_lane_file(self, idx : Tuple[str, str]) -> str:
        """ Returns the precomputed lane file.

        :param idx: tuple of directory name, image_idx, like 00775
        :returns: the name of the precomputed lines matrix file.
        """

        dir_name, image_idx = idx

        return os.path.join(dir_name, f'{image_idx}.lines.mtx.npy')

    def _raw_lane_file(self, idx : Tuple[str, str]) -> str:
        dir_name, image_idx = idx

        return os.path.join(dir_name, f'{image_idx}.lines.txt')

    @property
    def _train_lanes_class(self):
        return TrainLanesCULane

    def train_lanes(self, idx : Tuple[str, str]) -> np.ndarray:
        """ Constructs a marked lane image (array) name from index.

        :param idx: tuple of folder dir/filename.
        """

        # try to load the preconstructed image.
        preconstructed_lane_file = self._precomputed_lane_file(idx)
        if os.path.isfile(preconstructed_lane_file):
            return np.load(preconstructed_lane_file)

        # construct it from the lane image, save it to the location, return the image
        folder, fname = idx
        cl = self._train_lanes_class(self.base_dir, folder, f'{fname}.lines.txt')
        gen_image = cl.new_image(self._image_shape(idx))
        np.save(preconstructed_lane_file, gen_image)  # saves the precomputed image

        return gen_image

    def _roi_region(self) -> List[Tuple[int, int]]:
        """ Generates the roi region for the image.
        """

        # [[617, 448], [1080, 448], [726, 266], [885, 266]]])
        return [(200, 500), (600, 250), (1000, 250), (1400, 500) ]

    def construct_lanes_hough(self, idx : Tuple[str, str]):
        """ Constructs the lane image using hough_lines
            Image size is: # 590, 1640

        :param idx: index tuple to get the hough lines from.
        """

        image = self._image_from_idx(idx)
        cl = HoughLanesImage(image, self._roi_region())

        return np.array(cl.show_lines(self.image_shape(image))).astype(np.uint8)
