import os
import logging

import cv2
import numpy as np
import json

from typing    import List, Tuple, Dict
from functools import lru_cache

from openpilot.models.lane_detect.base_lines import LanesBase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainLanesCULane:
    """Constructs the training lanes from a file for the CULane set.
    """

    BASE_DIR = '/home/brumen/data/openpilot/CULane/drive/'

    def __init__(self, base_dir : str = BASE_DIR ):
        """

        :param base_dir: base dir for the subsequent files to obtain
        """

        self.base_dir = base_dir

    def _get_lines(self, lane_descr_file : str ) -> List[List[Tuple[float, float]]]:
        """ Extract lines from a CU lane description file.

        :returns: list of lines, each entry for one line, each line consists of a sequence
              of xs and ys.
        """

        with open(lane_descr_file, 'r') as fopen:
            all_lanes = []
            for file_line in fopen.readlines():
                curr_line = []
                for x in file_line.split(' '):
                    try:
                        curr_line.append(float(x))
                    except Exception as e:
                        logger.debug(f'Ignoring value {x} in {lane_descr_file}: {str(e)}')

                all_lanes.append(self._get_xy_from_line(curr_line))

            return all_lanes

    def _get_xy_from_line(self, coords : List[float]) -> List[Tuple[float, float]]:
        """ Extracts the coordinates from list of coords, in sequence,
            first element is x, then y, and so on. E.g. 549, 320, 540, 310, ...

        :param coords: list of alternating x and y coordinates for the line.
        :returns: List of tuples of (x,y)
        """

        xy_coords = []

        while coords:  # list is not-Empty
            try:  # this might fail
                x = coords.pop(0)
                y = coords.pop(0)

            except Exception as e:  # finish the loop
                return xy_coords

            else:
                xy_coords.append((x, y))

        return xy_coords

    def _precompiled_name(self, lane_path_name : str) -> str:
        """ Constructs the precompiled path name from the lane path name

        """

        return lane_path_name.replace('.txt', '.mtx.npy')

    def _lane_path(self, jpg_path : str) -> str:
        """ Constructs the lane path file name from jpg path

        :param jpg_path:
        """

        return jpg_path.replace('.jpg', '.lines.txt')

    @lru_cache(maxsize=30)
    def new_image( self
                 , jpg_path   : str
                 , image_size : Tuple[int, int]
                 , pixel_tol  : int = 10) -> np.ndarray :
        """ lines_fname has .lines.txt in the name. creates and saves the new image from the old one.
            Saves the resulting image in the folder_dir/lines_fname.lines.mtx

        :param jpg_path: path to the image file for which lines are constructed, 0342343.mp4/00075.jpg
        :param image_size: size of the image
        :param pixel_tol: pixel tolerance, default = 10
        :returns: np array of the image with constructed lanes.
        """

        lane_path = self._lane_path(os.path.join(self.base_dir, jpg_path))

        # precompiled, only read it
        precompiled_path = self._precompiled_name(lane_path)
        if os.path.isfile(precompiled_path):
            return np.load(precompiled_path)

        # if nothing else, use render right there - completely new work
        return np.array(LanesBase(self._get_lines(lane_path)).show_lines(image_size, pixel_tol=pixel_tol))

    def show_image(self, jpg_path: str, image_size : Tuple[int, int], pixel_tol : int = 10):
        """ Shows the image specified.

        """

        loaded_image = self.new_image(jpg_path, image_size, pixel_tol=pixel_tol)
        tsf_image = cv2.cvtColor(loaded_image.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)

        cv2.imshow('CULane_image', tsf_image)
        cv2.waitKey(100)

    def superimpose_img(self, jpg_path : str, pixel_tol : int = 10):
        orig_image = cv2.imread(os.path.join(self.base_dir, jpg_path))
        lane_img   = self.new_image(jpg_path, orig_image.shape[:2], pixel_tol=pixel_tol)
        lane_img_tsf = cv2.cvtColor(lane_img.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
        combined = cv2.addWeighted(orig_image, 0.6, lane_img_tsf, 0.8, 0)

        cv2.imshow('CULane_superimpose', combined)
        cv2.waitKey(100)


class TrainLanesTuSimple(TrainLanesCULane):
    """ Get the lanes from the TUSimple dataset.
    """

    FILE_PREFIX = '/home/brumen/data/openpilot/tusimple'
    BASE_DIR    = '/home/brumen/data/openpilot/tusimple/clips'

    # TU files where the labels are stored.
    TU_FILE_1 = 'label_data_0313.json'
    TU_FILE_2 = 'label_data_0531.json'
    TU_FILE_3 = 'label_data_0601.json'

    COMBINED_JSON = 'label_all_data.json'  # where the results are saved, for future extraction

    def __init__(self, base_dir : str = BASE_DIR):
        """

        :param base_dir: base dir for the subsequent files to obtain (e.g.
        """
        super().__init__(base_dir)

        # cached value
        self.__lane_idx = None

    def _lane_path(self, jpg_path : str) -> str:
        """ Constructs the lane path file name from jpg path

        :param jpg_path: path to the jpg file (under the base_dir)
        :returns: path to the lane file (if needed)
        """

        return jpg_path

    def _get_lines(self, jpg_path : str) -> List[List[Tuple[float, float]]]:
        """ Gets the lanes for the jpg file.

        :param jpg_path: lane description path, in this case the same as the jpg_path 0313-1/23700/20.jpg
        :returns: list of lanes in the form of list of tuples of y,x
        """

        return self.lane_idx.get( jpg_path )

    @lru_cache(maxsize=30)
    def new_image( self
                 , jpg_path   : str
                 , image_size : Tuple[int, int]
                 , pixel_tol  : int = 10) -> np.ndarray :
        """ lines_fname has .lines.txt in the name. creates and saves the new image from the old one.
            Saves the resulting image in the folder_dir/lines_fname.lines.mtx

        :param jpg_path: path to the image file for which lines are constructed, 0342343.mp4/00075.jpg
        :param image_size: size of the image
        :param pixel_tol: pixel tolerance, default = 10
        :returns: np array of the image with constructed lanes.
        """

        new_jpg_path = os.path.join(*(jpg_path.split('/')[:-1]), '20.jpg')  # last one

        return np.array(LanesBase(self._get_lines(new_jpg_path)).show_lines(image_size, pixel_tol=pixel_tol))

    @property
    def lane_idx(self):

        if self.__lane_idx:
            return self.__lane_idx

        # read the lane index and do
        combined_file = os.path.join(self.FILE_PREFIX, self.COMBINED_JSON)

        if os.path.exists(combined_file):
            self.__lane_idx = json.load(open(combined_file, 'r'))
            return self.__lane_idx

        # reconstruct the dictionary
        lane_idx = self._construct_lane_dict(os.path.join(self.FILE_PREFIX, self.TU_FILE_1))
        lane_idx.update(self._construct_lane_dict(os.path.join(self.FILE_PREFIX, self.TU_FILE_2)))
        lane_idx.update(self._construct_lane_dict(os.path.join(self.FILE_PREFIX, self.TU_FILE_3)))

        json.dump(lane_idx, open(combined_file, 'w'))
        self.__lane_idx = lane_idx
        return self.__lane_idx

    @staticmethod
    def _construct_lane_dict(lane_fname : str) -> Dict[str, List[List[Tuple[int, int]]]]:
        """ Reads one of the lane files and constructs the dictionary

        :param lane_fname: filename of the lane from where to extract the lane indices, these are really just
                           the TU_FILE_1, TU_FILE_2, TU_FILE_3
        :returns: dictionary where:
                          keys: file names in the form 0313_1/01412412321/20.jpg
                          values: list of lanes, where each lane is a series of tuples of points (x,y)
        """

        lane_indices = {}

        with open(lane_fname, 'r') as lane_file_open:

            for lane_file in lane_file_open.readlines():  # read them line by line
                lane_info = json.loads(lane_file)
                raw_fname = lane_info['raw_file']  # raw filename, like: clips/0313-1/5320/20.jpg

                _, clips_dir, frame_dir, curr_file = raw_fname.split('/')

                # we have found the right line for the file
                h_samples = lane_info['h_samples']

                all_lanes = []  # list of lanes
                for lane in lane_info['lanes']:
                    one_lane = []  # each lane is a list of tuples (float, float)
                    for h_value, v_value in zip(h_samples, lane):
                        if v_value < 0:  # this indicates no point there
                            continue
                        one_lane.append((v_value, h_value))  # vertical value first

                    all_lanes.append(one_lane)

                lane_indices['/'.join([clips_dir, frame_dir, curr_file])] = all_lanes

        return lane_indices


def example_1():
    culane = TrainLanesCULane()
    k3 = culane.new_image('06031737_0895.MP4/00000.jpg', (590, 1640), 10)
    culane.show_image('06031737_0895.MP4/00000.jpg', (590, 1640), 10)
    culane.superimpose_img('06031737_0895.MP4/00000.jpg')


def example_2():
    tu = TrainLanesTuSimple()
    #k1 = tu.new_image('0313-1/21060/1.jpg', (720, 1280))
    #tu.show_image('0313-1/21060/1.jpg', (720, 1280))
    #tu.superimpose_img('0313-1/21060/1.jpg')

    k1 = tu.new_image('0313-1/480/20.jpg', (720, 1280))
    tu.show_image('0313-1/480/20.jpg', (720, 1280))
    tu.superimpose_img('0313-1/480/20.jpg')


# example_2()
