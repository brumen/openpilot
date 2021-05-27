import os
import logging
import numpy as np
import json

from typing  import List, Tuple

from openpilot.models.lane_detect.construct_lanes import LanesBase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainLanesCULane(LanesBase):
    """Constructs the training lanes from a file for the CULane set.
    """

    def __init__(self, base_dir : str, folder : str, fname : str):
        """

        :param base_dir: base dir for the subsequent files to obtain
        :param folder: folder in base dir to read from
        :param fname: Complete path filename, e.g. file1.lines.txt
        """

        self.base_dir = base_dir
        self.folder   = folder
        self.fname    = fname  # only 00075.lines.txt

        super().__init__(self._get_lines())

        self._file_to_write = f'{self._lane_nb}.lines.mtx.npy'

        # cached
        self.__rendered_image = None

    def _lane_nb(self) -> str:
        """ Returns the lane number from the file.
        """

        lines_nb, _, _ = self.fname.split('.')

        return lines_nb

    def _complete_fname(self) -> str:
        """ Complete filename from the particular pieces.
        """

        return os.path.join(self.base_dir, self.folder, self.fname)

    def _get_lines(self) -> List[List[Tuple[float, float]]]:
        """ Extract lines from a file.

        :returns: list of lines, each entry for one line, each line consists of a sequence
              of xs and ys.
        """

        with open(self._complete_fname(), 'r') as fopen:
            all_lanes = []
            for file_line in fopen.readlines():
                curr_line = []
                for x in file_line.split(' '):
                    try:
                        curr_line.append(float(x))
                    except Exception as e:
                        logger.debug(f'Ignoring value {x} in {self._complete_fname()}: {str(e)}')

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

    def new_image( self
                   , image_size : Tuple[int, int]
                   , pixel_tol : int = 10 ) -> np.ndarray :
        """ lines_fname has .lines.txt in the name. creates and saves the new image from the old one.
            Saves the resulting image in the folder_dir/lines_fname.lines.mtx

        :param image_size: size of the image
        :param pixel_tol: pixel tolerance, default = 10
        """

        # cached
        if self.__rendered_image is not None:
            return self.__rendered_image

        # precompiled
        precompiled = os.path.join(self.base_dir, self._file_to_write)
        if os.path.isfile(precompiled):
            self.__rendered_image = np.load(precompiled)
            return self.__rendered_image

        # if nothing else, use render right there - completely new work
        self.__rendered_image = np.array(self.show_lines(image_size, pixel_tol=pixel_tol))
        return self.__rendered_image


class TrainLanesTuSimple(TrainLanesCULane):
    """ Get the lanes from the TUSimple dataset.
    """

    def __init__(self, base_dir : str, folder : str, fname : str, lane_file : str):
        """

        :param base_dir: base dir for the subsequent files to obtain
        :param filename: Complete path filename, e.g. /home/brumen/.... /file1.lines.txt
        """

        self._lane_file = os.path.join(base_dir, lane_file)  # the file from where lanes are read.

        super().__init__(base_dir, folder, fname)

        self.__lane_file_entry = {}

    def _lane_file(self, folder : str, fname : str):

        if self.__lane_file_entry is not None:
            return self.__lane_file_entry[(folder, fname)]

        # construct the self.__lane_file_entry
        self.__lane_file_entry = None  # TODO: THIS IS COMPLETELY WRONG

    def _get_lines(self) -> List[List[Tuple[float, float]]]:
        """ Gets the lanes for the jpg file.
        """

        with open(self._lane_file, 'r') as lane_file_open:
            for lane_file in lane_file_open.readlines():  # read them line by line
                lane_info = json.loads(lane_file)
                raw_fname = lane_info['raw_file']  # raw filename, like: clips/0313-1/5320/20.jpg

                _, _, curr_dir, curr_file = raw_fname.split('/')

                if curr_file != self.fname:
                    continue

                # we have found the right line for the file
                h_samples = lane_info['h_samples']

                all_lanes = []  # list of lanes
                for lane in lane_info['lanes']:
                    one_lane = []  # each lane is a list of tuples (float, float)
                    for h_value, v_value in zip(lane, h_samples):
                        if v_value < 0:
                            continue
                        one_lane.append((h_value, v_value))

                    all_lanes.append(one_lane)

                return all_lanes
