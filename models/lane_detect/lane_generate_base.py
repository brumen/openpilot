import os
import numpy as np
import cv2
import logging

from typing  import Tuple, Union

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageGenerateBase:
    """ Generates the lanes from the base dictionary.
    The structure has to be:
         base_dir/
             subdir/
                 filename.jpg
    """

    def __init__(self, base_dir : str):
        """ Base directory is where all the images, and lane annotations live.

        :param base_dir: base directory where images and annotations live.
        """

        self.base_dir = base_dir

    @staticmethod
    def _compare_fnames(fname : str) -> int:
        """ Compares two filenames.

        :param fname: filename to compute the key on.
        :returns: key of the filename, which is just the integer.
        """

        fname_nb, _ = fname.split('.')

        return int(fname_nb)

    def _file_selection(self, file_name : str, file_directory : str) -> Union[None, str]:
        """ Selects whether the file is included in the walk.

        :param file_name: file name considered for inclusion
        :param file_directory: directory where the file is located.
        :returns: file_nb if the file is considered, otherwise None
        """

        if 'jpg' not in file_name:
            return None

        # .jpg is in the name
        file_nb, _ = file_name.split('.')

        if os.path.exists(os.path.join(file_directory, f'{file_nb}.lines.txt')):
            return file_name

        return None

    def __iter__(self):
        """ Getting all the files from the base dir in ascending order.
        """

        for directory, dir_names, dir_files in os.walk(self.base_dir, followlinks=True):
            selected_files = filter(lambda file_name: self._file_selection(file_name, directory), dir_files)

            for dir_file in sorted(selected_files, key=self._compare_fnames):
                file_name = self._file_selection(dir_file, directory)
                if file_name is not None:
                    yield os.path.join(directory, file_name)

    def show_movie(self, wait_between_frames : int = 100 ):
        """ Shows the movie from the images in the folder.
        """

        for dir_name_file_name in self:  # this is the iterator, dir_name_file_nb is a tuple (directory, file_name)
            logger.info(f'Displaying file {dir_name_file_name}')
            orig_image = cv2.imread(dir_name_file_name)
            cv2.imshow('Video', orig_image)
            cv2.waitKey(wait_between_frames)  # IMPORTANT LINE, _DO NOT DELETE_


def main():
    igb = ImageGenerateBase('/home/brumen/data/openpilot/CULane/driver_37_30frame')
    igb.show_movie()

# main()
