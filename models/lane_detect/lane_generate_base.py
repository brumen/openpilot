# old lane generator class.
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
    def __compare_fnames(fname : str) -> int:
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
            return file_nb

        return None

    def __iter__(self):
        """ Getting all the files from the base dir in ascending order.
        """

        for directory, dir_names, dir_files in os.walk(self.base_dir, followlinks=True):
            selected_files = filter(lambda file_name: self._file_selection(file_name, directory), dir_files)
            for dir_file in sorted(selected_files, key=self.__compare_fnames):
                file_nb = self._file_selection(dir_file, directory)
                if file_nb is not None:
                    yield directory, file_nb

    def _image_name_from_idx(self, idx : Tuple[str, str]) -> str:
        """ Constructs a file name from index.

        :param idx: index, a tuple of folder name, and image name.
        """

        dir_name, image_idx = idx

        return f'{dir_name}/{image_idx}.jpg'

    def _image_from_idx(self, idx: Tuple[str, str]) -> np.ndarray:
        """ Returns the image associated w/ the index idx.

        :param idx: index for which image we want to retrieve.
        """

        return cv2.imread(self._image_name_from_idx(idx))

    def show_movie(self):
        """ Shows the movie from the images in the folder.
        """

        for dir_name_file_nb in self:  # this is the iterator, dir_name_file_nb is a tuple (directory, file_name)
            logger.info(f'Displaying file {dir_name_file_nb}')
            cv2.imshow('Video', self._image_from_idx(dir_name_file_nb) )
            cv2.waitKey(100)  # IMPORTANT LINE, _DO NOT DELETE_

    @staticmethod
    def image_shape(image : np.ndarray) -> Tuple[int, int]:
        """ Returns the shape of the image.
        """

        return image.shape[0], image.shape[1]

    def _image_shape(self, idx) -> Tuple[int, int]:
        return self.image_shape(self._image_from_idx(idx))


def main():
    igb = ImageGenerateBase('/home/brumen/data/openpilot/CULane/driver_37_30frame')
    igb.show_movie()

# main()
