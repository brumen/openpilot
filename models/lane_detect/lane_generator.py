import os
import numpy as np
import cv2
import logging

from typing  import Union, Tuple

from openpilot.models.lane_detect.train_lanes import TrainLanesCULane, TrainLanesTuSimple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LaneGeneratorCU:
    """ Generates images for the lane model.
    """

    def __init__( self
                , base_dir         : str
                , batch_size       : int    = 32
                , train_percentage : float  = .8
                , to_train         : bool   = True ):
        """

        :param base_dir: base dir where the files are located.
        :param batch_size: batch size
        :param train_percentage: what percentage of the images is used for training (the remaining is used for
               validating)
        :param to_train: whether to generate the images for training or validating
        """

        self.base_dir         = base_dir
        self.train_lanes_obj  = TrainLanesCULane(base_dir)
        self.batch_size       = batch_size
        self.train_percentage = train_percentage
        self.to_train         = to_train

    @staticmethod
    def _compare_fnames(fname : str) -> int:
        """ Comparator comparing 2 file names, to be ordered and displayed correctly.

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

    def _file_iterator(self):
        """ Getting all the files from the base dir in ascending order.
        """

        for directory, dir_names, dir_files in os.walk(self.base_dir, followlinks=True):
            selected_files = filter(lambda file_name: self._file_selection(file_name, directory), dir_files)

            for dir_file in sorted(selected_files, key=self._compare_fnames):
                file_name = self._file_selection(dir_file, directory)
                if file_name is not None:
                    yield os.path.join(directory, file_name)

    def __iter__(self):
        """ Iterator over the frames of the movie.
        """

        curr_idx = 0
        X_list = []
        y_list = []

        for curr_filename in self._file_iterator():
            if curr_idx < self.batch_size:
                X, y = self._generate_one_Xy(curr_filename)
                X_list.append(X)
                y_list.append(y)
                curr_idx += 1
            else:
                yield np.array(X_list), np.array(y_list)
                curr_idx = 0
                X_list = []
                y_list = []

    def show_movie_with_lanes(self, wait_between_frames : int = 100 ):
        """ Shows the movie from images.

        :param wait_between_frames: waits between frames.
        """

        for X, y in self:  # X of size (batch_size, image_X, image_y, nb_channels), same for y
            for X_frame, y_frame in zip(X, y):  # unpack along the first axis, which is the batch size
                cv2.imshow('lines2', cv2.addWeighted(X_frame, 0.6, y_frame, 0.8, 0))
                cv2.waitKey(wait_between_frames)

    def show_movie(self, wait_btw_frames : int =100):
        """ Shows the movie with lanes superimposed.

        :param wait_btw_frames: wait time between frames in milliseconds.
        """

        for X, _ in self:  # X of size (batch_size, image_X, image_y, nb_channels), same for y
            for X_frame in X:  # unpack along the first axis, which is the batch size
                cv2.imshow('original', X_frame)
                cv2.waitKey(wait_btw_frames)

    def _process_X(self, orig_image) -> Union[None, np.ndarray]:
        """ Processes the original image, this class doesnt do anything.
        """

        return orig_image

    def _process_y(self, line_image):
        """ Convert bools to colored image.

        """

        return cv2.cvtColor(line_image.astype(np.uint8) * 255, cv2.COLOR_GRAY2RGB)

    def _generate_one_Xy(self, curr_filename : str) -> Tuple[np.ndarray, np.ndarray]:
        """ Generates a tuple of original image (X) and the target image (y) for image_id.

        :param image_id: tuple of directory, and the filename.
        :returns: tuple of original (possible processed image), and the target (line image).
        """

        logger.info(f'Loading image {curr_filename}.')

        orig_image = cv2.imread(curr_filename)
        line_image = self.train_lanes_obj.new_image(curr_filename, orig_image.shape[:2])

        # X is the original image, y is the marked lane image
        X = self._process_X(orig_image)
        y = self._process_y(line_image)

        return X, y

    #     # TODO: CHECK THIS LINE BELOW
    #     return X, keras.utils.to_categorical(y, num_classes=2)
    #     # return X, y


class LaneGeneratorTU(LaneGeneratorCU):

    def __init__( self
                , base_dir         : str
                , batch_size       : int    = 32
                , train_percentage : float  = .8
                , to_train         : bool   = True ):

        super().__init__(base_dir, batch_size=batch_size, train_percentage=train_percentage, to_train=to_train)

        self.train_lanes_obj = TrainLanesTuSimple(base_dir)

    @staticmethod
    def _compare_fnames(fname : str) -> int:
        """ Comparator: Compares two filenames.

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

        if 'jpg' in file_name:
            return file_name

        # .jpg not in the file name, return None
        return None

    def _generate_one_Xy(self, curr_filename : str) -> Tuple[np.ndarray, np.ndarray]:
        """ Generates a tuple of original image (X) and the target image (y) for image_id.

        :param image_id: tuple of directory, and the filename.
        :returns: tuple of original (possible processed image), and the target (line image).
        """

        logger.info(f'Loading image {curr_filename}.')

        orig_image = cv2.imread(curr_filename)
        line_image = self.train_lanes_obj.new_image( curr_filename.replace(self.base_dir, '')
                                                   , orig_image.shape[:2])

        # X is the original image, y is the marked lane image
        X = self._process_X(orig_image)
        y = self._process_y(line_image)

        return X, y

# examples
def example_2():
    # new_image_size = (590, 1640, 3)
    scale_size = 1.
    batch_size = 32
    train_percentage = 0.8

    from openpilot.models.lane_detect.lane_config import BASE_BASE

    train_generator = LaneGeneratorCU( BASE_BASE
                                       , to_train = True
                                       , train_percentage  = train_percentage
                                       , batch_size=batch_size )

    train_generator.show_movie_with_lanes()


def example_1():
    # new_image_size = (590, 1640, 3)
    scale_size = 1.
    batch_size = 32
    train_percentage = 0.8

    from openpilot.models.lane_detect.lane_config import BASE_BASE

    train_generator = LaneGeneratorCU( BASE_BASE
                                     , to_train = True
                                     , train_percentage  = train_percentage
                                     , batch_size=batch_size )

#    for x in train_generator:
#        print(x)

    train_generator.show_movie()


example_2()
