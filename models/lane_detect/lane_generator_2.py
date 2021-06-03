import os
import numpy as np
import cv2
import logging

from typing  import Union, Tuple

from openpilot.models.lane_detect.construct_lanes import HoughLanesImage
from openpilot.models.lane_detect.lane_generate_cu import ImageAndLaneMarkingsCULane


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LaneGeneratorBase(ImageAndLaneMarkingsCULane):
    """ Generates images for the lane model.
    """

    def __init__(self
                 , base_dir         : str
                 , batch_size       : int                = 32
                 , train_percentage : float              = .8
                 , to_train         : bool               = True
                 , scale_image      : Union[None, float] = None ):
        """

        :param base_dir: base dir where the files are located.
        :param batch_size: batch size
        :param train_percentage: what percentage of the images is used for training (the remaining is used for
               validating)
        :param to_train: whether to generate the images for training or validating
        :param scale_image: if None, no reshaping, otherwise, a scale parameter
        """

        super().__init__(base_dir)

        self.batch_size       = batch_size
        self.train_percentage = train_percentage
        self.to_train         = to_train
        self.scale_image      = scale_image

        self._iterator = super().__iter__()

    def show_movie(self):
        """ Shows the movie from images.
        """

        for X, y in self:  # X of size (batch_size, image_X, image_y, nb_channels), same for y
            for X_frame, y_frame in zip(X, y):  # unpack along the first axis, which is the batch size
                cv2.imshow('lines2', cv2.addWeighted(X_frame, 0.6, y_frame, 0.8, 0))
                cv2.waitKey(100)  # IMPORTANT LINE, DO NOT DELETE

    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:

        curr_idx = 0
        X_list = []
        y_list = []

        while curr_idx < self.batch_size:
            idx = next(self._iterator)
            X, y = self._generate_one_Xy(idx)
            X_list.append(X)
            y_list.append(y)
            curr_idx += 1

        return np.array(X_list), np.array(y_list)

    def __iter__(self):
        """ Get the next self.batch_size images

        :returns: self, the iterator class.
        """

        return self

    def _process_X(self, orig_image) -> Union[None, np.ndarray]:
        """ Processes the original image, this class doesnt do anything.
        """

        return orig_image

    def _process_y(self, line_image):
        """ Convert bools to colored image.

        """

        return cv2.cvtColor(line_image.astype(np.uint8) * 255, cv2.COLOR_GRAY2RGB)

    def _generate_one_Xy(self, image_id : Tuple[str, str]) -> Tuple[np.ndarray, np.ndarray]:
        """ Generates a tuple of original image (X) and the target image (y) for image_id.

        :param image_id: tuple of directory, and the filename.
        :returns: tuple of original (possible processed image), and the target (line image).
        """

        image_id = tuple(image_id)

        logger.info(f'Loading image {image_id}')

        orig_image = self._image_from_idx(image_id)

        # check if the pre-processed image is saved
        line_image_name = self._precomputed_lane_file(image_id)

        if os.path.isfile(line_image_name):  # preprocessed image is saved
            line_image = np.load(line_image_name)

        else:
            logger.info(f'Generating train lane image {line_image_name}')
            line_image = self.train_lanes(image_id)

        X = self._process_X(orig_image)
        y = self._process_y(line_image)

        return X, y

    #     # TODO: CHECK THIS LINE BELOW
    #     return X, keras.utils.to_categorical(y, num_classes=2)
    #     # return X, y


class LaneGeneratorCU(LaneGeneratorBase):

    def _process_X(self, orig_image) -> Union[None, np.ndarray]:
        """ Processes the original image.
        """

        if orig_image is None:
            return None

        cl = HoughLanesImage( orig_image, [(200, 500), (600, 250), (1000, 250), (1400, 500)])

        # adding color
        lane_image = cv2.cvtColor( cl.preprocess_image(orig_image).astype(np.uint8) * 255
                                 , cv2.COLOR_GRAY2RGB)

        if self.scale_image is None:
            return np.array(lane_image, dtype=np.uint8)

        # resize the image
        nb_rows, nb_cols = self.image_shape(lane_image)
        return cv2.resize( lane_image, (int(nb_cols * self.scale_image), int(nb_rows * self.scale_image) ) )

    def _process_y(self, line_image):
        """ Processes the line image given.
        """

        if line_image is None:
            return None

        line_image = cv2.cvtColor(line_image.astype(np.uint8) * 255, cv2.COLOR_GRAY2RGB)

        if self.scale_image is None:
            return np.array(line_image, dtype=np.uint8)

        # resize the image
        nb_rows, nb_cols = self.image_shape(line_image)
        return cv2.resize(line_image.astype(np.uint8), (int(nb_cols * self.scale_image), int(nb_rows * self.scale_image) ) )


class LaneGeneratorCUHough(LaneGeneratorBase):

    def _process_X(self, orig_image) -> Union[None, np.ndarray]:
        """

        :param orig_image: original image of shape (image_x, image_y, nb_channels)
        :returns: image of the same shape
        """

        if orig_image is None:
            return None

        roi_vertex = [(200, 500), (600, 250), (1000, 250), (1400, 500)]
        cl = HoughLanesImage( orig_image, roi_vertex)

        hough_lines = cl.show_lines(self.image_shape(orig_image)).astype(np.uint8) * 255
        hough_lines = cv2.cvtColor(hough_lines, cv2.COLOR_GRAY2RGB)

        lane_image = cv2.addWeighted(orig_image, 0.6, hough_lines, 0.8, 0)
        # adding color

        if self.scale_image is None:
            return np.array(lane_image, dtype=np.uint8)

        # resize the image
        nb_rows, nb_cols = self.image_shape(lane_image)
        return cv2.resize( lane_image, (int(nb_cols * self.scale_image), int(nb_rows * self.scale_image) ) )

    def show_movie(self):
        """ Shows the movie from images.
        """

        for X, _ in self:  # X of size (batch_size, image_X, image_y, nb_channels), same for y
            for X_frame in X:  # unpack along the first axis, which is the batch size
                cv2.imshow('Hough', X_frame)
                cv2.waitKey(100)  # IMPORTANT LINE, DO NOT REMOVE



def example_2():
    # new_image_size = (590, 1640, 3)
    scale_size = 1.
    batch_size = 32
    train_percentage = 0.8

    from openpilot.models.lane_detect.lane_config import BASE_BASE

    train_generator = LaneGeneratorBase( BASE_BASE
                                     , to_train = True
                                     , train_percentage  = train_percentage
                                     , scale_image=scale_size
                                     , batch_size=batch_size )

    train_generator.show_movie()


def example_1():
    # new_image_size = (590, 1640, 3)
    scale_size = 1.
    batch_size = 32
    train_percentage = 0.8

    from openpilot.models.lane_detect.lane_config import BASE_BASE

    train_generator = LaneGeneratorCU( BASE_BASE
                                     , to_train = True
                                     , train_percentage  = train_percentage
                                     , scale_image=scale_size
                                     , batch_size=batch_size )

#    for x in train_generator:
#        print(x)


    train_generator.show_movie()

def example_3():
    # new_image_size = (590, 1640, 3)
    scale_size = 1.
    batch_size = 32
    train_percentage = 0.8

    from openpilot.models.lane_detect.lane_config import BASE_BASE

    train_generator = LaneGeneratorCUHough( BASE_BASE
                                     , to_train = True
                                     , train_percentage  = train_percentage
                                     , scale_image=scale_size
                                     , batch_size=batch_size )

#    for x in train_generator:
#        print(x)


    train_generator.show_movie()


example_3()
