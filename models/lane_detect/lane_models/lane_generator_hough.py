import cv2
import numpy as np

from typing import Union

from openpilot.models.lane_detect.hough_lines import HoughLanesImage

from openpilot.models.lane_detect.lane_models.lane_generator_tsf import ( LaneGeneratorCUShrink
                                                                        , LaneGeneratorTUShrink
                                                                        , LaneGeneratorCUTSShrink
                                                                        , LaneGeneratorTUTSShrink
                                                                        , )


class LaneGeneratorTUHough(LaneGeneratorTUShrink):

    def _process_X(self, orig_image) -> Union[None, np.ndarray]:
        first_img = super()._process_X(orig_image)

        # 590 x 1640  # [[500, 500], [600, 250], [1000, 250], [1280, 500] ]  [[200, 500], [600, 250], [1000, 250], [1400, 500] ]
        rois = [(200, 500), (730, 276), (1000, 250), (1400, 500)]

        hough_params = {'rho': 1
            , 'theta': np.pi / 180.
            , 'threshold': 50
            , 'min_line_len': None
            , 'max_line_gap': 20
            , }
        preprocess_params = {'gray_range': (150, 255)
            , 'canny_range': (50, 100)
            , }

        cl = HoughLanesImage(first_img
                             , roi_vertices=rois
                             , hough_lines_param=hough_params
                             , preprocess_param=preprocess_params
                             , )

        hough_img = cv2.cvtColor(cl.show_lines(first_img.shape[:2]).astype(np.uint8) * 255, cv2.COLOR_BGR2RGB)

        return cv2.addWeighted(first_img, 0.6, hough_img, 0.8, 0)




# examples
def example_2():
    # new_image_size = (590, 1640, 3)
    scale_size = 1.
    batch_size = 32
    train_percentage = 0.8

    from openpilot.models.lane_detect.lane_config import BASE_TU

    train_generator = LaneGeneratorTUHough( BASE_TU
                                     , to_train = True
                                     , train_percentage  = train_percentage
                                     , batch_size=batch_size
                                            , scale_img= 1.)

    train_generator.show_movie()


def example_1():
    # new_image_size = (590, 1640, 3)
    scale_size = 1.
    batch_size = 32
    train_percentage = 0.8

    from openpilot.models.lane_detect.lane_config import BASE_CU

    train_generator = LaneGeneratorTUHough( BASE_CU
                                     , to_train = True
                                     , train_percentage  = train_percentage
                                     , batch_size=batch_size )

    train_generator.show_movie_with_lanes()


example_2()
