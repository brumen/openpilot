import cv2
import numpy as np
import tkinter as tk
import threading

from typing import Union

from openpilot.models.lane_detect.hough_lines import HoughLanesImage

from openpilot.models.lane_detect.lane_models.lane_generator_tsf import ( LaneGeneratorCUCrop
                                                                        , LaneGeneratorTUCrop
                                                                        , LaneGeneratorCUTSCrop
                                                                        , LaneGeneratorTUTSCrop
                                                                        , )


class HoughLineMixinTU:
    """ Hough lines extractor for Tusimple images
        720 x 1280 is the pic shape
    """

    ROIS = [(0, 460), (0, 720), (1280, 720), (1280, 460), (840, 260), (400, 260)]  # roi vertices

    def _hough_lines_img(self, img):
        """ Creates the image of hough lines.

        :param img: image from which the params are extracted.
        """
        hough_params = { 'rho': 1
                       , 'theta': np.pi / 180.
                       , 'threshold': 30
                       , 'min_line_len': 20
                       , 'max_line_gap': 20
                       , 'gray_range': (150, 255)
                       , 'canny_range': (100, 200)
                       , }

        cl = HoughLanesImage( img
                            , roi_vertices = self.ROIS
                            , hough_params = hough_params )

        return cv2.cvtColor(cl.show_lines(img.shape[:2], pixel_tol=2).astype(np.uint8) * 255, cv2.COLOR_BGR2RGB)


class HoughLineMixinCU(HoughLineMixinTU):
    """ Hough lines extractor for the CU lane images.
    """

    # cu image is 590 x 1640
    _y_factor = 590/720
    _x_factor = 1640/1280

    ROIS = []
    for x, y in HoughLineMixinTU.ROIS:
        ROIS.append(( int(x * _x_factor), int(y * _y_factor) ))


class LaneGeneratorTUHough(HoughLineMixinTU, LaneGeneratorTUCrop):

    def _process_X(self, orig_image) -> Union[None, np.ndarray]:
        first_img = super()._process_X(orig_image)
        hough_img = self._hough_lines_img(first_img)

        return cv2.addWeighted(first_img, 0.6, hough_img, 0.8, 0)


class LaneGeneratorCUHough(HoughLineMixinCU, LaneGeneratorCUCrop):

    def _process_X(self, orig_image) -> Union[None, np.ndarray]:
        first_img = super()._process_X(orig_image)
        hough_img = self._hough_lines_img(first_img)

        return cv2.addWeighted(first_img, 0.6, hough_img, 0.8, 0)


class SliderLineMixin:

    @staticmethod
    def _slider_image(orig_image, slider_vec) -> Union[None, np.ndarray]:
        slider_mask = cv2.inRange(orig_image, np.uint8(slider_vec[:3]), np.uint8(slider_vec[3:]))

        return cv2.bitwise_and(orig_image, orig_image, mask=slider_mask)


class LaneGeneratorTUHoughSlider(SliderLineMixin, HoughLineMixinTU, LaneGeneratorTUCrop ):

    # good values for y are
    # (0, 175, 0) - (255,255,255)
    # (0, 175, 180) - (255,255,255)  <- THIS IS CHOSEN

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._y_vec = [0, 175, 180, 255, 255, 255]

    def _process_X(self, orig_image) -> Union[None, np.ndarray]:

        first_img = super()._process_X(orig_image)  # cropped image
        slider_line_img = self._slider_image(first_img, self._y_vec)
        hough_img = self._hough_lines_img(slider_line_img)

        return cv2.addWeighted(first_img, 0.6, hough_img, 0.8, 0)
        # return cv2.addWeighted(slider_line_img, 0.6, hough_img, 0.8, 0)


class LaneGeneratorCUHoughSlider(SliderLineMixin, HoughLineMixinCU, LaneGeneratorCUCrop):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._y_vec = [0, 175, 180, 255, 255, 255]

    def _process_X(self, orig_image) -> Union[None, np.ndarray]:

        first_img = super()._process_X(orig_image)  # cropped image
        slider_line_img = self._slider_image(first_img, self._y_vec)
        hough_img = self._hough_lines_img(slider_line_img)

        return cv2.addWeighted(first_img, 0.6, hough_img, 0.8, 0)
        # return cv2.addWeighted(slider_line_img, 0.6, hough_img, 0.8, 0)


class SlidersMixin:

    def __update_yellows(self, y_vec):
        self._y_vec = y_vec

    def _sliders(self):
        """ Constructs the sliders
        """

        canvas = tk.Tk()  # main canvas

        fct_update = lambda cc: self.__update_yellows([y1.get(), y2.get(), y3.get(), y4.get(), y5.get(), y6.get()])

        y1 = tk.Scale(canvas, from_=0, to=255, resolution=5, label='y1', command=fct_update)
        y2 = tk.Scale(canvas, from_=0, to=255, resolution=5, label='y2', command=fct_update)
        y3 = tk.Scale(canvas, from_=0, to=255, resolution=5, label='y3', command=fct_update)
        y4 = tk.Scale(canvas, from_=0, to=255, resolution=5, label='y4', command=fct_update)
        y5 = tk.Scale(canvas, from_=0, to=255, resolution=5, label='y5', command=fct_update)
        y6 = tk.Scale(canvas, from_=0, to=255, resolution=5, label='y6', command=fct_update)

        y1.grid(row=0, column=1)
        y2.grid(row=0, column=2)
        y3.grid(row=0, column=3)
        y1.set(0)
        y2.set(0)
        y3.set(0)

        y4.grid(row=0, column=4)
        y5.grid(row=0, column=5)
        y6.grid(row=0, column=6)
        y4.set(255)
        y5.set(255)
        y6.set(255)

        canvas.mainloop()


class LaneGenerateTUSliders(SlidersMixin, LaneGeneratorTUHoughSlider):
    """ Yellow line but with sliders.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # add the sliders from Tkinter
        sliders_th = threading.Thread(target = lambda : self._sliders())
        sliders_th.start()


class LaneGenerateCUSliders(SlidersMixin, LaneGeneratorCUHoughSlider):
    """ Yellow line but with sliders.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # add the sliders from Tkinter
        sliders_th = threading.Thread(target = lambda : self._sliders())
        sliders_th.start()



# examples
def example_2():
    # new_image_size = (590, 1640, 3)
    scale_size = 1.
    batch_size = 32
    train_percentage = 0.8

    from openpilot.models.lane_detect.lane_config import BASE_TU

    train_generator = LaneGenerateTUSliders( BASE_TU
                                           , to_train = True
                                           , train_percentage  = train_percentage
                                           , batch_size=batch_size
                                           , scale_img = 1.)

    train_generator.show_movie_cont()


def example_1():
    # new_image_size = (590, 1640, 3)
    scale_size = 1.
    batch_size = 32
    train_percentage = 0.8

    from openpilot.models.lane_detect.lane_config import BASE_CU

    train_generator = LaneGenerateCUSliders( BASE_CU
                                           , to_train = True
                                           , train_percentage  = train_percentage
                                           , batch_size=batch_size )

    train_generator.show_movie_cont()


example_1()


# yellow line:
# /home/brumen/data/openpilot/tusimple/clips/0313-1/10020/15.jpg
