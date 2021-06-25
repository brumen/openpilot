import cv2
import numpy as np
import tkinter as tk
import threading

from typing import Union

from openpilot.models.lane_detect.hough_lines import HoughLanesImage

from openpilot.models.lane_detect.lane_models.lane_generator_tsf import ( LaneGeneratorCUShrink
                                                                        , LaneGeneratorTUShrink
                                                                        , LaneGeneratorCUTSShrink
                                                                        , LaneGeneratorTUTSShrink
                                                                        , )


class LaneGeneratorTUHough(LaneGeneratorTUShrink):
    """ Tusimple images
        720 x 1280 is the pic shape
    """

    ROIS = [(0, 460), (0, 720), (1280, 720), (1280, 460), (840, 260), (400, 260)]  # roi vertices

    def _process_X(self, orig_image) -> Union[None, np.ndarray]:
        first_img = super()._process_X(orig_image)

        hough_params = { 'rho': 1
                       , 'theta': np.pi / 180.
                       , 'threshold': 30
                       , 'min_line_len': 20
                       , 'max_line_gap': 20
                       , 'gray_range': (150, 255)
                       , 'canny_range': (100, 200)
                       , }

        cl = HoughLanesImage(first_img
                            , roi_vertices=self.ROIS
                            , hough_params=hough_params )

        hough_img = cv2.cvtColor(cl.show_lines(first_img.shape[:2]).astype(np.uint8) * 255, cv2.COLOR_BGR2RGB)

        return cv2.addWeighted(first_img, 0.6, hough_img, 0.8, 0)


class YellowLineTU(LaneGeneratorTUHough):

    # good values for y are
    # (0, 175, 0) - (255,255,255)
    # (0, 175, 180) - (255,255,255)  <- THIS IS CHOSEN

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._y_vec = [0, 175, 180, 255, 255, 255]

    def _process_X(self, orig_image) -> Union[None, np.ndarray]:

        yellow_mask = cv2.inRange(orig_image, np.uint8(self._y_vec[:3]), np.uint8(self._y_vec[3:]))
        yellow_line_img = cv2.bitwise_and(orig_image, orig_image, mask=yellow_mask)

        hough_params = { 'rho': 1
                       , 'theta': np.pi / 180.
                       , 'threshold': 30
                       , 'min_line_len': 20
                       , 'max_line_gap': 20
                       , 'gray_range': (150, 255)
                       , 'canny_range': (100, 200)
                       , }

        cl = HoughLanesImage(yellow_line_img
                            , roi_vertices=self.ROIS
                            , hough_params=hough_params )

        hough_img = cv2.cvtColor(cl.show_lines(yellow_line_img.shape[:2], pixel_tol=2).astype(np.uint8) * 255, cv2.COLOR_BGR2RGB)

        # return cv2.addWeighted(orig_image, 0.6, hough_img, 0.8, 0)
        return cv2.addWeighted(yellow_line_img, 0.6, hough_img, 0.8, 0)


class YellowLineTUSliders(YellowLineTU):
    """ Yellow line but with sliders.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # add the sliders from Tkinter
        sliders_th = threading.Thread(target = lambda : self._sliders())
        sliders_th.start()

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


class RoadTU(YellowLineTU):

    def _process_X(self, orig_image) -> Union[None, np.ndarray]:

        image = orig_image
        # crop image
        h, w = image.shape[:2]
        #image = image[200:h - 20, 20:550]
        # create hsv
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        #low_val = (0, 0, 0)
        #high_val = (179, 45, 96)
        low_val = np.uint8(self._y_vec[:3])
        high_val = np.uint8(self._y_vec[3:])
        # Threshold the HSV image
        mask = cv2.inRange(hsv, low_val, high_val)

        # remove noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel=np.ones((8, 8), dtype=np.uint8))
        # close mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel=np.ones((20, 20), dtype=np.uint8))

        # improve mask by drawing the convexhull
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            hull = cv2.convexHull(cnt)
            cv2.drawContours(mask, [hull], 0, (255), -1)
        # erode mask a bit to migitate mask bleed of convexhull
        mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel=np.ones((5, 5), dtype=np.uint8))

        # remove this line, used to show intermediate result of masked road
        road = cv2.bitwise_and(image, image, mask=mask)

        return road
        # apply mask to hsv image
        road_hsv = cv2.bitwise_and(hsv, hsv, mask=mask)
        # set lower and upper color limits
        low_val = (0, 0, 102)
        high_val = (179, 255, 255)
        # Threshold the HSV image
        mask2 = cv2.inRange(road_hsv, low_val, high_val)
        # apply mask to original image
        return cv2.bitwise_and(image, image, mask=mask2)



# examples
def example_2():
    # new_image_size = (590, 1640, 3)
    scale_size = 1.
    batch_size = 32
    train_percentage = 0.8

    from openpilot.models.lane_detect.lane_config import BASE_TU

    train_generator = YellowLineTUSliders( BASE_TU
                                     , to_train = True
                                     , train_percentage  = train_percentage
                                     , batch_size=batch_size
                                            , scale_img= 1.)

    train_generator.show_movie_cont()


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


# yellow line:
# /home/brumen/data/openpilot/tusimple/clips/0313-1/10020/15.jpg