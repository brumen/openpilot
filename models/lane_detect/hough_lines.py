import cv2
import logging
import numpy as np

from typing  import List, Tuple, Optional
from openpilot.models.lane_detect.base_lines import LanesBase


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HoughLines(LanesBase):
    """ Constructs the lanes from hugh points.
    """

    def __init__( self, hough_lines : List[Tuple[float, float, float, float]]):
        """ Constructs the lines from hough lines.

        :param hough_lines: hugh lines detected from the image. they come in the form
          (x1, y1, x2, y2).
        """

        super().__init__(self._lines_from_hough_lines(hough_lines))

        self._hough_lines = hough_lines

    @property
    def hough_lines(self):
        return self._hough_lines

    @staticmethod
    def _lines_from_hough_lines_prev(hough_lines):

        lines_considered_left = {}
        lines_considered_right = {}

        hough_lines_sorted = sorted( hough_lines
                                   , key=lambda x: np.sqrt((x[0][2] - x[0][0])**2 + (x[0][3] - x[0][1])**2)
                                   , reverse=True)  # longest at the beginning

        # TODO: THIS SHOULD BE BETTER WRITTEN
        if hough_lines is not None:
            for hough_line in hough_lines_sorted:
                x1, y1, x2, y2 = hough_line[0]

                # case x1 == x2
                if x1 == x2:
                    if y2 >= y1:  # positive inf slope
                        lines_considered_left[np.inf] = [(x1, y1), (x2, y2)]
                    else:
                        lines_considered_right[np.inf] = [(x1, y1), (x2, y2)]

                    continue

                # x2 != x1
                line_slope = (y2 - y1)/(x2 - x1)

                if np.abs(line_slope) > 0.75:
                    if line_slope <= 0:
                        lines_considered_left[line_slope] = [(x1, y1), (x2, y2)]

                    if line_slope > 0:
                        lines_considered_right[line_slope] = [(x1, y1), (x2, y2)]

        # TODO: THESE TWO THINGS BELOW ARE BAD.
        # average of lines on the left
        if lines_considered_left:
            # take the average
            slope_left_keys = list(lines_considered_left.keys())
            avg_slope = np.average(slope_left_keys)
            lines_considered_left = {avg_slope: lines_considered_left[slope_left_keys[0]]}

        if lines_considered_right:
            # take the average
            slope_right_keys = list(lines_considered_right.keys())
            avg_slope = np.average(slope_right_keys)
            lines_considered_right = {avg_slope: lines_considered_right[slope_right_keys[0]]}

        complete_lines = lines_considered_left
        complete_lines.update(lines_considered_right)

        return [LanesBase.remove_duplicates(line) for line in complete_lines.values()]

    @staticmethod
    def _lines_from_hough_lines(hough_lines):

        lines_considered_left = {}
        lines_considered_right = {}

        hough_lines_sorted = sorted( hough_lines
                                   , key=lambda x: np.sqrt((x[0][2] - x[0][0])**2 + (x[0][3] - x[0][1])**2)
                                   , reverse=True)  # longest at the beginning

        # TODO: THIS SHOULD BE BETTER WRITTEN
        if hough_lines is not None:
            for hough_line in hough_lines_sorted:
                x1, y1, x2, y2 = hough_line[0]

                # case x1 == x2
                if x1 == x2:
                    if y2 >= y1:  # positive inf slope
                        lines_considered_left[np.inf] = [(x1, y1), (x2, y2)]
                    else:
                        lines_considered_right[np.inf] = [(x1, y1), (x2, y2)]

                    continue

                # x2 != x1
                line_slope = (y2 - y1)/(x2 - x1)

                if np.abs(line_slope) > 0.75:
                    if line_slope <= 0:
                        lines_considered_left[line_slope] = [(x1, y1), (x2, y2)]

                    if line_slope > 0:
                        lines_considered_right[line_slope] = [(x1, y1), (x2, y2)]

        if lines_considered_left:
            new_lines_left = []
            for _, line_xys in lines_considered_left.items():
                new_lines_left.extend(line_xys)

        else:
            new_lines_left = []

        if lines_considered_right:
            new_lines_right = []
            for _, line_xys in lines_considered_right.items():
                new_lines_right.extend(line_xys)

        else:
            new_lines_right = []

        complete_lines = [new_lines_left, new_lines_right]

        # TODO: HERE A LOT MORE TO DO.

        return [LanesBase.remove_duplicates(line) for line in complete_lines]

    def draw_hough_lines(self, image_shape : Tuple[int, int, int]):
        """ Draws the hough lines.

        :param image_shape: shape of the image to be drawn.
        """

        image = np.zeros_like(image_shape)

        for hough_line in self.hough_lines:
            x1, y1, x2, y2 = hough_line[0]
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 5)

        return image


class HoughLanesImage(HoughLines):

    def __init__( self
                , image             : np.ndarray
                , roi_vertices      : Optional[List[Tuple[int, int]]] = None
                , hough_params  = None
                , ):
        """ Constructs lanes from the image.

        :param image: image of consideration
        :param roi_vertices: region of interest vertices A, B, C, D, given as A, D, C, B
        :param hough_lines_params: params for the hough_lines transformation
        """

        self.image         = image
        self.roi_vertices  = roi_vertices
        self._hough_params = hough_params

        super().__init__(self._gen_hough_lines())

    @staticmethod
    def region_of_interest(img, vertices):
        """ Generates a mask the size of the image if vertices.

        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        `vertices` should be a numpy array of integer points.
        """

        # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        # filling pixels inside the polygon defined by "vertices" with the fill color
        mask = np.zeros_like(img)  # blank mask
        cv2.fillPoly(mask, vertices, ignore_mask_color)

        return cv2.bitwise_and(img, mask)  # returning the image only where mask pixels are nonzero

    def _gen_hough_lines( self ):
        """ generates the hugh lines from the image, given the parameters above:

        """

        gray_range = self._hough_params['gray_range']
        canny_range = self._hough_params['canny_range']

        gray_img = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)

        # color/intensity
        gray_select = cv2.inRange(gray_img, gray_range[0], gray_range[1])
        gray_select_roi = gray_select if self.roi_vertices is None else self.region_of_interest(gray_select, np.array([self.roi_vertices]))
        img_canny = cv2.Canny(gray_select_roi, canny_range[0], canny_range[1])

        img_gaussian = cv2.GaussianBlur(img_canny, (5, 5), 0)  # 3, 3 is kernel_size

        min_line_len = self._hough_params.get('min_line_len')
        if min_line_len is None:
            y_size, x_size = self.image.shape[0], self.image.shape[1]
            min_line_len_used = np.sqrt(x_size**2 + y_size**2)/20  # 1/20 ofe tenth of the
        else:
            min_line_len_used = min_line_len

        hough_lines_raw = cv2.HoughLinesP( img_gaussian
                                         , self._hough_params['rho']
                                         , self._hough_params['theta']
                                         , self._hough_params['threshold']
                                         , np.array([])
                                         , minLineLength = min_line_len_used
                                         , maxLineGap    = self._hough_params.get('max_line_gap') )

        return hough_lines_raw if hough_lines_raw is not None else []

    def show_lines( self
                  , image_size : Tuple[int, int]
                  , pixel_tol  : int = 10) -> np.ndarray:
        """ Generates the image with constructed hough lines.

        """

        curr_image = super().show_lines(image_size, pixel_tol=pixel_tol)

        return self.region_of_interest(curr_image.astype(np.uint8), np.array([self.roi_vertices]))
