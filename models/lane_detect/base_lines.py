import logging
import numpy as np

from typing import List, Tuple, Callable, Generator, Union
from scipy  import interpolate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LanesBase:
    """ Base class for lanes.
    """

    def __init__(self, lines : Union[ List[List[Tuple[float, float]]]
                                    , Generator[Tuple[float, float], None, None]]):
        """ Lines given to be drawn on the image.

        :param lines: lines to be drawn (list, where each element corresponds to the line.
             each line is represented as a list of tuples of float,float)
        """

        self._lines = lines

        self._sorted = False  # indicator whether the lines are sorted.

    @staticmethod
    def remove_duplicates(xys : List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """ Removes the duplicates in the y dimension (the spline is done over that dimension.
        Maybe not needed with the inclusion of the smoothness factor.

        :params xys: list of tuples from which the lane is constructed
        :returns: list of tuples with removed ones which have the same second element (y-axis)
        """

        if not xys:
            return xys

        if len(xys) == 1:
            return xys

        # more than 1 element
        xys_pos = 0
        xys_len = len(xys)
        new_xys = [xys[0]]  # first element

        while xys_pos < xys_len - 1:
            prev_elt = xys[xys_pos]
            next_elt = xys[xys_pos + 1]
            if next_elt[1] == prev_elt[1]:  # order by ys
                new_xys.pop()
                new_xys.append(next_elt)
            else:
                new_xys.append(next_elt)
            xys_pos += 1

        return new_xys

    @property
    def lines(self) -> List[List[Tuple[float, float]]]:
        """ Potentially sort this by the first element in the tuple
        """

        if not self._sorted:
            for line in self._lines:
                line.sort(key=lambda x: x[1])  # sort by second elt.
            self._sorted = True

        return [self.remove_duplicates(line) for line in self._lines]

    @staticmethod
    def cubic_spline(points : List[Tuple[float, float]], smooth=1000, k=3):
        """ Returns a cubic spline which interpolates between the points in xy_list.

        :param points: list of points (represented by tuples) representing a line.
        :param smooth: smoothness factor of the lines.
        :returns: a function of y where the line is.
        """

        xy_ordered = sorted(points, key=lambda x: x[1])

        xs = [x[0] for x in xy_ordered]
        ys = [x[1] for x in xy_ordered]
        # create a function of ys - order the points by ys.

        return lambda y : interpolate.splev(y, interpolate.splrep( ys, xs, s=smooth, k=k))

    @staticmethod
    def image_from_spline( spline     : Callable
                         , x_min_max  : Tuple[float, float]
                         , y_min_max  : Tuple[float, float]
                         , image_size : Tuple[int, int]
                         , pixel_tol  : int = 10 ) -> np.ndarray:
        """ Generates the image from one given spline -

        :param spline: function (of y) representing x-s of the points.
        :param x_min_max: minimum and maximum of the x coordinates.
        :param y_min_max: minimum and max. of the y coordinates.
        :param image_size: size of the image (first coord height), second coord width
        :param pixel_tol: tolerance of pixel from the line mark
        """

        x_min, x_max = x_min_max
        y_min, y_max = y_min_max

        line_x_coords = np.array([spline(y_coord)  # if y_min <= y_coord <= y_max else np.inf
                                  for y_coord in range(image_size[0])]).reshape((image_size[0], 1))

        return np.abs(line_x_coords - np.arange(image_size[1])) < pixel_tol

    def show_line( self
                 , marked_line : List[Tuple[float, float]]
                 , image_size  : Tuple[int, int]
                 , pixel_tol   : int = 10 ) -> np.ndarray:
        """ generates the image for the particular marked_line (taken mostly from self.lines)

        :param marked_line: line to be drawn
        :param image_size: image size to be generated.
        :param pixel_tol: pixel tolerance when drawing a line.
        :returns: 2 dimensional array of bools.
        """

        # common usage
        # marked_line = self.lines[line_nb]

        nb_points = len(marked_line)

        if nb_points < 2:
            return np.zeros(image_size, dtype=bool)

        if nb_points == 2:
            x1, y1 = marked_line[0]
            x2, y2 = marked_line[1]
            line_fct = lambda y: (y-y1) * (x2 - x1)/(y2 - y1) + x1

        else:
            line_fct   = self.cubic_spline(marked_line, k=3 if len(marked_line) >= 4 else 2)

        line_ext_x = (min([x[0] for x in marked_line]), max([x[0] for x in marked_line]))
        line_ext_y = (min([x[1] for x in marked_line]), max([x[1] for x in marked_line]))

        return self.image_from_spline(line_fct, line_ext_x, line_ext_y, image_size, pixel_tol=pixel_tol)

    def show_lines( self
                  , image_size : Tuple[int, int]
                  , pixel_tol  : int = 10) -> np.ndarray:
        """ Generates the image of size image_size (without channels).

        :param image_size: size of the image (y_size, x_size), that is the nb. of rows comes first.
        :param pixel_tol: pixel tolerance from the line
        :returns: image constructed from the lines, a 2 dimensional array of bools
        """

        if not self.lines:
            return np.zeros(image_size, dtype=bool)

        curr_line_img = self.show_line(self.lines[0], image_size, pixel_tol=pixel_tol)
        for marked_line in self.lines[1:]:
            curr_line_img |= self.show_line(marked_line, image_size, pixel_tol=pixel_tol)

        return curr_line_img
