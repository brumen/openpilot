import cv2
import logging
import numpy as np

from typing  import List, Tuple, Optional, Callable, Any, Dict, Generator, Union
from scipy   import interpolate

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
        :param image_size: size of the image (first coord height), second coord width
        :param pixel_tol: tolerance of pixel from the line mark
        """

        # x_min, x_max = x_min_max
        # y_min, y_max = y_min_max

        line_x_coords = np.array([spline(y_coord)
                                  for y_coord in range(image_size[0])]).reshape((image_size[0], 1))

        return np.abs(line_x_coords - np.arange(image_size[1])) < pixel_tol

    def show_line( self
                 , marked_line : List[Tuple[float, float]]
                 , image_size  : Tuple[int, int]
                 , pixel_tol   : int = 10 ):
        """ generates the image for the particular marked_line (taken mostly from self.lines)

        :param marked_line: line to be drawn
        :param image_size: image size to be generated.
        :param pixel_tol: pixel tolerance when drawing a line.
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
        :returns: image constructed from the lines.
        """

        if not self.lines:
            return np.zeros(image_size, dtype=bool)

        curr_line_img = self.show_line(self.lines[0], image_size, pixel_tol=pixel_tol)
        for marked_line in self.lines[1:]:
            curr_line_img |= self.show_line(marked_line, image_size, pixel_tol=pixel_tol)

        return curr_line_img


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
    def __slope_among_lines(new_slope, existing_slopes, slope_tol):
        """ Is the new line slope among the line slopes of existing lines.

        """

        slope_matches = []
        dual_slope = new_slope - np.pi if 0. <= new_slope <= np.pi else new_slope + np.pi

        for existing_slope in existing_slopes:
            slope_matches.append(np.abs(new_slope - existing_slope) < slope_tol)
            slope_matches.append(np.abs(dual_slope - existing_slope) < slope_tol)

        return any(slope_matches)

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

                if np.abs(line_slope) > 0.5:
                    if line_slope <= 0:
                        lines_considered_left[line_slope] = [(x1, y1), (x2, y2)]

                    if line_slope > 0:
                        lines_considered_right[line_slope] = [(x1, y1), (x2, y2)]

        complete_lines = lines_considered_left
        complete_lines.update(lines_considered_right)

        return [LanesBase.remove_duplicates(line) for line in complete_lines.values()]

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
                , hough_lines_param : Dict[str, Any] = None
                , preprocess_param  : Dict[str, Any] = None
                , ):
        """ Constructs lanes from the image.

        :param image: image of consideration
        :param roi_vertices: region of interest vertices A, B, C, D, given as A, D, C, B
        :param hough_lines_param: params for the hough_lines transformation
        :param preprocess_param: params for the preprocess image
        """

        self.image              = image
        self.roi_vertices       = roi_vertices

        super().__init__(self.generate_hough_lines( preprocess_param, hough_lines_param))

    def preprocess_image( self
                        , preprocess_params):

        gray_range = preprocess_params['gray_range']
        canny_range = preprocess_params['canny_range']

        gray_img = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)

        # color/intensity
        gray_select = cv2.inRange(gray_img, gray_range[0], gray_range[1])
        gray_select_roi = gray_select if self.roi_vertices is None else self.__region_of_interest(gray_select, np.array([self.roi_vertices]))
        img_canny = cv2.Canny(gray_select_roi, canny_range[0], canny_range[1])

        return cv2.GaussianBlur(img_canny, (3, 3), 0)  # 3, 3 is kernel_size

    def generate_hough_lines( self
                            , preprocess_params
                            , hough_params ):
        """ generates the hugh lines from the image, given the parameters above:

        """

        img_gaussian = self.preprocess_image(preprocess_params)

        min_line_len = hough_params.get('min_line_len')
        if min_line_len is None:
            y_size, x_size = self.image.shape[0], self.image.shape[1]
            min_line_len_used = np.sqrt(x_size**2 + y_size**2)/20  # 1/20 ofe tenth of the
        else:
            min_line_len_used = min_line_len

        hough_lines_raw = cv2.HoughLinesP( img_gaussian
                                         , hough_params['rho']
                                         , hough_params['theta']
                                         , hough_params['threshold']
                                         , np.array([])
                                         , minLineLength = min_line_len_used
                                         , maxLineGap    = hough_params.get('max_line_gap') )

        return hough_lines_raw if hough_lines_raw is not None else []

    @staticmethod
    def __region_of_interest(img, vertices):
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

    def show_lines( self
                      , image_size : Tuple[int, int]
                      , pixel_tol  : int = 10) -> np.ndarray:
        """ Generates the image with constructed hough lines.

        """

        curr_image = super().show_lines(image_size, pixel_tol=pixel_tol)

        return self.__region_of_interest(curr_image.astype(np.uint8), np.array([self.roi_vertices]))


class HoughLanesImage2(HoughLanesImage):

    def preprocess_image(self, image):
        gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # color/intensity
        gray_select = cv2.inRange(gray_img, 150, 255)
        gray_select_roi = gray_select if self.roi_vertices is None else self.__region_of_interest(gray_select, np.array([self.roi_vertices]))
        img_canny = cv2.Canny(gray_select_roi, 50, 100)  # 50 = low_threshold 100 =  high_threshold)
        return cv2.GaussianBlur(img_canny, (3, 3), 0)  # 3, 3 is kernel_size

    def _identify_white_line(self, image):
        """ Emphasize white line on the image, and identify it.
        """
        pass

    def show_lines( self
                      , image_size : Tuple[int, int]
                      , pixel_tol  : int = 10) -> np.ndarray:
        """ Generates the image with constructed hough lines.

        """

        curr_image = super().show_lines(image_size, pixel_tol=pixel_tol)

        return self.__region_of_interest(curr_image.astype(np.uint8), np.array([self.roi_vertices]))


def calibrate_pic(params, orig_mtx : np.ndarray, sol_mtx : np.ndarray, roi_vertices):
    hough_params = {'rho': params[0]
                    , 'theta': params[1]
                    , 'threshold': int(params[2])
                    , 'min_line_len': params[3]
                    , 'max_line_gap': 20}
    preprocess_params = {'gray_range': (int(params[4]), int(params[5]))
                         , 'canny_range': (int(params[5]), int(params[6]))}

    cl = HoughLanesImage( orig_mtx
                        , roi_vertices=roi_vertices
                        , hough_lines_param= hough_params
                        , preprocess_param= preprocess_params )

    res = cl.show_lines(sol_mtx.shape).astype(np.uint8)

    diff_mtx = res - sol_mtx

    return np.sum((diff_mtx * (diff_mtx > 0))**2)
    #return np.sum(diff_mtx ** 2)


from matplotlib import pyplot as plt

image_nb = '00150'
dir_nb = '05151640_0419'
fname2 = f'/home/brumen/data/openpilot/CULane/driver_23_30frame/{dir_nb}.MP4/{image_nb}.jpg'

#im = cv2.imread(fname2)

#hl = HoughLanesImage(im, roi_vertices=[(200, 500), (730, 276), (1000, 250), (1400, 500) ])
#hl.show_lines(im.shape[:2])

# 540 x 960   # [[100, 540], [450, 320], [515, 320], [900, 540]]
# 590 x 1640  # [[500, 500], [600, 250], [1000, 250], [1280, 500] ]  [[200, 500], [600, 250], [1000, 250], [1400, 500] ]
def fname1(image_nb = '00150', dir_nb = '05151640_0419', rois = [(200, 500), (730, 276), (1000, 250), (1400, 500) ], fname3 = None):
    fname2 = f'/home/brumen/data/openpilot/CULane/driver_23_30frame/{dir_nb}.MP4/{image_nb}.jpg' if fname3 is None else fname3
    solution = f'/home/brumen/data/openpilot/CULane/driver_23_30frame/{dir_nb}.MP4/{image_nb}.lines.mtx.npy'

    im = cv2.imread(fname2)

    hough_params = { 'rho': 1
                   , 'theta': np.pi/180.
                   , 'threshold': 50
                   , 'min_line_len': None
                   , 'max_line_gap': 20
                   , }
    preprocess_params = { 'gray_range': (150, 255)
                        , 'canny_range': (50, 100)
                        , }

    cl = HoughLanesImage(im
                        , roi_vertices=rois
                        , hough_lines_param=hough_params
                        , preprocess_param= preprocess_params
                        , )

    #preproc = cl.preprocess_image(im)

    im2 = np.array(cl.show_lines(im.shape[:2])).astype(np.uint8)
    im2 = cv2.cvtColor(im2 * 255, cv2.COLOR_GRAY2RGB)
    combined = cv2.addWeighted(im, 0.6, im2, 0.8, 0)

    cv2.imshow('orig', im)
    cv2.imshow('combine', combined)
    #cv2.imshow('preproc', preproc)
    cv2.waitKey(100)

    #fig = plt.figure()
    #fig.add_subplot(3, 1, 1)
    #plt.imshow(im)
    #fig.add_subplot(3, 1, 2)
    #plt.imshow(im2)
    #fig.add_subplot(3, 1, 3)
    #plt.imshow(combined)
    #plt.show()

    return im, cl, im2, preproc


def fname5(image_nb = '00150', dir_nb = '05151640_0419', rois = [(200, 500), (730, 276), (1000, 250), (1400, 500) ], fname3 = None):
    fname2 = f'/home/brumen/data/openpilot/CULane/driver_23_30frame/{dir_nb}.MP4/{image_nb}.jpg' if fname3 is None else fname3
    solution = f'/home/brumen/data/openpilot/CULane/driver_23_30frame/{dir_nb}.MP4/{image_nb}.lines.mtx.npy'

    im = cv2.imread(fname2)
    sol = np.load(solution)

    bounds = [(0.5, 1.5), (0.1, np.pi - 0.1), (20, 70), (20, 200), (50, 200), (200, 255), (20, 80), (80, 150)]
    curr_idx = 0
    curr_min = 10**100

    lin_interp = lambda idx, nb_steps : bounds[idx][0] + nb_steps * (bounds[idx][1] - bounds[idx][0])/10
    for iter1 in range(10000):
        rho_idx, theta_idx, threshold_idx, line_len_idx, gray_range_d, gray_range_u, canny_range_d, canny_range_u = np.random.random_integers(0, 10, 8)

        rho = lin_interp(0, rho_idx)
        theta = lin_interp(1, theta_idx)
        threshold = lin_interp(2, threshold_idx)
        line_len = lin_interp(3, line_len_idx)
        gray_d = lin_interp(4, gray_range_d)
        gray_u = lin_interp(5, gray_range_u)
        canny_d = lin_interp(6, canny_range_d)
        canny_u = lin_interp(7, canny_range_u)
        res = calibrate_pic([rho, theta, threshold, line_len, gray_d, gray_u, canny_d, canny_u], im, sol, roi_vertices=rois)
        if res < curr_min:
            curr_min = res
            curr_sol = (rho, theta, threshold, line_len, gray_d, gray_u, canny_d, canny_u)
            print(curr_min, curr_sol)
        curr_idx += 1
        # print(curr_idx, res, curr_min,  rho, theta, threshold, line_len, gray_d, gray_u, canny_d, canny_u)

    return res, curr_sol

    #from scipy.optimize import minimize
    # res = minimize( lambda p: calibrate_pic(p, im, sol, roi_vertices=rois)
    #               , np.array([1., np.pi/180., 50, 50, 150, 255, 50, 100])
    #               , bounds=bounds
    #               , )

k1 = fname5()

# fname1(rois = [[100, 540], [515, 320], [450, 320], [900, 540]], fname3 = '/home/brumen/tmp/Simple-Lane-Detection/test_images/solidWhiteCurve.jpg')
# im, cl, im2 = fname1(rois = [[100, 540], [450, 320], [515, 320], [900, 540]  ], fname3 = '/home/brumen/tmp/Simple-Lane-Detection/test_images/solidWhiteCurve.jpg')

# THE FIRST LINE IS THE TEST LINE
# im, cl, im2 = fname1(rois = [[200, 500], [600, 250], [1000, 250], [1400, 500] ])
# # im, cl, im2 = fname1(rois = [[100, 540], [450, 320], [515, 320], [900, 540]  ])
# plt.imshow(im2)

# fname = '/home/brumen/data/openpilot/driver_23_30frame/05151640_0419.MP4/00150.jpg'
# im = cv2.imread(fname)
# # rois = np.array([[[100, 540], [900, 540], [515, 320], [450, 320]]])
# #rois = np.array([[[617, 448], [1080, 448], [726, 266], [885, 266]]])
# rois = [[500, 500], [600, 250], [1000, 250], [1280, 500] ]
#
# cl = ConstructLanesFromImageHugh(im, rois)
# im2 = np.array(cl.generate_image((540, 960))).astype(np.uint8)
# from matplotlib import pyplot as plt
# plt.imshow(im)
# plt.show()


# demonstration
#folder_dir = '/home/brumen/work/openpilot/models/lane_detect/data/driver_23_30frame/05161020_0504.MP4'
#fname   = '/home/brumen/work/openpilot/models/lane_detect/data/driver_23_30frame/05161020_0504.MP4/00075.lines.txt'
#fname2  = '00075.lines.txt'
#eof_jpg = '/home/brumen/work/openpilot/models/lane_detect/data/driver_23_30frame/05161020_0504.MP4/00075.jpg'

#image = '/home/brumen/work/openpilot/models/lane_detect/data/driver_161_90frame/06032338_0992.MP4/05310.lines.txt'
#cl = ConstructLanes(image)
#cl.show_image((1640, 590))  # this is IMPORTANT HERE

# cl = ConstructLanes(fname)
# im = cl.new_image((590, 1640))
