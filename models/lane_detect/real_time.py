""" Real time lane detection
"""

import cv2
import numpy as np

from openpilot.models.lane_detect.hough_lines import HoughLanesImage

cap = cv2.VideoCapture('/dev/video0')

_y_vec = [0, 175, 180, 255, 255, 255]
ROIS = [(0, 460), (0, 720), (1280, 720), (1280, 460), (840, 260), (400, 260)]  # roi vertices

while True:
    ret, orig_image = cap.read()

    if not ret:
        print("Can't receive frame.")
        continue

    yellow_mask = cv2.inRange(orig_image, np.uint8(_y_vec[:3]), np.uint8(_y_vec[3:]))
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
                        , roi_vertices=ROIS
                        , hough_params=hough_params )

    hough_img = cv2.cvtColor(cl.show_lines(yellow_line_img.shape[:2], pixel_tol=2).astype(np.uint8) * 255, cv2.COLOR_BGR2RGB)

    cv2.imshow('Lanes', cv2.addWeighted(orig_image, 0.6, hough_img, 0.8, 0))

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
