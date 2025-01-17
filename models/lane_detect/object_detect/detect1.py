# object detection

import cv2
import torch
import numpy as np

from typing import Union

from openpilot.models.lane_detect.lane_generator import LaneGeneratorCU, LaneGeneratorTU


class ObjectDetectMixin:
    """ Image shrink class.
    """

    def _score_frame(self, frame):
        """ Scores the model

        """

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(device)
        # frame = torch.tensor(frame)
        results = self.model(frame)

        labels = results.xyxyn[0][:, -1].cpu().numpy().astype(np.uint)
        cord = results.xyxyn[0][:, :-1].cpu().numpy()

        return labels, cord

    def _plot_boxes(self, results, frame):
        """ Plots boxes around the detected stuff in the picture.

        :param results:
        :param frame:
        :returns: a frame with the boxes drawn around detected objects.
        """

        labels, cord = results
        nb_labels = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        bgr = (0, 255, 0)  # color of the box
        classes = self.model.names  # Get the name of label index

        for i in range(nb_labels):
            x1, y1, x2, y2, prob = cord[i]

            if prob < 0.2:  # If score is less than 0.2 we avoid making a prediction.
                continue

            x1 = int(x1 * x_shape)
            y1 = int(y1 * y_shape)
            x2 = int(x2 * x_shape)
            y2 = int(y2 * y_shape)

            frame = cv2.rectangle( frame, (x1, y1), (x2, y2), bgr, 2)
            frame = cv2.putText(frame, classes[labels[i]], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

        return frame

    def _process_X(self, orig_image) -> Union[None, np.ndarray]:

        first_tsf = super()._process_X(orig_image)

        results = self._score_frame(first_tsf)

        return self._plot_boxes(results, first_tsf)


class ObjectDetectCU(ObjectDetectMixin, LaneGeneratorCU):

    def __init__( self
                , base_dir         : str
                , batch_size       : int   = 32
                , train_percentage : float = .8
                , to_train         : bool  = True
                , detect_params    = None ):

        super().__init__(base_dir, batch_size, train_percentage, to_train)

        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.detect_params = detect_params


class ObjectDetectTU(ObjectDetectMixin, LaneGeneratorTU):

    def __init__( self
                , base_dir         : str
                , batch_size       : int   = 32
                , train_percentage : float = .8
                , to_train         : bool  = True
                , detect_params    = None ):

        super().__init__(base_dir, batch_size, train_percentage, to_train)

        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.detect_params = detect_params


# examples
def example_2():
    # new_image_size = (590, 1640, 3)
    scale_size = 1.
    batch_size = 32
    train_percentage = 0.8

    from openpilot.models.lane_detect.lane_config import BASE_TU

    train_generator = ObjectDetectTU( BASE_TU
                                     , to_train = True
                                     , train_percentage  = train_percentage
                                     , batch_size=batch_size
                                     , )

    train_generator.show_movie()


def example_1():
    # new_image_size = (590, 1640, 3)
    scale_size = 1.
    batch_size = 32
    train_percentage = 0.8

    from openpilot.models.lane_detect.lane_config import BASE_CU

    train_generator = ObjectDetectCU( BASE_CU
                                     , to_train = True
                                     , train_percentage  = train_percentage
                                     , batch_size=batch_size )

    train_generator.show_movie()


example_2()

# Testing lanenet
# python tools/test_lanenet.py --weights_path /home/brumen/work/lanenet-net-detection/lanenet_model/pretrained_model/ --image_path /home/brumen/data/openpilot/tusimple/clips/0313-1/4080/10.jpg
