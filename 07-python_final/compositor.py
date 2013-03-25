#!/usr/bin/env python

import numpy as np
from frame import Frame
import cv2
import copy


class Compositor:
    """
    Class representing the Compositor object that takes care of taking frames,
    processing them and putting them together in the Model instance.
    """

    def __init__(self, video_source=0, debug=False):
        """
        Initialises an instance of Compositor class.

        Parameter video_source can be number of video device or filename of
        a video file.
        """
        self.debug = debug
        self.cap = cv2.VideoCapture(video_source);
        # TODO: Check if video stream is open (cap.isOpen())

        self.frame = self.grabNextFrame()
        self.prev_frame = None


    def grabNextFrame(self):
        """
        Function reads next image from video capture device/file and returns
        created Frame object using the image previously read.
        """
        # TODO: Rewrite as a iterator class !!!
        ret, img = self.cap.read()

        if not ret:
            return None

        return Frame(img, False)


    def run(self):
        """
        The main working function of Compositor class. It reads images from
        video source in a loop and process them so that the final model is
        created and further processed.
        """
        # Process first frame first:
        self.frame.detectKeyPoints()

        # TODO: fix the loop when grabNextFrame is an iterator
        while True:
            self.prev_frame = copy.deepcopy(self.frame)
            self.frame = self.grabNextFrame()

            if self.frame == None:
                break

            if self.debug:
                cv2.imshow("DEBUG", self.frame.img)
                if cv2.waitKey(30) >= 0:
                    cv2.destroyWindow("DEBUG")
                    break

        # .... (to be finished)
