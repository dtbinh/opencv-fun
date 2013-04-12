#!/usr/bin/env python

import cv2
import numpy as np

import settings
from frame import Frame


class Model:
    """
    Class representing the model of mapped area.
    """

    def __init__(self, frame=None):
        """
        Initialises the model object.

        Must be initialised with the first frame.
        """
        self.model = None
        self.current_pos = ((0,0),(0,0),(0,0),(0,0))
        self.img = np.zeros(settings.model_size, dtype=np.uint8)
        self.mask = None

        # if the first frame has been supplied
        if frame != None:

            # TODO: compute the coordinates
            y1 = int(self.img.shape[0]/2-frame.img.shape[0]/2)
            x1 = int(self.img.shape[1]/2-frame.img.shape[1]/2)
            y2 = int(self.img.shape[0]/2+frame.img.shape[0]/2)
            x2 = x1
            y3 = y2
            x3 = int(self.img.shape[1]/2+frame.img.shape[1]/2)
            y4 = y1
            x4 = x3

            self.img[y1:y2, x1:x3] = frame.img
            self.mask = self.makeMask()

            model_frame = Frame(self.img)
            model_frame.detectKeyPoints(self.mask)

            self.model = [kp for kp in model_frame.kp]

            self.current_pos = ((y1,x1),(y2,x2),(y3,x3),(y4,x4))


    def add(self, frame):
        """
        Adds KeyPoints from frame supplied to the model.
        """
        # if no KeyPoints have been detected before
        if len(frame.kp) == 0:
            frame.detectKeyPoints()

        # TODO: implement ...
        # ..... Match
