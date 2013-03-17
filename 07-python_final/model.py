#!/usr/bin/env python

import cv2
import numpy as np

from frame import Frame

class Model:
    ""
    def __init__(self, frame=None, debug=False):
        self.debug = debug

        if frame == None:
            self.model = None
            self.kp = None
            self.desc = None
        else:
            self.model = np.copy(frame.img)
            self.kp = np.copy(frame.kp)
            self.desc = np.copy(frame.desc)

        self.act_pos = ((0,0),(0,0),(0,0),(0,0))

        if self.debug:
            print("Model initialised (debug={}).".format(self.debug))


    def __del__(self):
        ""
        if self.debug:
            print("Model deleted.")


    def addToModel(self, frame):
        ""
        if self.debug:
            print("Adding an image to model.")

        if self.model == None:
            if self.debug:
                print("Adding first image to model.")

            self = self.__init__(frame, self.debug)

        else:
            if self.debug:
                print("Adding another image to model.")
