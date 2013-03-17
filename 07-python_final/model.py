#!/usr/bin/env python

import cv2
import numpy as np

import frame

class Model:

    def __init__(self, debug=False):
        ""
        self.debug = debug

        self.model = None
        self.kp = None
        self.desc = None
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

            self.model = np.copy(frame.img)
            self.kp = np.copy(frame.kp)
            self.desc = np.copy(frame.desc)

        else:
            if self.debug:
                print("Adding another image to model.")
