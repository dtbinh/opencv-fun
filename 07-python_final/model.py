#!/usr/bin/env python

import cv2
import numpy as np
import copy

from frame import Frame
import common

class Model:
    """
    Class representing the model of stitched images.
    """
    def __init__(self, frame=None, debug=False):
        """
        Initialises an instance of Model class.

        The initial image is passed here as 'frame' parameter or later using
        the 'add()' method.
        """
        self.debug = debug

        if frame == None:
            self.model = None
        else:
            self.model = copy.deepcopy(frame)

        self.act_pos = ((0,0),(0,0),(0,0),(0,0))

        if self.debug:
            print("Model initialised (debug={}).".format(self.debug))


    def __del__(self):
        """
        Removes the instance of Model class.
        """
        if self.debug:
            print("Model deleted.")


    def add(self, frame):
        """
        A method for adding a new frame to the model.
        """
        if self.debug:
            print("Adding an image to model.")

        if self.model == None:
            if self.debug:
                print("Adding first image to model.")

            self = self.__init__(frame, self.debug)

        else:
            if self.debug:
                print("Adding another image to model.")

            #TODO: implement adding another image to the model
