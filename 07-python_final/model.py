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


    def add(self, frame, movement):
        """
        A method for adding a new frame to the model.
        """
        if self.model == None:
            if self.debug:
                print("Adding first image to model.")

            self.__init__(frame, self.debug)
            # clock-wise direction, beginning in the top-left corner
            self.act_pos = ((0,0), (0,frame.img.shape[1]), frame.img.shape[:2], (frame.img.shape[0],0))

        else:
            if self.debug:
                print("Adding another image to model.")

            # TODO: theese coordinates will have to be warped and corrected over time
            self.act_pos = tuple((item[0]+movement[0], item[1]+movement[1]) for item in self.act_pos)

                # TODO: use the current position as a mask for KP detection
                #       frame comes with detected KP already
                #       match the keypoints and compute homography out of them
                #       warp the image onto the model
                #       warp the coordinates (???)
                #       save the current position on model

            #TODO: implement adding another image to the model
