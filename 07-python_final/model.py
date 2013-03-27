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

        Movement is a tuple of Point coords (integer): (y, x)
        """
        if self.model == None:
            if self.debug:
                print("Adding first image to model.")

            self.__init__(frame, self.debug)
            # clock-wise direction, beginning in the top-left corner
            self.act_pos = ((0,0), (frame.img.shape[0], 0), frame.img.shape[:2], (0, frame.img.shape[1]))

        else:
            if self.debug:
                print("Adding another image to model.")

            # TODO: use the current position as a mask for KP detection

            #if self.debug:
                #print("Mask of size {}, ones are here {}:{}, {}:{}".format((self.model.img.shape[1], self.model.img.shape[0]), self.act_pos[0][0], self.act_pos[2][0]-1, self.act_pos[0][1], self.act_pos[2][1]-1))

            #######################
            ## THIS DOESN'T WORK! #
            #######################

            #mask = np.zeros((self.model.img.shape[1], self.model.img.shape[0]), np.uint8)
            ## Detect KeyPoints only in current position:
            #print("Line 64")
            #mask[self.act_pos[0][0]:self.act_pos[2][0]-1, self.act_pos[0][1]:self.act_pos[2][1]-1] = 1
            #print("Line 66")
            #self.model.detectKeyPoints(mask)
            #print("Line 68")
            ## TODO: theese coordinates will have to be warped and corrected over time
            self.act_pos = tuple((item[0]+movement[0], item[1]+movement[1]) for item in self.act_pos)


            # TODO: match the keypoints and compute homography out of them

            # TODO: warp the image onto the model

            # TODO: warp the coordinates (???)

            # TODO: save the current position on model

            #TODO: implement adding another image to the model
