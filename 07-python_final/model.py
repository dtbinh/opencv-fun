#!/usr/bin/env python

import cv2
import numpy as np

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
            self.model = frame

        self.act_pos = ((0,0),(0,0),(0,0),(0,0))

        if self.debug:
            print("Model initialised (debug={}).".format(self.debug))


    def __del__(self):
        """
        Removes the instance of Model class.
        """
        if self.debug:
            print("Model deleted.")


    def expand_model(self, movement):
        """
        Creates a new model image expanded by movement coordinates.
        """
        if movement[0] == 0 and movement[1] == 0:
            return

        new_img = np.zeros((self.model.img.shape[0]+abs(movement[0]), self.model.img.shape[1]+abs(movement[1]), 3), np.uint8)

        x_start = x_end = y_start = y_end = 0

        # Find out which directon the image was expanded to:
        # if Y <= 0 (movement UP) => image extended UP
        if movement[0] <= 0:
            y_start = abs(movement[0])
            y_end = self.model.img.shape[0]+abs(movement[0])
        # if Y > 0 (movement DOWN) => image extended DOWN
        else:
            y_start = 0
            y_end = self.model.img.shape[0]

        # if X <= 0 (movement LEFT) => image extended LEFT
        if movement[1] <= 0:
            x_start = abs(movement[1])
            x_end = self.model.img.shape[1]+abs(movement[1])
        # if X > 0 (movement RIGHT) => image extended RIGHT
        else:
            x_start = 0
            x_end = self.model.img.shape[1]

        # Copy model image to extended one:
        new_img[y_start:y_end, x_start:x_end] = self.model.img

        # Assign extended image to image model:
        self.model.img = new_img



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

            #######################
            ## THIS DOESN'T WORK! #
            #######################

            mask = np.zeros(self.model.img.shape[:2], np.uint8)

            if self.debug:
                print("Mask of size {} created.".format(self.model.img.shape[:2]))
                print("-> the ones are here: {}:{}, {}:{}".format(self.act_pos[0][0], self.act_pos[2][0]-1, self.act_pos[0][1], self.act_pos[2][1]-1))

            # Detect KeyPoints only in current position:
            mask[self.act_pos[0][0]:self.act_pos[2][0]-1, self.act_pos[0][1]:self.act_pos[2][1]-1] = 1
            self.model.detectKeyPoints(mask)

            self.expand_model(movement)

            if self.debug:
                cv2.imshow("x", self.model.img)
                cv2.waitKey(0)
                print("New img size = {}".format(self.model.img.shape))

            if self.debug:
                print("Number of detected KP's: {}".format(len(self.model.kp)))

            ## TODO: theese coordinates will have to be warped and corrected over time
            ## TODO: don't change the act_pos just yet ...
            #self.act_pos = tuple((item[0]+movement[0], item[1]+movement[1]) for item in self.act_pos)


            # TODO: match the keypoints and compute homography out of them

            # TODO: warp the image onto the model

            # TODO: warp the coordinates (???)

            # TODO: save the current position on model

            #TODO: implement adding another image to the model
