#!/usr/bin/env python

import numpy as np
from frame import Frame
from model import Model
import cv2
import copy

class Compositor:
    """
    Class representing the Compositor object that takes care of taking frames,
    processing them and putting them together in the Model instance.
    """
    model = None

    def __init__(self, video_source=0, rt_result=True, debug=False):
        """
        Initialises an instance of Compositor class.

        Parameter video_source can be number of video device or filename of
        a video file.
        """
        self.rt_result = rt_result
        self.debug = debug
        self.cap = cv2.VideoCapture(video_source);
        # TODO: Make this throw an exception:
        if not self.cap.isOpened():
            print("Could not open: {}".format(video_source))
            exit()

        self.frame = self.grabNextFrame()
        Compositor.model = Model(self.frame, debug=True)
        self.prev_frame = None


    def grabNextFrame(self):
        """
        Function reads next image from video capture device/file and returns
        created Frame object using the image previously read.
        """
        for x in range(2): # TODO: SETTINGS
            ret, img = self.cap.read()

            if not ret:
                return None

        return Frame(img, crop=True, debug=False)


    def addFrameToModel(self, movement=(0.0, 0.0)):
        """
        Function takes a frame instance and its movement coords and calls
        the appropriate Model methods in order to add the frame to it.

        The movement is a tuple of Point coords: (y, x)
        """
        # Make movement coords integers:
        movement = [int(round(item)) for item in movement]

        self.frame.detectKeyPoints()

        if self.debug:
            print("Adding the frame to Model with movement (Y,X): {}".format(movement))

        Compositor.model.add(self.frame, movement)

        if self.debug:
            print("Current position in model: {}".format(Compositor.model.current_pos))


    def run(self):
        """
        The main working function of Compositor class. It reads images from
        video source in a loop and process them so that the final model is
        created and further processed.
        """
        # Process first frame first:
        self.addFrameToModel()

        # Movement sum in format (y, x)
        movement_sum = (0.0, 0.0)

        while True:
            self.prev_frame = self.frame
            self.frame = self.grabNextFrame()

            if self.frame == None:
                # add the last frame to the Model:
                self.frame = self.prev_frame
                self.addFrameToModel(movement_sum)
                break

            tracked = self.frame.trackKeyPoints(self.prev_frame)

            if tracked == None or tracked < 50: # TODO: SETTINGS
                # TODO: here we should check for the movement size!
                # TODO: or rather find out why exactly the tracking went wrong and fix it
                self.frame.detectKeyPoints()
                continue

            # tracking:
            H = Compositor.model.computeHomography(self.frame)

            if H == None:
                self.frame.detectKeyPoints()
                continue

            # transformation of corners according to the homography matrix:
            warped_corners = Compositor.model.warpCorners(self.frame, H)
            # copy the model img for drawing:
            drawed = np.copy(Compositor.model.model.img)

            #if Compositor.model.numOfPointsOutOfModel(warped_corners, 200) < 2:
            if Compositor.model.cornerTooFarOut(warped_corners, 100): # SETTINGS
                Compositor.model.drawStr(drawed, (40,120), "OUT OF MODEL!")
            else:
                Compositor.model.drawRect(drawed, warped_corners)


            # TODO: work on this condition!
            #       like if the combined size of x and y is > xx ... (a function maybe?)
            if abs(movement_sum[0]) > 100 or abs(movement_sum[1]) > 100: # TODO: SETTINGS
                self.addFrameToModel(movement_sum)
                movement_sum = (0.0, 0.0)
                continue

            movement_sum = tuple(sum(item) for item in zip(movement_sum, self.frame.displacement))

            if self.debug:
                cv2.imshow("DEBUG", self.frame.img)
                # TODO: implement text on image when camera out of model
                cv2.imshow("model", cv2.resize(drawed, dsize=(0,0), fx=0.5, fy=0.5))
                if cv2.waitKey(30) >= 0:
                    cv2.destroyWindow("DEBUG")
                    break
