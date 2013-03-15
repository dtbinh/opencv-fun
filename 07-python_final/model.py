#!/usr/bin/env python

import cv2
import numpy as np

class Model:

    def __init__(self, debug = False):
        self.debug = debug

        self.detector = None # TODO: specify a FP detector

        # Model (stitched image):
        self.model = None
        # Keypoints of model:
        self.keypoints = None
        # Keypoint's prices (to be implemented later):
        self.keypoint_prices = None

        if self.debug:
            print("Model initialised (debug={}).".format(self.debug))


    def __del__(self):
        if self.debug:
            print("Model deleted.")


    def findFP(self, image, mask):
        self.keypoints = surf.detect(image, mask)
        if self.debug:
            print("Keypoints detected.")


    def addToModel(self, image):
        if self.debug:
            print("Adding an image to model.")
