#!/usr/bin/env python

import cv2
import numpy as np

class Frame():
    """
    Class representing the Frame class.
    """
    def __init__(self, image, debug=False):
        """
        Initialises an instance of the Frame class.
        """
        self.debug = debug
        self.img = image

        self.kp = None
        self.desc = None
        self.displacement = (0.0, 0.0)

        if self.debug:
            print("Frame initialised (debug={}).".format(self.debug))


    def detectKeyPointsORB(self, nFeatures=4000):
        """
        A method used for KeyPoints detection and extraction (using the ORB
        detector/extractor).

        Returns the number of KeyPoints detected.
        """
        self.detector = cv2.FeatureDetector_create("ORB")
        self.detector.setInt("nFeatures", nFeatures)
        self.extractor = cv2.DescriptorExtractor_create("ORB")

        self.grayscale = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        self.kp = self.detector.detect(self.grayscale)
        (self.kp, self.desc) = self.extractor.compute(self.grayscale, self.kp)

        if self.debug:
            print("Keypoints detected ({}) and descriptors extracted.".format(len(self.kp)))

        return len(self.kp)


    def trackKeyPoints(self, prev_img, prev_kp):
        """
        A method used for KeyPoints tracking using KLTracker.

        Returns the number of successfully tracked points.
        """

        if self.debug:
            print("Keypoints tracked ({}), displacement: {}.".format(len(self.kp), self.displacement))

        return len(self.kp)


    def getDisplacement(self):
        """
        Returns the calculated displacement after 'trackKeyPoints()' has been
        successfully called.
        """
        return self.displacement


    def __del__(self):
        """
        Removes the instance of Frame class.
        """
        if self.debug:
            print("Frame deleted.")

