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


    def detectKeyPointsORB(self, nFeatures=4000, mask=None):
        """
        A method used for KeyPoints detection and extraction (using the ORB
        detector/extractor).

        Returns the number of KeyPoints detected.
        """

        self.detector = cv2.FeatureDetector_create("ORB")
        self.detector.setInt("nFeatures", nFeatures)
        self.extractor = cv2.DescriptorExtractor_create("ORB")

        self.grayscale = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        if mask == None:
            self.kp = self.detector.detect(self.grayscale)
        else:
            self.kp = self.detector.detect(self.grayscale, mask)

        (self.kp, self.desc) = self.extractor.compute(self.grayscale, self.kp)

        if self.debug:
            print("Keypoints detected ({}) and descriptors extracted.".format(len(self.kp)))

        return len(self.kp)


    def showDetectedKeyPoints(self):
        """
        Displays an image with detected KeyPoints.
        """
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        for i in range(len(self.kp)):
            x,y = self.kp[i].pt
            center = (int(x), int(y))
            cv2.circle(gray, center, 2, (0,128,255), -1)

        cv2.imshow("KeyPoints", gray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


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

