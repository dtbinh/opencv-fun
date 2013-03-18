#!/usr/bin/env python

import cv2
import numpy as np
import common

#TODO: rewrite to use SURF detector and descriptor instead of ORB

class Frame():
    """
    Class representing the Frame class.
    """
    def __init__(self, image, debug=False):
        """
        Initialises an instance of the Frame class.
        """
        self.detector = cv2.FeatureDetector_create("ORB")
        self.extractor = cv2.DescriptorExtractor_create("ORB")

        self.debug = debug
        self.img = image

        self.grayscale = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        self.kp = None
        self.desc = None
        self.displacement = (0.0, 0.0)

        if self.debug:
            print("Frame initialised (debug={}).".format(self.debug))


    def detectKeyPointsORB(self, nFeatures=1000, mask=None):
        """
        A method used for KeyPoints detection and extraction (using the ORB
        detector/extractor).

        Returns the number of KeyPoints detected.
        """
        self.detector.setInt("nFeatures", nFeatures)
        self.detector.setInt("edgeThreshold", 4)
        self.detector.setInt("patchSize", 4)

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


    def _calcAvgDisplacement(self, prev_kp, kp):
        """
        Calculates average displacement of two sets of KeyPoints.
        """
        ds_x = []
        ds_y = []
        for i in range(len(kp)):
            ds_x.append(kp[i].pt[0] - prev_kp[i].pt[0])
            ds_y.append(kp[i].pt[1] - prev_kp[i].pt[1])

        return (sum(ds_x)/len(ds_x), sum(ds_y)/len(ds_y))


    def trackKeyPoints(self, prev_frame):
        """
        A method used for KeyPoints tracking using 'calcOpticalFlowPyrLK'.

        Returns the number of successfully tracked points.
        """
        #TODO: check if the results of displacement are real

        prev_p = common.keyPoint2Point(prev_frame.kp)
        (points, status, err) = cv2.calcOpticalFlowPyrLK(prev_frame.img, self.img, prev_p)

        tmp_kp = common.point2Keypoint(points, prev_frame.kp)

        # filter out untracked points:
        prev_kp = [kp for kp, flag in zip(prev_frame.kp, status) if flag]
        self.kp = [kp for kp, flag in zip(tmp_kp, status) if flag]
        # filter out descriptors of untracked KP:
        (self.kp, self.desc) = self.extractor.compute(self.grayscale, self.kp)


        good_matches = common.filterGoodKeyPoints(prev_frame, self)
        H = common.findHomographyMatrix(prev_frame, self, good_matches)

        self.displacement = self._calcAvgDisplacement(prev_kp, self.kp)

        if self.debug:
            print("Keypoints tracked ({}), displacement: {}.".format(len(self.kp), self.displacement))
            print("Homography matrix: {}".format(H))

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

