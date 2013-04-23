#!/usr/bin/env python

import cv2
import numpy as np
import common
import copy


class Frame():
    """
    Class representing the Frame class.
    """
    def __init__(self, image, crop=True, debug=False):
        """
        Initialises an instance of the Frame class.
        """
        # TODO: experiment with the SURF() value to get better results faster
        self.detector = cv2.SURF(1000) # TODO: SETTINGS
        self.extractor = cv2.DescriptorExtractor_create("SURF")

        self.debug = debug

        # cropping in order to get rid of potentially disrupted borders
        if crop:
            crop_by = (int(image.shape[0]/20), int(image.shape[1]/20)) # TODO: settings
            cropped_img = image[crop_by[0]:image.shape[0]-2*crop_by[0], crop_by[1]:image.shape[1]-2*crop_by[1]]
            self.img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2BGRA)
        else:
            self.img = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

        self.kp = None
        self.desc = None
        self.displacement = (0.0, 0.0)

        if self.debug:
            print("Frame initialised (debug={}).".format(self.debug))


    def detectKeyPoints(self, mask=None, hessian=None):
        """
        A method used for KeyPoints detection and extraction (using the SURF
        detector/extractor).

        Returns the number of KeyPoints detected.
        """
        if hessian != None:
            old_detector = self.detector
            self.detector = cv2.SURF(hessian)

        self.grayscale = cv2.cvtColor(self.img, cv2.COLOR_BGRA2GRAY)
        (self.kp, self.desc) = self.detector.detectAndCompute(self.grayscale, mask)

        if self.debug:
            print("Keypoints detected ({}) and descriptors extracted.".format(len(self.kp)))

        if hessian != None:
            self.detector = old_detector

        return len(self.kp)


    def showDetectedKeyPoints(self):
        """
        Displays an image with detected KeyPoints.
        """
        gray = cv2.cvtColor(self.grayscale, cv2.COLOR_GRAY2BGR)

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

        Returns a tuple representing Point coords: (y, x)
        """
        # TODO: If values differ too much:
        #         * filter out such values
        #         * abort and detect KP

        ds_x = []
        ds_y = []
        for i in range(len(kp)):
            ds_x.append(prev_kp[i].pt[0] - kp[i].pt[0])
            ds_y.append(prev_kp[i].pt[1] - kp[i].pt[1])

        return (sum(ds_y)/len(ds_y), sum(ds_x)/len(ds_x))


    def trackKeyPoints(self, prev_frame):
        """
        A method used for KeyPoints tracking using 'calcOpticalFlowPyrLK'.

        Returns the number of successfully tracked points.
        """
        lk_params = dict( winSize  = (19, 19),
                          maxLevel = 2,
                          criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        prev_p = common.keyPoint2Point(prev_frame.kp)
        (points, status, err) = cv2.calcOpticalFlowPyrLK(prev_frame.img, self.img, prev_p, None, **lk_params)

        tmp_kp = common.point2Keypoint(points, prev_frame.kp)

        # filter out untracked points:
        prev_kp = [kp for kp, flag in zip(prev_frame.kp, status) if flag]
        self.kp = [kp for kp, flag in zip(tmp_kp, status) if flag]

        # filter out descriptors of untracked KP:
        self.kp, self.desc = self.extractor.compute(self.img, self.kp)
        prev_frame.kp, prev_frame.desc = prev_frame.extractor.compute(prev_frame.img, prev_kp)

        # filter KeyPoints using Homography matrix with RANSAC computation:
        (prev_gkp, gkp) = common.filterKPUsingHomography(prev_frame, self)

        if prev_gkp == None or gkp == None:
            return None

        prev_frame.kp = copy.copy(prev_gkp)
        self.kp = copy.copy(gkp)

        # filter out descriptors of unsuitable KP:
        self.kp, self.desc = self.extractor.compute(self.img, self.kp)
        prev_frame.kp, prev_frame.desc = prev_frame.extractor.compute(prev_frame.img, prev_frame.kp)

        # calculate displacement
        self.displacement = self._calcAvgDisplacement(prev_frame.kp, self.kp)

        if self.debug:
            print("Keypoints tracked ({}), displacement: {}.".format(len(self.kp), self.displacement))

        return len(self.kp)


    def __del__(self):
        """
        Removes the instance of Frame class.
        """
        if self.debug:
            print("Frame deleted.")

