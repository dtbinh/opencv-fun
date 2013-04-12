#!/usr/bin/env python

import cv2
import numpy as np
import settings


class Frame():
    """
    Class representing the Frame class.
    """
    def __init__(self, image):
        """
        Initialises an instance of the Frame class.
        """
        # TODO: experiment with the SURF() value to get better results faster
        self.detector = cv2.SURF(settings.surf_hessian)
        self.extractor = cv2.DescriptorExtractor_create("SURF")

        # cropping in order to get rid of potentially disrupted borders
        crop_by = (int(image.shape[0]/settings.frame_crop_border), int(image.shape[1]/settings.frame_crop_border))
        cropped_img = image[crop_by[0]:image.shape[0]-2*crop_by[0], crop_by[1]:image.shape[1]-2*crop_by[1]]
        self.img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2BGRA)

        self.kp = None
        self.desc = None
        self.H = None
        self.displacement = (0.0, 0.0)

        if settings.debug_frame:
            print("Frame initialised")


    def detectKeyPoints(self, mask=None):
        """
        A method used for KeyPoints detection and extraction (using the SURF
        detector/extractor).

        Returns the number of KeyPoints detected.
        """
        if mask == None:
            mask = np.ones(self.img.shape[:2], np.uint8)

        self.grayscale = cv2.cvtColor(self.img, cv2.COLOR_BGRA2GRAY)
        (self.kp, self.desc) = self.detector.detectAndCompute(self.grayscale, mask)

        if self.debug:
            print("Keypoints detected ({}) and descriptors extracted.".format(len(self.kp)))


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

        prev_p = common.keyPoint2Point(prev_frame.kp)
        (points, status, err) = cv2.calcOpticalFlowPyrLK(prev_frame.img, self.img, prev_p)

        tmp_kp = common.point2Keypoint(points, prev_frame.kp)

        # filter out untracked points:
        prev_kp = [kp for kp, flag in zip(prev_frame.kp, status) if flag]
        self.kp = [kp for kp, flag in zip(tmp_kp, status) if flag]

        # filter out descriptors of untracked KP:
        self.kp, self.desc = self.extractor.compute(self.img, self.kp)
        prev_frame.kp, prev_frame.desc = prev_frame.extractor.compute(prev_frame.img, prev_kp)

        # find good matches only:
        #good_matches = common.findGoodMatches(prev_frame, self)
        #(prev_gkp, gkp) = common.extractGoodKP(prev_frame, self, good_matches)

        #prev_frame.kp = copy.copy(prev_gkp)
        #self.kp = copy.copy(gkp)

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

