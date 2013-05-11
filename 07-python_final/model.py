#!/usr/bin/env python

import cv2
import numpy as np

from frame import Frame
import common
from settings import s


class Model:
    """
    Class representing the model of stitched images.
    """
    def __init__(self, frame=None):
        """
        Initialises an instance of Model class.

        The initial image is passed here as 'frame' parameter or later using
        the 'add()' method.
        """
        if frame != None:
            img = np.zeros((s["model_h"], s["model_w"], 4), dtype=np.uint8)
            self.model = Frame(img, crop=False)

            # Coordinates:
            y1 = int(self.model.img.shape[0]/2-frame.img.shape[0]/2)
            x1 = int(self.model.img.shape[1]/2-frame.img.shape[1]/2)
            y2 = int(self.model.img.shape[0]/2+frame.img.shape[0]/2)
            x2 = x1
            y3 = y2
            x3 = int(self.model.img.shape[1]/2+frame.img.shape[1]/2)
            y4 = y1
            x4 = x3

            self.model.img[y1:y2, x1:x3] = frame.img

            self.current_pos = ((y1,x1),(y2,x2),(y3,x3),(y4,x4))

            self.mask = self.mkMask(self.model.img)
            self.model.detectKeyPoints(cv2.cvtColor(self.mask, cv2.COLOR_BGRA2GRAY))

            self.user_points = []

        else:
            print("No frame supplied. Exiting.")
            exit(1)


    def addUserPoint(self, point):
        """
        Adds a user defined point to a list of tracked points.
        """
        self.user_points.append(point)


    def computeHomography(self, frame):
        """
        Computes Homography Matrix and returns it.
        """
        FLANN_INDEX_KDTREE = 1
        FLANN_INDEX_LSH    = 6
        flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)

        # Compute descriptors:
        self.model.kp, self.model.desc = self.model.extractor.compute(self.model.img, self.model.kp)
        frame.kp, frame.desc = frame.extractor.compute(frame.img, frame.kp)

        # Match them:
        self.matcher = cv2.FlannBasedMatcher(flann_params, {})

        matches = self.matcher.knnMatch(frame.desc, self.model.desc, k=2)
        matches = [m[0] for m in matches if len(m) == 2 and m[0].distance < m[1].distance * 0.75]

        prev_gkp = [frame.kp[m.queryIdx].pt for m in matches]
        gkp = [self.model.kp[m.trainIdx].pt for m in matches]

        prev_gkp, gkp = np.float32((prev_gkp, gkp))

        if len(gkp) < s["min_matches"]:
            return None

        H, status = cv2.findHomography(prev_gkp, gkp, cv2.RANSAC, 3.0)

        # In case the number of matched points is less than four:
        if status.sum() < 4:
            return None

        prev_gkp, gkp = prev_gkp[status], gkp[status]

        return H


    def drawRect(self, img, points):
        """
        Returns an image with Rectangle drawn (defined by points).
        """
        cv2.polylines(img, [np.array(points, np.int32)], True, s["drawing_col"], 3)

        return img


    def drawStr(self, img, (x, y), string):
        """
        Returns an image with given text (s) on given coords drawn.
        """
        cv2.putText(img, string, (x, y), cv2.FONT_HERSHEY_PLAIN, 6.0, s["drawing_col"], thickness=10, lineType=cv2.CV_AA)

        return img


    def drawPoints(self, img, points):
        """
        Returns an image with given points drawn.
        """
        for point in points:
            cv2.circle(img, (int(point[0]), int(point[1])), 10, s["drawing_col"], thickness=3, lineType=cv2.CV_AA)

        return img


    def mkMask(self, img):
        """
        Creates a mask of model image based on thresholding the color value.
        """
        mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
        alpha = np.dsplit(img, 4)[3]

        ret, mask = cv2.threshold(alpha, 254, 1, cv2.THRESH_BINARY)

        ## This is done because of the np.copyto() => mask has to have the same
        ## amount of channels as the image
        return np.dstack((mask, mask, mask, mask))


    def placeNotMapped(self, frame, warped_corners):
        """
        Returns sum of points in mask that are not mapped yet and number of pixels
        in the mask.
        """
        count = 0

        y = (warped_corners[0][0], warped_corners[1][0])
        x = (warped_corners[0][1], warped_corners[2][1])

        count = np.sum(np.dsplit(self.mask, 4)[3][x[0]:x[1], y[0]:y[1]])
        pixels = (y[1]-y[0])*(x[1]-x[0])

        return (pixels - count, pixels)


    def cornerTooFarOut(self, warped_corners, max_offset):
        """
        Returns True if one of points coords is too far outside of map (distance
        is given by max_offset in pixels.
        Otherwise returns False.
        """
        for point in warped_corners:
            if point[0] < -max_offset or point[0] > self.model.img.shape[1] + max_offset:
                return True
            if point[1] < -max_offset or point[1] > self.model.img.shape[0] + max_offset:
                return True

        return False


    def warpCorners(self, frame, H):
        """
        Warps corners of given image according to given homography matrix.

        Returns transformed corners.
        """
        cor1 = (0,0)
        cor2 = (frame.img.shape[1],0)
        cor3 = (frame.img.shape[1],frame.img.shape[0])
        cor4 = (0,frame.img.shape[0])
        corners = np.array([[cor1, cor2, cor3, cor4]], dtype=np.float32)
        warped_corners = cv2.perspectiveTransform(corners, H)

        # Transform points to int and round them
        warped_corners = [(int(round(corner[0])), int(round(corner[1]))) for corner in warped_corners[0]]

        return warped_corners


    def warpUserPoints(self, H, warped_corners):
        """
        Warps user defined points according to given inverse homography matrix.
        """
        points = np.array([self.user_points], dtype=np.float32)

        warped_points = cv2.perspectiveTransform(points, np.matrix(H).I)
        warped_points = [(int(round(point[0])), int(round(point[1]))) for point in warped_points[0]]

        return warped_points


    def add(self, frame, movement):
        """
        A method for adding a new frame to the model.

        Movement is a tuple of Point coords (integer): (y, x)
        """
        # Adding first frame:
        if self.model == None:
            self.__init__(frame, self.debug)

            self.mask = self.mkMask(self.model.img)
            self.model.detectKeyPoints(cv2.cvtColor(self.mask, cv2.COLOR_BGRA2GRAY))

        else:
            # match points => compute homography matrix
            H = self.computeHomography(frame)

            if H == None:
                return

            # warp corner points
            warped_corners = self.warpCorners(frame, H)

            # update current position:
            self.current_pos = warped_corners

            # check if warped corners are too much out of mask's dimensions:
            if self.cornerTooFarOut(warped_corners, s["max_offset"]):
                return

            (not_mapped, size) = self.placeNotMapped(frame, warped_corners)
            thresh = size*(s["min_unmapped_area"]/100)

            if not_mapped <= thresh or size-not_mapped < size*(s["min_mapped_area"]/100):
                return

            new = np.zeros([self.model.img.shape[0], self.model.img.shape[1], 4], np.uint8)
            warped = cv2.warpPerspective(frame.img, H, (self.model.img.shape[1], self.model.img.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

            new_mask = self.mkMask(warped)

            # dst first, then src
            np.copyto(self.model.img, warped, where=np.array(new_mask, dtype=np.bool))

            cv2.imwrite("result.png", self.model.img)

            self.mask = self.mkMask(self.model.img)
            self.model.detectKeyPoints(cv2.cvtColor(self.mask, cv2.COLOR_BGRA2GRAY), s["model_surf_hessian"])
