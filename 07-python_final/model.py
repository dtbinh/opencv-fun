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

        if frame != None:
            img = np.zeros((1024, 2048, 4), dtype=np.uint8) # TODO: settings!
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

        else:
            print("No frame supplied. Exiting.")
            exit(1)

        if self.debug:
            print("Model initialised (debug={}).".format(self.debug))


    def __del__(self):
        """
        Removes the instance of Model class.
        """
        if self.debug:
            print("Model deleted.")


    def warpKeyPoints(self, frame, H):
        """
        Transforms KeyPoints coordinates according to given homography matrix.

        Returns list of warped KeyPoints.
        """
        points = common.keyPoint2Point(frame.kp)
        points = np.array([points], np.float32)

        warped_points = cv2.perspectiveTransform(points, H)

        #for i in range(len(points[0])):
        for i in range(50):
            frame.kp[i].pt = tuple(np.array(warped_points[0][i], np.uint8))


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

        self.matcher = cv2.FlannBasedMatcher(flann_params, {})

        matches = self.matcher.knnMatch(frame.desc, self.model.desc, k=2)
        matches = [m[0] for m in matches if len(m) == 2 and m[0].distance < m[1].distance * 0.75]

        prev_gkp = [frame.kp[m.queryIdx].pt for m in matches]
        gkp = [self.model.kp[m.trainIdx].pt for m in matches]

        prev_gkp, gkp = np.float32((prev_gkp, gkp))
        H, status = cv2.findHomography(prev_gkp, gkp, cv2.RANSAC, 3.0)

        #if status.sum() < 4:
            #continue

        prev_gkp, gkp = prev_gkp[status], gkp[status]

        if self.debug:
            print("After homography: {}/{} inliers/matched".format(np.sum(status), len(status)))

        return H


    def drawRect(self, img, points):
        """
        Returns an image with Rectangle drawn (defined by points).
        """
        cv2.polylines(img, [np.array(points, np.int32)], True, (0, 0, 255, 255), 2)

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


    def numOfPointsInMask(self, points):
        """
        Returns the number of points that occur __outside__ of the mask
        """
        count = 0

        for point in points:
            if point[0] < 0 or point[0] >= self.model.img.shape[1]:
                #print("Point coord {} is out of model.".format(point[0]))
                continue
            if point[1] < 0 or point[1] >= self.model.img.shape[0]:
                #print("Point coord {} is out of model.".format(point[1]))
                continue
            # if the point lies outside of the mask, count++:
            if not self.mask[point[1]][point[0]][0]:
                count += 1
                #print("Point {} is OUTSIDE of the mask".format(point))
            #else:
                #print("Point {} is INSIDE of the mask".format(point))

        return count


    def add(self, frame, movement):
        """
        A method for adding a new frame to the model.

        Movement is a tuple of Point coords (integer): (y, x)
        """
        if self.model == None:
            if self.debug:
                print("Adding first image to model.")

            self.__init__(frame, self.debug)

        else:
            if self.debug:
                print("Adding another image to model.")

            self.mask = self.mkMask(self.model.img)
            self.model.detectKeyPoints(cv2.cvtColor(self.mask, cv2.COLOR_BGRA2GRAY))

            # 2) match points => compute homography matrix
            H = self.computeHomography(frame)

            # warp KeyPoints and add them to model KeyPoints
            #print("Model keypoints before: {} + {}".format(len(self.model.kp), len(frame.kp)))
            #self.warpKeyPoints(frame, H)
            #for i in range(len(frame.kp)):
                #self.model.kp.append(frame.kp[i])
            #print("Model keypoints after: {}".format(len(self.model.kp)))

            # 3) warp corner points
            cor1 = (0,0)
            cor2 = (frame.img.shape[1],0)
            cor3 = (frame.img.shape[1],frame.img.shape[0])
            cor4 = (0,frame.img.shape[0])
            corners = np.array([[cor1, cor2, cor3, cor4]], dtype=np.float32)
            warped_corners = cv2.perspectiveTransform(corners, H)

            # Transform points to int and round them
            warped_corners = [(int(round(corner[0])), int(round(corner[1]))) for corner in warped_corners[0]]

            if self.debug:
                print("Warped Corners (X,Y): {}".format(warped_corners))

            # 4) check the points with numOfPointsMask
            num = self.numOfPointsInMask(warped_corners)
            if self.debug:
                print("$$$$ Number of points outside of mask: {} $$$$".format(num))

            if num <= 1:
                return

            # update current position:
            self.current_pos = warped_corners

            new = np.zeros([self.model.img.shape[0], self.model.img.shape[1], 4], np.uint8)
            warped = cv2.warpPerspective(frame.img, H, (self.model.img.shape[1], self.model.img.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)

            new_mask = self.mkMask(warped)

            # dst first, then src
            np.copyto(self.model.img, warped, where=np.array(new_mask, dtype=np.bool))
            cv2.imwrite("/home/milan/result.png", self.model.img)
            #np.copyto(self.model.img, warped, where=np.array(self.mask, dtype=np.bool))
            #self.model.img = warped

            drawed = np.copy(self.model.img)

            if self.debug:
                #cv2.imshow("expanded", self.model.img)
                cv2.imshow("model", self.drawRect(drawed, warped_corners))
                cv2.waitKey(0)

            if self.debug:
                print("Number of detected KP's: {}".format(len(self.model.kp)))
