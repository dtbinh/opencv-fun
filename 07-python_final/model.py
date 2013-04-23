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

        if len(gkp) < 50: # TODO: SETTINGS!
            return None

        H, status = cv2.findHomography(prev_gkp, gkp, cv2.RANSAC, 3.0)

        # In case the number of matched points is less than four:
        if status.sum() < 4:
            return None

        prev_gkp, gkp = prev_gkp[status], gkp[status]

        #if self.debug:
            #print("After homography: {}/{} inliers/matched".format(np.sum(status), len(status)))

        return H


    def drawRect(self, img, points):
        """
        Returns an image with Rectangle drawn (defined by points).
        """
        cv2.polylines(img, [np.array(points, np.int32)], True, (0, 128, 255, 255), 3)

        return img


    def drawStr(self, img, (x, y), s):
        """
        Returns an image with given text (s) on given coords drawn.
        """
        cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 6.0, (0, 128, 255, 255), thickness = 10, lineType=cv2.CV_AA)

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
            # Here, the X, Y coordinates position is out of sync ...
            if point[0] < 0 or point[0] >= self.model.img.shape[1]:
                continue
            if point[1] < 0 or point[1] >= self.model.img.shape[0]:
                continue
            # if the point lies outside of the mask, count++:
            if not self.mask[point[1]][point[0]][0]:
                count += 1

        return count


    def numOfPointsOutOfModel(self, points, max_offset):
        """
        Returns the number of points that occur __out__ of model coordinates
        by more than given max_offset.
        """
        count = 0

        for point in points:
            if point[0] < -max_offset or point[0] > self.model.img.shape[1] + max_offset:
                count += 1
            if point[1] < -max_offset or point[1] > self.model.img.shape[0] + max_offset:
                count += 1

        return count


    def placeNotMapped(self, frame, warped_corners):
        """
        Returns sum of points in mask that are not mapped yet and number of pixels
        in the mask.
        """
        count = 0

        y = (warped_corners[0][0], warped_corners[1][0])
        x = (warped_corners[0][1], warped_corners[2][1])

        print("Checking place: Y: {}, X: {}".format(y, x))

        count = np.sum(np.dsplit(self.mask, 4)[3][x[0]:x[1], y[0]:y[1]])
        print("Sum MAPPED: {}".format(count))

        pixels = (y[1]-y[0])*(x[1]-x[0])

        print("Sum NOT MAPPED: {}".format(pixels-count))

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


    def add(self, frame, movement):
        """
        A method for adding a new frame to the model.

        Movement is a tuple of Point coords (integer): (y, x)
        """
        if self.model == None:
            if self.debug:
                print("Adding first image to model.")

            self.__init__(frame, self.debug)

            self.mask = self.mkMask(self.model.img)
            self.model.detectKeyPoints(cv2.cvtColor(self.mask, cv2.COLOR_BGRA2GRAY)) # SETTINGS

        else:
            if self.debug:
                print("Adding another image to model.")

            # match points => compute homography matrix
            H = self.computeHomography(frame)

            if H == None:
                print("Not enough points matched.")
                return

            # warp KeyPoints and add them to model KeyPoints
            #print("Model keypoints before: {} + {}".format(len(self.model.kp), len(frame.kp)))
            #self.warpKeyPoints(frame, H)
            #for i in range(len(frame.kp)):
                #self.model.kp.append(frame.kp[i])
            #print("Model keypoints after: {}".format(len(self.model.kp)))

            # warp corner points
            warped_corners = self.warpCorners(frame, H)

            #if self.debug:
                #print("Warped Corners (X,Y): {}".format(warped_corners))

            # update current position:
            self.current_pos = warped_corners

            # TODO: check if warped corners are too much out of mask's dimensions:
            if self.cornerTooFarOut(warped_corners, 20): # SETTINGS!
                print("TOO FAR OUT in Model.add()")
                return

            (not_mapped, size) = self.placeNotMapped(frame, warped_corners)
            thresh = size/12 # SETTINGS!

            if not_mapped <= thresh or size-not_mapped < size/2:
                return

            new = np.zeros([self.model.img.shape[0], self.model.img.shape[1], 4], np.uint8)
            warped = cv2.warpPerspective(frame.img, H, (self.model.img.shape[1], self.model.img.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

            new_mask = self.mkMask(warped)
            print("SUM OF MASK: {}".format(np.sum(np.dsplit(self.mask, 4)[3])))

            # dst first, then src
            np.copyto(self.model.img, warped, where=np.array(new_mask, dtype=np.bool))
            cv2.imwrite("/home/milan/result.png", self.model.img)

            drawed = np.copy(self.model.img)

            self.mask = self.mkMask(self.model.img)
            #thread.start_new_thread(self.model.detectKeyPoints, (cv2.cvtColor(self.mask, cv2.COLOR_BGRA2GRAY), 600))
            self.model.detectKeyPoints(cv2.cvtColor(self.mask, cv2.COLOR_BGRA2GRAY), 2000) # SETTINGS

            #if self.debug:
                #print("Number of detected KP's: {}".format(len(self.model.kp)))
