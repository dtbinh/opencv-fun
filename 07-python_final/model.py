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
            img = np.zeros((1500, 2000, 4), dtype=np.uint8) # TODO: settings!
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

        if self.debug:
            print("Model initialised (debug={}).".format(self.debug))


    def __del__(self):
        """
        Removes the instance of Model class.
        """
        if self.debug:
            print("Model deleted.")


    def addUserPoint(self, point):
        """
        Adds a user defined point to a list of tracked points.
        """
        self.user_points.append(point)
        print("Points: {}".format(self.user_points))
        # TODO: finish implementation


    #def warpKeyPoints(self, frame, H):
        #"""
        #Transforms KeyPoints coordinates according to given homography matrix.

        #Returns list of warped KeyPoints.
        #"""
        #points = common.keyPoint2Point(frame.kp)
        #points = np.array([points], np.float32)

        #warped_points = cv2.perspectiveTransform(points, H)

        ##for i in range(len(points[0])):
        #for i in range(50):
            #frame.kp[i].pt = tuple(np.array(warped_points[0][i], np.uint8))


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

        if len(gkp) < 20: # TODO: SETTINGS!
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
        cv2.polylines(img, [np.array(points, np.int32)], True, (0, 255, 0, 255), 3)

        return img


    def drawStr(self, img, (x, y), s):
        """
        Returns an image with given text (s) on given coords drawn.
        """
        cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 6.0, (0, 255, 0, 255), thickness=10, lineType=cv2.CV_AA)

        return img


    def drawPoints(self, img, points):
        """
        Returns an image with given points drawn.
        """
        for point in points:
            cv2.circle(img, (int(point[0]), int(point[1])), 10, (0, 255, 0, 255), thickness=3, lineType=cv2.CV_AA)

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


    #def makeKPMask(self, img, movement, offset):
        #"""
        #Creates a mask of given image out of current position of camera + given
        #movement + offset (into all directions).
        #"""

        ## current_pos je ve formatu (X, Y) !!! Je v tom bordel, predelat ...

        #y = [min(pt[1] for pt in self.current_pos), max(pt[1] for pt in self.current_pos)]
        #x = [min(pt[0] for pt in self.current_pos), max(pt[0] for pt in self.current_pos)]

        #print("BEFORE CORRECTION: Y: {}, X: {}".format(y, x))

        #if movement[0] < 0:
            #y = (y[0] + movement[0] - offset, y[1] + movement[0] - offset)
        #else:
            #y = (y[0] + movement[0] + offset, y[1] + movement[0] + offset)

        #if movement[1] < 0:
            #x = (x[0] + movement[1] - offset, x[1] + movement[1] - offset)
        #else:
            #x = (x[0] + movement[1] + offset, x[1] + movement[1] + offset)

        #for point in y:
            #if point < 0:
                #point = 0
            #if point > self.model.img.shape[1]:
                #point = self.model.img.shape[1]

        #for point in x:
            #if point < 0:
                #point = 0
            #if point > self.model.img.shape[0]:
                #point = self.model.img.shape[0]

        #print("AFTER CORRECTION: Y: {}, X: {}".format(y, x))

        #mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        #mask_th = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

        #alpha = np.dsplit(img, 4)[3]
        #ret, mask_th = cv2.threshold(alpha, 254, 1, cv2.THRESH_BINARY)

        #print(mask_th.shape)
        #print("Mask_th sum = {}".format(np.sum(mask_th)))

        ##if y[0] > y[1]:
            ##y[0], y[1] = y[1], y[0]
        ##if x[0] > x[1]:
            ##x[0], x[1] = x[1], x[0]

        #mask[y[0]:y[1], x[0]:x[1]] = 1
        ##mask[x[0]:x[1], y[0]:y[1]] = 1

        #print(mask.shape)
        #print("Mask sum = {}".format(np.sum(mask)))

        #final_mask = np.array(np.logical_and(mask, mask_th), dtype=np.uint8)

        #print(final_mask.shape)
        #print("Final_mask sum = {}".format(np.sum(final_mask)))

        #return np.dstack((final_mask, final_mask, final_mask, final_mask))


    #def numOfPointsInMask(self, points):
        #"""
        #Returns the number of points that occur __outside__ of the mask
        #"""
        #count = 0

        #for point in points:
            ## Here, the X, Y coordinates position is out of sync ...
            #if point[0] < 0 or point[0] >= self.model.img.shape[1]:
                #continue
            #if point[1] < 0 or point[1] >= self.model.img.shape[0]:
                #continue
            ## if the point lies outside of the mask, count++:
            #if not self.mask[point[1]][point[0]][0]:
                #count += 1

        #return count


    #def numOfPointsOutOfModel(self, points, max_offset):
        #"""
        #Returns the number of points that occur __out__ of model coordinates
        #by more than given max_offset.
        #"""
        #count = 0

        #for point in points:
            #if point[0] < -max_offset or point[0] > self.model.img.shape[1] + max_offset:
                #count += 1
            #if point[1] < -max_offset or point[1] > self.model.img.shape[0] + max_offset:
                #count += 1

        #return count


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
        if self.model == None:
            if self.debug:
                print("Adding first image to model.")

            self.__init__(frame, self.debug)

            self.mask = self.mkMask(self.model.img)
            self.model.detectKeyPoints(cv2.cvtColor(self.mask, cv2.COLOR_BGRA2GRAY)) # SETTINGS

        else:
            if self.debug:
                print("Adding another image to model.")

            #self.mask = self.makeKPMask(self.model.img, movement, 200)
            #self.model.detectKeyPoints(cv2.cvtColor(self.mask, cv2.COLOR_BGRA2GRAY), 1000) # SETTINGS

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

            if self.debug:
                print("Warped Corners (X,Y): {}".format(warped_corners))

            # update current position:
            self.current_pos = warped_corners

            # TODO: check if warped corners are too much out of mask's dimensions:
            if self.cornerTooFarOut(warped_corners, 20): # SETTINGS!
                print("TOO FAR OUT in Model.add()")
                return

            (not_mapped, size) = self.placeNotMapped(frame, warped_corners)
            thresh = size/12 # SETTINGS!

            if not_mapped <= thresh or size-not_mapped < size/2: # SETTINGS
                return

            new = np.zeros([self.model.img.shape[0], self.model.img.shape[1], 4], np.uint8)
            warped = cv2.warpPerspective(frame.img, H, (self.model.img.shape[1], self.model.img.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

            new_mask = self.mkMask(warped)

            # dst first, then src
            np.copyto(self.model.img, warped, where=np.array(new_mask, dtype=np.bool))
            cv2.imwrite("/home/milan/result.png", self.model.img)

            drawed = np.copy(self.model.img)

            self.mask = self.mkMask(self.model.img)
            self.model.detectKeyPoints(cv2.cvtColor(self.mask, cv2.COLOR_BGRA2GRAY), 1200) # SETTINGS
            #thread.start_new_thread(self.model.detectKeyPoints, (cv2.cvtColor(self.mask, cv2.COLOR_BGRA2GRAY), 600))

            #if self.debug:
                #print("Number of detected KP's: {}".format(len(self.model.kp)))
