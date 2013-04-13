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
            img = np.zeros((800, 2048, 4), dtype=np.uint8) # TODO: settings!
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
        # TODO: this will have to be changed as it counts with the whole model image

        FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
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

        # TODO: check if the point is outside of the image

        for point in points:
            # if the point lies outside of the mask, count++:
            if not self.mask[point[0]][point[1]]:
                count += 1
                if self.debug:
                    print("Point {} is OUTSIDE of the mask".format(point))

        return count


    #def correctCoords(self, movement, int_warped_corners):
        #"""
        #Corrects coordinates of points after expandModel() has been called.
        #int_warped_corners are a list of rounded integer coords of warped corners
        #of added image

        #Returns corrected warped_corners and assignes self.act_pos
        #"""
        #if movement[0] == 0 and movement[1] == 0:
            #return int_warped_corners

        ## if Y < 0 => correct coords:
        #if movement[0] < 0:
            ## fix act_pos:
            #self.act_pos = [(coord[0]+abs(movement[0]), coord[1]) for coord in self.act_pos]
            ## fix int_warped_corners:
            #int_warped_corners = [(corner[0]+abs(movement[0]), corner[1]) for corner in int_warped_corners]

        ## if X < 0 => correct coords:
        #if movement[1] < 0:
            ## fix act_pos:
            #self.act_pos = [(coord[0], coord[1]+abs(movement[1])) for coord in self.act_pos]
            ## fix int_warped_corners:
            #int_warped_corners = [(corner[0], corner[1]+abs(movement[1])) for corner in int_warped_corners]

        #return int_warped_corners


    ## TODO: remove this when done, this function is no longer used
    #def expandModel(self, movement):
        #"""
        #Creates a new model image expanded by movement coordinates.
        #"""
        #if movement[0] == 0 and movement[1] == 0:
            #return

        #new_img = np.zeros((self.model.img.shape[0]+abs(movement[0]), self.model.img.shape[1]+abs(movement[1]), 4), np.uint8)

        #x_start = x_end = y_start = y_end = 0

        ## Find out which directon the image was expanded to:
        ## if Y <= 0 (movement UP) => image extended UP
        #if movement[0] <= 0:
            #y_start = abs(movement[0])
            #y_end = self.model.img.shape[0]+abs(movement[0])
        ## if Y > 0 (movement DOWN) => image extended DOWN
        #else:
            #y_start = 0
            #y_end = self.model.img.shape[0]

        ## if X <= 0 (movement LEFT) => image extended LEFT
        #if movement[1] <= 0:
            #x_start = abs(movement[1])
            #x_end = self.model.img.shape[1]+abs(movement[1])
        ## if X > 0 (movement RIGHT) => image extended RIGHT
        #else:
            #x_start = 0
            #x_end = self.model.img.shape[1]

        ## Copy model image to extended one:
        #new_img[y_start:y_end, x_start:x_end] = self.model.img

        ## Assign extended image to image model:
        #self.model.img = new_img



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

            # TODO for TODO:
            # check what is needed ...

            # TODO: use the current position as a mask for KP detection

            ## Detect KeyPoints only in current position:
            #mask[self.act_pos[0][0]:self.act_pos[2][0]-1, self.act_pos[0][1]:self.act_pos[2][1]-1] = 1
            #self.model.detectKeyPoints(mask)

            # TODO: HOW TO DETERMINE WHETHER OR NOT IS THE IMAGE TO BE ADDED
            # 1) create a mask of current model img
            # 2) determine if corner points of added image (warped?) are inside the mask or not
            # 3) if they are -> the image is already in the model
            #    else -> add the image


            # TODO: HOW TO ADD THE IMAGE:
            # 1) detect KP on the whole model img (using mask) => later only using current position
            # TODO: detect KP only on a current area, not the whole model
            self.mask = self.mkMask(self.model.img)
            #self.model.detectKeyPoints(mask)

            self.model.detectKeyPoints(cv2.cvtColor(self.mask, cv2.COLOR_BGRA2GRAY))

            # 2) match points => compute homography matrix
            H = self.computeHomography(frame)

            #if self.debug:
                #print("Homography matrix: {}".format(H))

            # warp KeyPoints and add them to model KeyPoints
            #print("Model keypoints before: {} + {}".format(len(self.model.kp), len(frame.kp)))
            #self.warpKeyPoints(frame, H)
            #for i in range(len(frame.kp)):
                #self.model.kp.append(frame.kp[i])
            #print("Model keypoints after: {}".format(len(self.model.kp)))

            # 3) warp corner points
            #corners
            corners = np.array([[[0,0], [frame.img.shape[0], 0], [frame.img.shape[0], frame.img.shape[1]], [0, frame.img.shape[1]]]], dtype=np.float32)
            warped_corners = cv2.perspectiveTransform(corners, H)

            if self.debug:
                #print("Corners before: {}".format(corners))
                print("Warped Corners: {}".format(warped_corners))

            ## Transform points to int and round them
            #int_warped_corners = [(int(round(corner[0])), int(round(corner[1]))) for corner in warped_corners[0]]
            #print(int_warped_corners)

            #if self.debug:
                #print("ACT_POS: {}".format(self.act_pos))
                #print("CORNERS: {}".format(int_warped_corners))

            #int_warped_corners = self.correctCoords(movement, int_warped_corners) # corrects corners and act_pos

            #if self.debug:
                #print("Corrected ACT_POS: {}".format(self.act_pos))
                #print("Corrected CORNERS: {}".format(int_warped_corners))

            self.mask = self.mkMask(self.model.img)
            #print("Mask size: {}".format(self.mask.shape))
            # 4) check the points with numOfPointsMask
            #num = self.numOfPointsInMask(int_warped_corners)

            new = np.zeros([self.model.img.shape[0], self.model.img.shape[1], 4], np.uint8)

            new = cv2.warpPerspective(frame.img, H, (self.model.img.shape[1], self.model.img.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
            cv2.imwrite("/home/milan/last_warped.png", new)

            new_mask = self.mkMask(new)

            # dst first, then src
            np.copyto(self.model.img, new, where=np.array(new_mask, dtype=np.bool))
            cv2.imwrite("/home/milan/result.png", self.model.img)
            #np.copyto(self.model.img, new, where=np.array(self.mask, dtype=np.bool))
            #self.model.img = new

            if self.debug:
                cv2.imshow("expanded", self.model.img)
                cv2.waitKey(0)
                #print("New img size = {}".format(self.model.img.shape))

            if self.debug:
                print("Number of detected KP's: {}".format(len(self.model.kp)))

            #self.act_pos = tuple((item[0]+movement[0], item[1]+movement[1]) for item in self.act_pos)
