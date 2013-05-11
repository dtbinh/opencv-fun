#!/usr/bin/env python

"""
Source code file containing the Compositor class implementation.
"""

import numpy as np
import cv2

from frame import Frame
from model import Model
from settings import s

class Compositor:
    """
    Class representing the Compositor object that provides methods for taking
    frames, processing them and putting them together in the Model instance.
    """
    model = None

    def __init__(self, video_source=s["video_source"]):
        """
        Initializes an instance of Compositor class.

        Parameter video_source can be number of video device or filename of
        a video file.
        """
        self.cap = cv2.VideoCapture(video_source);

        if not self.cap.isOpened():
            print("Could not open: {}".format(video_source))
            exit()

        self.model_window = cv2.namedWindow("model")
        cv2.setMouseCallback("model", self.onMouseClick)

        self.paused = False
        self.show_points = True

        self.frame = self.grabNextFrame()
        Compositor.model = Model(self.frame)
        self.prev_frame = None

        self.text_pos = (int(40*s["scale"]), int(80*s["scale"]))

    def grabNextFrame(self):
        """
        Function reads next image from video capture device/file and returns
        created Frame object using the image previously read.
        """
        for x in range(s["skip"]+1):
            ret, img = self.cap.read()

            if not ret:
                return None

        return Frame(img, s["crop"])


    def addFrameToModel(self, movement=(0.0, 0.0)):
        """
        Function takes a frame instance and its movement coords and calls
        the appropriate Model methods in order to add the frame to it.

        The movement is a tuple of Point coords: (y, x)
        """
        # Make movement coords integers:
        movement = [int(round(item)) for item in movement]

        self.frame.detectKeyPoints()
        Compositor.model.add(self.frame, movement)


    def onMouseClick(self, event, x, y, flags, param):
        """
        Callback function for mouse clicking.
        """
        inv_scale = 1/s["scale"]

        pt = (x*inv_scale, y*inv_scale)
        if self.paused and event == cv2.EVENT_LBUTTONDOWN:
            self.model.addUserPoint(pt)

            Compositor.model.drawPoints(self.drawed_model, Compositor.model.user_points)
            cv2.imshow("model", cv2.resize(self.drawed_model, dsize=(0,0), fx=s["scale"], fy=s["scale"]))


    def run(self):
        """
        The main working function of Compositor class. It reads images from
        video source in a loop and process them so that the final model is
        created and further processed.
        """
        # Process first frame first:
        self.addFrameToModel()

        # Movement sum in format (y, x)
        movement_sum = (0.0, 0.0)

        while True:
            self.prev_frame = self.frame
            self.frame = self.grabNextFrame()

            if self.frame == None:
                # add the last frame to the Model:
                self.frame = self.prev_frame
                self.addFrameToModel(movement_sum)
                break

            # tracking:
            tracked = self.frame.trackKeyPoints(self.prev_frame)

            if tracked == None or tracked < s["min_tracked"]:
                self.frame.detectKeyPoints()
                continue

            # copy images for drawing:
            self.drawed_model = np.copy(Compositor.model.model.img)
            drawed_frame = np.copy(self.frame.img)

            # homography:
            H = Compositor.model.computeHomography(self.frame)

            if H == None:
                self.frame.detectKeyPoints()
                Compositor.model.drawStr(self.drawed_model, self.text_pos, "Not Enough KeyPoint Tracked!")

                cv2.imshow("frame", drawed_frame)
                cv2.imshow("model", cv2.resize(self.drawed_model, dsize=(0,0), fx=s["scale"], fy=s["scale"])) # TODO: settings

                ch = cv2.waitKey(1)
                if ch == 27 or ch == ord('q'): # ESC or q
                    cv2.destroyAllWindows()
                    break
                elif ch == ord('s'):
                    self.show_points = not(self.show_points)
                elif ch == ord(' '):
                    self.paused = True
                    Compositor.model.drawStr(self.drawed_model, self.text_pos, "Paused, press SPACE to resume")
                    cv2.imshow("model", cv2.resize(self.drawed_model, dsize=(0,0), fx=s["scale"], fy=s["scale"])) # TODO: settings
                    while self.paused:
                        ch = cv2.waitKey(1)
                        if ch == ord(' '):
                            self.paused = False
                            break
                        elif ch == 27 or ch == ord('q'):
                            cv2.destroyAllWindows()
                            return

                continue

            # transformation of corners according to the homography matrix:
            warped_corners = Compositor.model.warpCorners(self.frame, H)

            if self.show_points and Compositor.model.user_points:
                warped_user_points = Compositor.model.warpUserPoints(H, warped_corners)
                Compositor.model.drawPoints(drawed_frame, warped_user_points)
                Compositor.model.drawPoints(self.drawed_model, Compositor.model.user_points)

            if Compositor.model.cornerTooFarOut(warped_corners, s["max_offset"]):
                Compositor.model.drawStr(self.drawed_model, self.text_pos, "Out of Model!")
            else:
                Compositor.model.drawRect(self.drawed_model, warped_corners)


            if abs(movement_sum[0]) > s["max_mv"] or abs(movement_sum[1]) > s["max_mv"] or sum(movement_sum[:]) > s["max_mv"]:
                self.addFrameToModel(movement_sum)
                movement_sum = (0.0, 0.0)
                continue

            movement_sum = tuple(sum(item) for item in zip(movement_sum, self.frame.displacement))

            cv2.imshow("frame", drawed_frame)
            cv2.imshow("model", cv2.resize(self.drawed_model, dsize=(0,0), fx=s["scale"], fy=s["scale"]))
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord('q'): # ESC or q
                cv2.destroyAllWindows()
                break
            elif ch == ord('s'):
                self.show_points = not(self.show_points)
            elif ch == ord(' '):
                self.paused = True
                Compositor.model.drawStr(self.drawed_model, self.text_pos, "Paused, press SPACE to resume")
                cv2.imshow("model", cv2.resize(self.drawed_model, dsize=(0,0), fx=s["scale"], fy=s["scale"]))
                while self.paused:
                    ch = cv2.waitKey(1)
                    if ch == ord(' '):
                        self.paused = False
                        break
                    elif ch == 27 or ch == ord('q'):
                        cv2.destroyAllWindows()
                        return

