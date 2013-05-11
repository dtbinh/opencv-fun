#!/usr/bin/env python

"""
Source code file containing functions called by various class methods.
"""

import numpy as np
import cv2

from settings import s


def keyPoint2Point(kp):
    """
    A helping function for conversion from KeyPoint to tuple representing point
    coordinates (eg. '(20, 11)').

    Returns a numpy array of tuples representing points coordinates.
    """
    return np.array([p.pt for p in kp], np.float32)


def point2Keypoint(points, prev_kp=None, kp_size=1):
    """
    A helping function for conversion from Point representation to KeyPoint.

    When 'prev_kp' of the same size not supplied, the method cannot determine
    the size of each KeyPoint, therefore assignes 'kp_size' to all of them.
    """
    kp = []
    for i in range(len(points)):
        if (prev_kp != None) and (len(prev_kp) == len(points)):
            kp.append(cv2.KeyPoint(points[i][0], points[i][1], prev_kp[i].size))
        else:
            kp.append(cv2.KeyPoint(points[i][0], points[i][1], kp_size))

    return kp


def filterKPUsingHomography(prev_frame, frame):
    """
    Filters Key Points using the RANSAC algorithm.
    """
    prev_points = keyPoint2Point(prev_frame.kp)
    points = keyPoint2Point(frame.kp)


    if len(prev_points) != len(points):
        return (None, None)

    if len(points) < s["min_tracked"]:
        return (None, None)

    H, status = cv2.findHomography(prev_points, points, cv2.RANSAC, 3.0)

    prev_gkp = []
    gkp = []
    prev_gkp = [kp for kp, flag in zip(prev_frame.kp, status) if flag]
    gkp = [kp for kp, flag in zip(frame.kp, status) if flag]

    return (prev_gkp, gkp)
