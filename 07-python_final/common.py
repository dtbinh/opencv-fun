#!/usr/bin/env python

import numpy as np
import cv2
import copy

def keyPoint2Point(kp):
    """
    A helping function for conversion from KeyPoint to tuple representing point
    coordinates (eg. '(20, 11)').

    Returns a numpy array of tuples representing points coordinates.
    """
    return np.array([p.pt for p in kp], np.float32)


def point2Keypoint(points, prev_kp=None, kp_size=31):
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
            # Not sure what effects will this have (messing with KP size)
            kp.append(cv2.KeyPoint(points[i][0], points[i][1], kp_size))

    return kp


def filterGoodKeyPoints(prev_frame, frame):
    """
    Filters KeyPoints to only Good KeyPoints.

    Returns 'good_matches'.
    """
    matcher = cv2.DescriptorMatcher_create("BruteForce")
    print("Prev: {}, current: {}".format(len(prev_frame.desc), len(frame.desc)))
    matches = matcher.match(prev_frame.desc, frame.desc)

    print '#matches:', len(matches)
    dist = [m.distance for m in matches]

    print 'distance: min: %.3f' % min(dist)
    print 'distance: mean: %.3f' % (sum(dist) / len(dist))
    print 'distance: max: %.3f' % max(dist)

    # threshold: half the mean
    thres_dist = (sum(dist) / len(dist)) * 0.5

    # keep only the reasonable matches
    good_matches = [m for m in matches if m.distance < thres_dist]

    print '#selected matches:', len(good_matches)
    prev_gkp = []
    gkp = []
    for i in range(len(good_matches)):
        prev_gkp.append(prev_frame.kp[good_matches[i].queryIdx])
        gkp.append(frame.kp[good_matches[i].trainIdx])

    print "GKP01/02: ", len(prev_gkp), len(gkp)

    prev_frame.kp = copy.copy(prev_gkp)
    frame.kp = copy.copy(gkp)

    return good_matches


def findHomographyMatrix(prev_frame, frame, good_matches):
    """
    Returns a homography matrix. 'good_matches' are to be obtained by running
    'filterGoodKeyPoints() function'.
    """
    prev_points = keyPoint2Point(prev_frame.kp)
    points = keyPoint2Point(frame.kp)

    H, status = cv2.findHomography(prev_points, points, cv2.RANSAC, 1.0)
    print '%d / %d inliers/matched' % (np.sum(status), len(status))
    # do not draw outliers (there will be a lot of them)
    kp_pairs = [matches for matches, flag in zip(good_matches, status) if flag]

    #prev_gkp = []
    #gkp = []
    #for m in kp_pairs:
        #prev_gkp.append(prev_frame.kp[m.queryIdx])
        #gkp.append(frame.kp[m.trainIdx])

    return H
