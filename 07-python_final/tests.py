#!/usr/bin/env python

import cv2
import numpy as np

from frame import Frame
from model import Model
from compositor import Compositor

def TestImread(img_file="../img/s01.jpg"):
    img = cv2.imread(img_file)

    print("Image shape: {}".format(img.shape))

    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return img


def TestFeatures():
    img = cv2.imread("../img/s01.jpg")
    f = Frame(img, crop=True)
    kp = f.detectKeyPoints()

    f._showDetectedKeyPoints()


def TestFeaturesMask():
    img = cv2.imread("../img/s01.jpg")
    ff = Frame(img, crop=True)
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros(img.shape[:2], np.uint8)

    mask[200:500, 0:300] = 1

    kp = ff.detectKeyPoints(mask=mask)
    ff._showDetectedKeyPoints()


def TestFPTracking():
    i = cv2.imread("../img/s01.jpg")
    f = Frame(i, crop=True)
    f.detectKeyPoints()

    j = cv2.imread("../img/s02.jpg")
    g = Frame(j, crop=True)
    kp = g.trackKeyPoints(f)


def TestCompositor():
    compositor = Compositor("../img/video/poster01-lowres.mp4")
    #compositor = Compositor(1, rt_result=True, debug=True)
    compositor.run()


tests_dict = {#"TestImread": TestImread,
              #"TestFeatures": TestFeatures,
              #"TestFeaturesMask": TestFeaturesMask,
              #"TestFPTracking": TestFPTracking,
              "TestCompositor": TestCompositor}


if __name__ == "__main__":
    counter = 1
    for key, value in tests_dict.items():
        print("#{}: {}".format(counter, key))
        value()
        print("#{} => DONE".format(counter))
        counter += 1
