#!/usr/bin/env python

import argparse
import sys

from settings import s


def main():
    """
    Main function of the application.
    Gets apps parameters and runs the compositing.
    """

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=
"""Application can be controlled interactively using following keys:

  SPACE               toggles pause mode; when paused, points to be tracked can be marked
  's'                 toggles visibility of user marked points
  ESC, 'q'            quits the application""",
        epilog="Other parameters can be adjusted by editing the 'settings.py' file.")
    parser.add_argument("--input", help="input device number or filename (0)", metavar="SRC", dest="video_source")
    parser.add_argument("--surf-hessian", help="hessian threshold parameter of SURF detector (1000)", type=int, metavar="VAL", dest="surf_hessian")
    parser.add_argument("--model-hessian", help="hessian threshold parameter of SURF detector for model image (1200)", type=int, metavar="VAL", dest="model_surf_hessian")

    parser.add_argument("--width", help="model image width (1500)", type=int, metavar="VAL", dest="model_w")
    parser.add_argument("--height", help="model image height (2500)", type=int, metavar="VAL", dest="model_h")

    parser.add_argument("--frame-skip", help="number of input frames to be skipped (0)", type=int, metavar="VAL", dest="skip")
    parser.add_argument("--max-move", help="maximal movement of camera for an image to be added to model (100)", type=int, metavar="VAL", dest="max_mv")
    parser.add_argument("--scale", help="scale of output model image (0.5)", type=float, metavar="VAL", dest="scale")

    args = parser.parse_args()

    for opt,val in vars(args).items():
        if val != None:
            s[opt] = val

    # import the Compositor class
    from compositor import Compositor

    # Run the compositor
    compositor = Compositor()
    compositor.run()

if __name__ == "__main__":
    main()
