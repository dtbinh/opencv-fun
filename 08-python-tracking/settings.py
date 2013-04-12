#!/usr/bin/env python

"""
This file is used for storing various settings.
"""

# Debug settings: #
# --------------- #

debug_frame = True
debug_model = True


# Keypoint detector settings (SURF): #
# ---------------------------------- #

surf_hessian = 500


# Frame settings: #
# --------------- #

# size of cropped area = size/frame_crop_border
frame_crop_border = 20


# Model settings: #
# --------------- #

# size of model in format: (Y, X, 4)
model_size = (960, 2048, 4)
