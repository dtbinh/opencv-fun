#!/usr/bin/env python

"""
Settings file

Stores all application parameters
"""

# Frame
# ==============================================================================

# Whether or not to crop frames
crop = True

# Crop by number of pixels
crop_by = 20

# Hessian parameter of SURF detector
surf_hessian = 1000


# Model
# ==============================================================================

# Model width
model_w = 1500

# Model height
model_h = 2500

# Hessian parameter of SURF detector for model image:
model_surf_hessian = 1200

# Minimal number of key point matches before homography computation
min_matches = 10

# Drawing colour (B, R, G, A)
drawing_col = (0, 255, 0, 255)

# Maximum distance of camera corner position out of model boundaries in pixels
max_offset = 20

# Minimal area of unmapped pixels in percents when adding new image to the model
min_unmapped_area = 10

# Minimal area of mapped pixels in percents when adding new image to the model
min_mapped_area = 50


# Compositor
# ==============================================================================

# Video source (number of device or filename)
video_source = 0

# Number of frames to skip when reading input
skip = 0

# Scale of displayed model image
scale = 0.5

# Minimal number of tracked key points
min_tracked = 20

# Maximal movement of camera in x, y and combined directions before a frame
# is added to model
max_mv = 100


# Other
# ==============================================================================
