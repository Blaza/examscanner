"""
This module is used for reading of the numbers from input images.

We start by removing the long horizontal line that indicates input on the notebook and returning
a binary image that contains only an outline of the number. That is done by :func:`extract_text`
function.
"""

from examscanner import imutils
import numpy as np
import cv2


def extract_text(image):
    gray = imutils.to_grayscale(image)
    # detect edges
    edged = cv2.Canny(gray, 50, 200)

    # Use morphological transforms (erode + dilate) to find long
    # horizontal lines from the image so we have clearer digits
    linek=np.zeros((3,20),dtype=np.uint8)
    linek[1,...]=1
    x = cv2.morphologyEx(edged, cv2.MORPH_OPEN, linek, iterations=1)

    # remove the horizontal lines
    edged -= x

    return(edged)
