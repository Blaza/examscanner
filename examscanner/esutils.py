from examscanner import imutils
import numpy as np
import cv2


_REF_IMAGE_HEIGHT = 1280

def fit_to_scale(image):
    """
    Our reference image has a height of 1280 (_REF_IMAGE_HEIGHT) pixels and
    is in portrait mode. From it we took the templates which we will try to
    find. Thus we want to turn our working image to portraitand scale it to
    the same height so they are both of the similar size to make our work
    easier later (e.g. try fewer scales for multiple scale pattern matching)
    """
    # get dimensions of image
    (iH, iW) = image.shape[:2]

    # turn image to portrait if it was landscape
    if iW > iH:
        image = imutils.change_aspect(image)

    # resize the image so we are close to scale
    result = imutils.resize(image, height=_REF_IMAGE_HEIGHT)

    return(result)



def fix_orientation(image):
    """
    We can get images that are sideways or upside down and we want our
    image to be in portrait mode and with the writing on the bottom, so
    we will here make sure the image is in correct layout.

    Firstly, we make the image portrait. The result may be the correct
    orientation or upside down.
    We rely on the fact that all input fields on the notebook will be on
    the bottom half of the image. We filter only the long horizontal lines
    in the image which should represent input fields and get an image that
    is nearly blank, with only long lines grouped in an area. We then find
    the 'centroid' (average point) of the image, which will give us a point
    in the center of that area. Taking into consideration the position of
    that point (is it in the upper or lower half of the image) we determine
    whether the image should be flipped.

    We use morphological transformations (erosion + dilation) to get
    the horizontal lines image. We use a wide and short kernel to search
    for long lines. For reference, see:
    http://stackoverflow.com/questions/19094642/
    http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html#opening
    """

    (iH, iW) = image.shape[:2]

    # make image grayscale which is easier to work with
    gray = imutils.to_grayscale(image)

    # turn image to portrait if needed
    if iW > iH:
        gray = imutils.change_aspect(gray)


    # detect edges in the grayscale image
    lines = cv2.Canny(gray, 50, 200)

    # get an image consisting of only long horizontal lines which are input
    # fields we use to check for orientation
    linek = np.zeros((3,20),dtype=np.uint8)
    linek[1,...] = 1
    lines = cv2.morphologyEx(lines, cv2.MORPH_OPEN, linek ,iterations=1)

    # calculate moments to find the centroid, i.e. the mean point of the image
    # which tells us whether the writing is on the bottom or top of the image
    M = cv2.moments(lines)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    # if the writing (input lines) is at the top half of the image
    # rotate it 180deg
    if cy < iH / 2:
        result = imutils.rotate(image, 180)
    else:
        result = image

    return(result)
