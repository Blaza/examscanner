"""
The Locator module is used to find the fields in the notebook where the student
id number and points scored on the exam will are written. We use multi-scale
template matching to find those fields with the :func:`match_template` function

Once we find where to look for our inputs, we get images which only contain
input numbers with the :func:`get_inputs` function.

Every function and method used will be described in a bit more detail in the
corresponding documentation.

Functions with names beginning with FLT\_ are filters. They all take
grayscale images and apply some filter to it and return the result.
"""

from examscanner import imutils
from examscanner.consts import _REF_INPUT_HEIGHT
import numpy as np
import cv2

class InputField():
    """
    This class represents one input from the notebook (e.g. index, points,...).

    It has three attributes:

    * template - the template image we use to locate the input in the image
    * input_count - the number of input fields the input needs (e.g. index input takes two)
    * offsets - the list of left and right offsets from the right edge of the template bounding \
            box, found empirically for the reference image.

    In the diagram below, offsets are distances from the right edge od the bounding box

    .. image:: _static/offsets.png
    """
    def __init__(self, template, offsets, input_count=None):
        self.template = template
        self.offsets = offsets
        self.input_count = input_count if input_count is not None else len(offsets)


def FLT_identity(gray):
    """ Identity filter, returns the same image. """
    return(gray)

def FLT_clahe(gray):
    """ CLAHE filter, applies CLAHE contrast equalization """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return(clahe.apply(gray))

def match_template(image, template, mscale=0.975, Mscale=1.2, n=10):
    """
    We use multiple scale template matching (as found here: http://www.pyimagesearch.com/2015/01/26/multi-scale-template-matching-using-python-opencv/)

    We try resizing the image in ``n`` different scales from ``mscale`` to
    ``Mscale`` and see where we get the best match for our template.

    We will find the template given to our function in the provided image
    and return the maximum value of corelation coefficient, the location of
    the point of maximum, the ratio of the resizing, so we can map the
    location to the original image, and the filter that was used for matching.

    In addition to the several scales, we will try several filters on the
    image to try and get an even better estimate of the location. A quick
    google search can find a reference to the specified filters. For example
    in one test image, a contrast change was needed in order to find the
    location correctly, so we use the CLAHE contrast equalization to fix that.
    """

    tpl = imutils.to_grayscale(template)
    tpl = cv2.Canny(tpl, 50, 200)
    (tH, tW) = tpl.shape[:2]

    # set filters we will use
    filters = [FLT_identity, FLT_clahe]

    gray = imutils.to_grayscale(image)

    found = None

    for f in filters:
        # apply the filter
        gray = f(gray)

        # loop over the scales of the image
        for scale in np.linspace(mscale, Mscale, n):
            # resize the image according to the scale, and keep track
            # of the ratio of the resizing
            resized = imutils.resize(gray, width = int(gray.shape[1] * scale))
            r = gray.shape[1] / float(resized.shape[1])

            # detect edges in the resized, grayscale image and apply template
            # matching to find the template in the image
            edged = cv2.Canny(resized, 50, 200)

            # if the resized image is smaller than the template, then break
            # from the loop
            if resized.shape[0] < tH or resized.shape[1] < tW:
                continue

            result = cv2.matchTemplate(edged, tpl, cv2.TM_CCOEFF_NORMED)
            (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

            # if we have found a new maximum correlation value, then update
            # the bookkeeping variable
            if found is None or maxVal > found[0]:
                found = (maxVal, maxLoc, r, f.__name__)

        # stop here if the coefficient is larger than 0.5
        # which is a good enough match, in order to improve speed
        # [IMPROVE] it is a bit ugly solution, should be cleaned up some time
        if found[0] > 0.5:
            break

    return(found)


def get_inputs(image, input_field, loc_info = None):
    """
    With this function we get cut images of input fields which we will use to read the numbers.
    We take an image we are working with and an InputField instance which gives us information
    about which field we are trying to read. There is an optional argument loc_info which tells
    us about the location of the input field's template in the image. If loc_info is not provided,
    it is calculated using the :func:`match_template` function.

    We use the loc_info to calculate the bounding rectangle of the template. Then we take a strip
    from the image of height _REF_INPUT_HEIGHT, scaled using the ratio r given in loc_info, which
    is centered at the template bounding box.

    We then cut out that strip to get input images from the image, using the offset information
    given in input_field. The documentation for :class:`InputField` gives more information
    about how the offsets are given.

    We return a list of input images of the same size as the offset list, which represent the
    number of input fields for the specified InputField. For example, the student id no. (index)
    has two input fields, while the points scored has one input field.
    """

    (tH, tW) = input_field.template.shape[:2]

    if loc_info is None:
        loc_info = match_template(image, input_field.template)

    (_, maxLoc, r, flt) = loc_info

    # get bounding box for input template, scaled using ratio r
    (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
    (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

    # find the middle y coordinate of the bounding box
    centerY = int((maxLoc[1] + tH/2) * r)

    # calculate top and bottom y coordinates of the strip from which we get inputs
    # using the input height from the reference image
    strip_top = int(centerY - _REF_INPUT_HEIGHT * r / 2)
    strip_bottom = int(centerY + _REF_INPUT_HEIGHT * r / 2)

    # cut out the horizontal strip from the image
    strip = image[strip_top:strip_bottom, : ]

    inputs = []
    # cut the strip at x coordinates specified by offsets in input_field
    for offset in input_field.offsets:
        field = strip[ : , endX + int(offset[0]*r) : endX + int(offset[1]*r)]
        inputs.append(field)

    return(inputs)

