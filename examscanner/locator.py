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

def match_templates(image, templates, mscale=0.975, Mscale=1.2, n=10):
    """
    We use multiple scale template matching (as found here: http://www.pyimagesearch.com/2015/01/26/multi-scale-template-matching-using-python-opencv/)

    We try resizing the image in ``n`` different scales from ``mscale`` to
    ``Mscale`` and see where we get the best match for our template.

    We will find all templates given to our function in the provided image
    and return the maximum value of corelation coefficient, the location of
    the point of maximum and the ratio of the resizing, so we can map the
    location to the original image.

    In addition to the several scales, we will try several filters on the
    image to try and get an even better estimate of the location. A quick
    google search can find a reference to the specified filters. For example
    in one test image, a contrast change was needed in order to find the
    location correctly, so we use the CLAHE contrast equalization to fix that.
    """

    # We create a list of dictionaries to store both the template for
    # comparison and the information about best location estimate.

    tpl_info= []
    # prepare templates for use, we use canny edge detection to improve
    # accuracy as it's easier to compare lines than images.
    for template in templates:
        temp = imutils.to_grayscale(template)
        tpl_info.append( { 'template': cv2.Canny(temp, 50, 200),
                           'loc_info': None } )

    # set filters we will use
    filters = [FLT_identity, FLT_clahe]

    gray = imutils.to_grayscale(image)

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

            for tpl in tpl_info:
                (tH, tW) = tpl['template'].shape[:2]
                # if the resized image is smaller than the template, then break
                # from the loop
                if resized.shape[0] < tH or resized.shape[1] < tW:
                    continue

                result = cv2.matchTemplate(edged, tpl['template'], cv2.TM_CCOEFF_NORMED)
                (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

                # if we have found a new maximum correlation value, then ipdate
                # the bookkeeping variable
                if tpl['loc_info'] is None or maxVal > tpl['loc_info'][0]:
                    tpl['loc_info'] = (maxVal, maxLoc, r, f.__name__)
        # stop here if the coefficient is larger than 0.5 for all templates
        # which is a good enough match, in order to improve speed
        # [IMPROVE] it is a bit ugly solution, should be cleaned up some time
        if all( [ tpl['loc_info'][0] > 0.5 for tpl in tpl_info] ):
            break

    # unpack location information for all templates and return it
    locations = [ tpl['loc_info'] for tpl in tpl_info]

    return(locations)
