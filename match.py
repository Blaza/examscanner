# USAGE
# python match.py --template cod_logo.png --images images

# import the necessary packages
import numpy as np
import argparse
import cv2
import glob
from examscanner import imutils, esutils, locator, reader
from examscanner.locator import InputField

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--template", required=False, help="Path to template image")
ap.add_argument("-T", "--tpl_name", required=False, help="Name of template")
ap.add_argument("-i", "--image", required=False)
ap.add_argument("-I", "--images", required=False)
ap.add_argument("-v", "--visualize",
    help="Flag indicating whether or not to visualize each iteration")
args = vars(ap.parse_args())

#if args["template"] is not None:
     #load the image image, convert it to grayscale, and detect edges
    #template = cv2.imread(args["template"])
#else:
    #template = cv2.imread(args["tpl_name"])

templates = [cv2.imread('templates/tpl_index.png'),
             cv2.imread('templates/tpl_points.png')]

input_index = InputField(templates[0], [(2, 95), (105, 195)])
input_points = InputField(templates[1], [(2, 310)])

if args['images'] is not None:
    image_paths = glob.glob(args["images"] + "/*")
else:
    image_paths = [args['image']]
for imagePath in image_paths:
    print(imagePath)
    # loop over the images to find the template in
    # load the image, convert it to grayscale, and initialize the
    # bookkeeping variable to keep track of the matched region
    image = cv2.imread(imagePath)
    #image = cv2.bilateralFilter(image, 11, 17, 17)
    image = esutils.prepare_image(image)

    # crop the image to only bottom half where all the writing is
    image = imutils.get_bottom_half(image)

    loc_info = locator.match_template(image, input_index.template)


    # unpack the bookkeeping variable and compute the (x, y) coordinates
    # of the bounding box based on the resized ratio
    (m, maxLoc, r, flt) = loc_info
    print('I: scale',"{0:.3f}".format(1/r),'max:',"{0:.3f}".format(m), flt)
    print('-----------------------')


    inputs = locator.get_inputs(image, input_index, loc_info)

    cv2.imshow('first', inputs[0])
    cv2.imshow('second', inputs[1])
    cv2.waitKey(0)

    in0 = reader.extract_text(inputs[0])
    in1 = reader.extract_text(inputs[1])

    cv2.imshow('first', in0)
    cv2.imshow('second', in1)
    cv2.waitKey(0)
