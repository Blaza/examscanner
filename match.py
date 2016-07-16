# USAGE
# python match.py --template cod_logo.png --images images

# import the necessary packages
import numpy as np
import argparse
import cv2
import glob
from examscanner import imutils, esutils, locator

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
    image = esutils.fix_orientation(image)
    image = esutils.fit_to_scale(image)

    # crop the image to only bottom half where all the writing is
    image = imutils.get_bottom_half(image)

    locations = locator.match_templates(image, templates)


    # unpack the bookkeeping variable and compute the (x, y) coordinates
    # of the bounding box based on the resized ratio
    (m, maxLoc0, r, flt) = locations[0]
    (m1, maxLoc1, r1, flt1) = locations[1]
    print('I: scale',"{0:.3f}".format(1/r),'max:',"{0:.3f}".format(m), flt)
    print('P: scale',"{0:.3f}".format(1/r1),'max:',"{0:.3f}".format(m1), flt1)
    print('-----------------------')


if len(image_paths) == 1:
    # reset the image changed by clahe
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (tH, tW) = templates[0].shape[:2]
    (startX, startY) = (int(maxLoc0[0] * r), int(maxLoc0[1] * r))
    (endX, endY) = (int((maxLoc0[0] + tW) * r), int((maxLoc0[1] + tH) * r))

    # draw a bounding box around the detected result and display the image
    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)

    # draw two bounding boxes for the 'broj dosijea' fields. We have some defaults
    # for the reference image and scale the boxes using the ratio r. This is
    # hardcoded and maybe could be improved in the future. For r=1 we get the
    # values we got empirically for defaults.
    cv2.rectangle(image, (endX + int(2*r), startY - int(15*r)),
                         (endX + int(95*r), endY + int(5*r)),
                         (0, 255, 0), 1)
    cv2.rectangle(image, (endX + int(105*r), startY - int(15*r)),
                         (endX + int(195*r), endY + int(5*r)),
                         (255, 0, 0), 1)

    (tH, tW) = templates[1].shape[:2]
    (startX, startY) = (int(maxLoc1[0] * r1), int(maxLoc1[1] * r1))
    (endX, endY) = (int((maxLoc1[0] + tW) * r1), int((maxLoc1[1] + tH) * r1))

    # draw a bounding box around the detected result and display the image
    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
    cv2.rectangle(image, (endX + int(2*r1), startY - int(15*r1)),
                         (endX + int(310*r1), endY + int(5*r1)),
                         (0, 255, 0), 1)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    exit(0)
    # detect edges in the resized, grayscale image
    edged = cv2.Canny(gray, 50, 200)

    # Use morphological transforms (erode + dilate) to remove long
    # horizontal lines from the image so we have clearer digits
    linek=np.zeros((3,20),dtype=np.uint8)
    linek[1,...]=1
    x=cv2.morphologyEx(edged, cv2.MORPH_OPEN, linek ,iterations=1)
    edged -= x

    cv2.rectangle(edged, (startX, startY), (endX, endY), 175, 2)

    if args['tpl_name']=="tpl_index.png":
        # draw two bounding boxes for the 'broj dosijea' fields. We have some defaults
        # for the reference image and scale the boxes using the ratio r. This is
        # hardcoded and maybe could be imporeved in the future. For r=1 we get the
        # values we got empirically for defaults.
        cv2.rectangle(edged, (endX + int(2*r), startY - int(15*r)),
                             (endX + int(95*r), endY + int(5*r)),
                             255, 1)
        cv2.rectangle(edged, (endX + int(105*r), startY - int(15*r)),
                             (endX + int(195*r), endY + int(5*r)),
                             255, 1)
    else:
        cv2.rectangle(edged, (endX + int(2*r), startY - int(15*r)),
                             (endX + int(310*r), endY + int(5*r)),
                             255, 1)

    cv2.imshow("edged", edged)
    cv2.waitKey(0)