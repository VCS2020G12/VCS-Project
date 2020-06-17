import skimage.segmentation as seg
from skimage import img_as_ubyte, img_as_float
import skimage.color as color

import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from scipy import signal, ndimage
# from sklearn.preprocessing import normalize
from skimage.transform import resize
import imutils
from sklearn.cluster import KMeans


# ------------ CUT ROI IMAGE ------------
'''
Cut the original image to the shape of the ROI
'''


def cut2ROI(file_name):
    img = cv2.imread("data/images/" + file_name)
    img_h, img_w, c = img.shape
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    with open("data/labels/" + file_name.split('.')[0] + ".txt", 'r') as file:
        for row in file:
            robj_class, rcenter_X, rcenter_Y, rwidth, rheight = row.split()
            rcenter_X_px = float(rcenter_X) * img_w
            rcenter_Y_px = float(rcenter_Y) * img_h
            rwidth_px = float(rwidth) * img_w
            rheight_px = float(rheight) * img_h

            pt1_x = int(rcenter_X_px - (float(rwidth_px) / 2))
            pt1_y = int(rcenter_Y_px - (float(rheight_px) / 2))
            pt2_x = int(pt1_x + float(rwidth_px))
            pt2_y = int(pt1_y + float(rheight_px))

        roi = img[max(pt1_y, 0):max(pt2_y, 0), max(pt1_x, 0):max(pt2_x, 0)]

    return roi


# ------------ GET ROI boundary points ------------
'''
Get the 4 boundary points of the ROI inside the image
'''


def get_ROI_points(file_name):
    img = cv2.imread("data/images/" + file_name)
    img_h, img_w, c = img.shape

    with open("data/labels/" + file_name.split('.')[0] + ".txt", 'r') as file:
        for row in file:
            robj_class, rcenter_X, rcenter_Y, rwidth, rheight = row.split()
            rcenter_X_px = int(float(rcenter_X) * img_w)
            rcenter_Y_px = int(float(rcenter_Y) * img_h)
            rwidth_px = int(float(rwidth) * img_w)
            rheight_px = int(float(rheight) * img_h)

            # pt4 ---- pt3
            #  |        |
            # pt1 ---- pt2

            pt1 = [int(rcenter_X_px - (float(rwidth_px) / 2)), int(rcenter_Y_px - (float(rheight_px) / 2))]
            pt2 = [int(pt1[0] + float(rwidth_px)), pt1[1]]
            pt3 = [pt2[0], int(pt1[1] + float(rheight_px))]
            pt4 = [pt1[0], pt2[1]]

    return [pt1, pt2, pt3, pt4]


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped


    # Load image, bilaterial blur, and Otsu's threshold
'''
def rectification_Casa(roi_img):
    dst=roi_img
    image_slic = seg.slic(roi_img, n_segments=3)
    roi_img = img_as_ubyte(color.label2rgb(image_slic, roi_img, kind='avg'))
    image = cv2.bilateralFilter(roi_img, 9, 75, 75)
    cv2.imshow('SLIC', roi_img)
    cv2.waitKey(0)

    mask = np.zeros(image.shape, dtype=np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Perform morpholgical operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    close = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)
    #close = cv2.erode(close, kernel, iterations=3)
    cv2.imshow('Close',close)
    cv2.waitKey(0)

    size=5
    kernel=np.ones((size,size))/(size**2)

    close = cv2.filter2D(close, -1,kernel)
    cv2.imshow('Close+filter', close)
    cv2.waitKey(0)
    # Find distorted rectangle contour and draw onto a mask
    cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]


    rect = cv2.minAreaRect(cnts[0])
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(image, [box], 0, (0, 0, 255), 2)
    cv2.imshow('box', image)
    cv2.waitKey(0)
    cv2.fillPoly(mask, [box], (255, 255, 255))

    # Find corners
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(mask, 4, .009, 5)
    #offset = 25
    out_corners=[]
    if corners is not None:

        for corner in corners:
            # x, y = corner.ravel()
            # cv2.circle(image, (x, y), 5, (36, 255, 12), -1)
            # x, y = int(x), int(y)
            # cv2.rectangle(image, (x - offset, y - offset), (x + offset, y + offset), (36, 255, 12), 3)
            # print("({}, {})".format(x, y))
            # for corner in corners:
            x, y = corner.ravel()
            out_corners.append([x, y])
            cv2.circle(mask, (x, y), 3, 255, - 1)

    cv2.imshow('Corners', mask)
    cv2.waitKey(0)
    corners_src = np.asarray(out_corners, dtype=np.int)
    result = four_point_transform(dst, corners_src)
    # cv2.imshow('ROI Image', roi_img)
    cv2.imshow('Perspective Transform', result)
    cv2.waitKey(0)


    # cv2.imshow('image', roi_img)
    # cv2.waitKey(0)
    # cv2.imshow('thresh', thresh)
    # cv2.waitKey(0)
    # cv2.imshow('close', close)
    # cv2.waitKey(0)
    # cv2.imshow('mask', mask)
    # cv2.waitKey(0)
    cv2.destroyAllWindows()
'''
def rectification_Casa(img):
    dst=img
    #kernel = np.ones((5, 5)) / (25)
    #roi_img = cv2.filter2D(roi_img, -1, kernel)

    # roi_img = cv2.bilateralFilter(res2, 9, 75, 75)
    # cv2.imshow('Filter', roi_img)
    # cv2.waitKey(0)

    Z = img.reshape((-1, 3))
    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    K = 10
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 5, cv2.KMEANS_PP_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8, 8))
    lab = cv2.cvtColor(res2, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
    l, a, b = cv2.split(lab)  # split on 3 different channels
    l2 = clahe.apply(l)  # apply CLAHE to the L-channel
    lab = cv2.merge((l2, a, b))  # merge channels

    image_slic = seg.slic(lab, n_segments=4, sigma=1, compactness=5,start_label=1)
    image = img_as_ubyte(color.label2rgb(image_slic, img, kind='avg',bg_label=0))

    mask = np.zeros(image.shape, dtype=np.uint8)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    edges = cv2.Canny(thresh, 100, 200)
    # Perform morpholgical operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    #erosion = cv2.erode(edges, kernel, iterations=1)

    #opening = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel, iterations=1)
    close = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Find distorted rectangle contour and draw onto a mask
    cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    rect = cv2.minAreaRect(cnts[0])
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(close, [box], 0, (0, 0, 255), 2)
    cv2.fillPoly(mask, [box], (255, 255, 255))

    mask_seg = np.zeros(image.shape, dtype=np.uint8)
    mask_seg[10:-10,10:-10] = 255
    mask = mask & mask_seg
    # Find corners
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(mask, 4, .9, 5)
    out_corners=[]
    if corners is not None:
        for corner in corners:
            x, y = corner.ravel()
            out_corners.append([x, y])
            cv2.circle(mask, (x, y), 3, 255, - 1)

    corners_src = np.asarray(out_corners, dtype=np.int)
    result = four_point_transform(dst, corners_src)
    # cv2.imshow('ROI Image', roi_img)
    cv2.imshow('Perspective Transform', result)
    cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    files = [f for f in listdir("data/images/") if isfile(join("data/images/", f))]
    files = sorted(files)

    for i, file_name in enumerate(files):
        print(file_name)
        roi_img = cut2ROI(file_name)
        cv2.imshow('ROI Image', roi_img)
        cv2.waitKey(0)

        found = False
        rectification_Casa(roi_img)

