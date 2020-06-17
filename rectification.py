import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import skimage.segmentation as seg
from skimage import img_as_ubyte, img_as_float
import skimage.color as color

DEBUG = False

# ------------ CORNER DETECTION: 2 WAYS ------------
'''
Results are pretty similar, the first uses the goodFeaturesToTrack method, the second the Harris method
'''


def findCorners(src):
    out_corners = []
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    corners = cv2.goodFeaturesToTrack(gray, 4, 0.009, 5)
    try:
        corners = np.int0(corners)
    except:
        return out_corners, False

    for corner in corners:
        x, y = corner.ravel()
        out_corners.append([x, y])
        cv2.circle(src, (x, y), 3, 255, - 1)

    # check if it's barely regular

    if DEBUG:
        cv2.imshow('Corners', src)
        cv2.waitKey(0)

    return out_corners, True



def cornerHarris(src, thresh):
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    count_corners = 0
    corners = []

    # Detecting corners
    dst = cv2.cornerHarris(src_gray, 2, 3, 0.04)  # image, blockSize, apertureSize, k
    # Normalizing
    dst_norm = np.empty(dst.shape, dtype=np.float32)
    cv2.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # Drawing a circle around corners
    for i in range(dst_norm.shape[0]):
        for j in range(dst_norm.shape[1]):
            if int(dst_norm[i, j]) > thresh:
                corners.append([j, i])
                count_corners += 1
                if DEBUG:
                    print("Count:", count_corners, "Center coords: (", j, ",", i, ")")
                cv2.circle(src, (j, i), 5, 0, 2)

    if DEBUG:
        cv2.imshow("Corners", src)
        cv2.waitKey(0)
    return corners


# ------------ FIND A POLYGON IN THE IMAGE: 2 WAYS ------------
'''
First method (polygon) searches and detects a rectangular shape after working on the image
Second method (mask) applies other kinds of approaches such as slic before detecting a rectangle in the form of a mask
'''

def rectification_polygon(src_img, alpha, beta, threshold):

    new_image = np.zeros(src_img.shape, src_img.dtype)
    blank = 255 * np.ones_like(src_img)

    new_image[:, :, :] = np.clip(alpha * src_img[:, :, :] + beta, 0, 255)
    #cv2.imshow('Contrast', new_image)
    #cv2.waitKey(0)

    gray_img = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)

    #flt = cv2.GaussianBlur(gray_img, (5, 5), 0)

    _, threshold = cv2.threshold(gray_img, threshold, 255, cv2.THRESH_BINARY_INV)
    #cv2.imshow('Threshold Image', threshold)
    #cv2.waitKey(0)

    # Find contours on the thresholded image
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #x, y, w, h = cv2.boundingRect(contours)


    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        cnt_size = w * h
        if area > 10000:
            try:
                approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)
            except:
                return blank, False

            if (len(approx) == 4):
                cv2.drawContours(blank, [approx], 0, (0, 0, 255), 1)

                # discard the contour if it's equal to the image
                if abs(cnt_size - src_img.shape[0] * src_img.shape[1]) <= 0.1:
                    continue
                #cv2.imshow('Contoured Blank Image', blank)
                #cv2.waitKey(0)

                return blank, True  # break to the first found
            #if (len(approx) == 8):
                #cv2.drawContours(blank, [approx], 0, (0, 0, 255), 1)
                #cv2.imshow('Contoured Blank Image', blank)
                #cv2.waitKey(0)
                #return blank, True  # break to the first found

    return blank, False


def rectification_mask(roi_img):
    dst = roi_img
    # kernel = np.ones((5, 5)) / (25)
    # roi_img = cv2.filter2D(roi_img, -1, kernel)

    # roi_img = cv2.bilateralFilter(res2, 9, 75, 75)
    Z = roi_img.reshape((-1, 3))
    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    K = 10
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 5, cv2.KMEANS_PP_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((roi_img.shape))

    clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8, 8))
    lab = cv2.cvtColor(res2, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
    l, a, b = cv2.split(lab)  # split on 3 different channels
    l2 = clahe.apply(l)  # apply CLAHE to the L-channel
    lab = cv2.merge((l2, a, b))  # merge channels
    
    image_slic = seg.slic(img, n_segments=4, sigma=1, compactness=5, start_label=1)
    image = img_as_ubyte(color.label2rgb(image_slic, img, kind='avg', bg_label=0))

    mask = np.zeros(image.shape, dtype=np.uint8)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    edges = cv2.Canny(thresh, 100, 200)
    # Perform morpholgical operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    # erosion = cv2.erode(edges, kernel, iterations=1)

    # opening = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel, iterations=1)
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
    mask_seg[10:-10, 10:-10] = 255
    mask = mask & mask_seg

    return mask, True


# ------------ CUT ROI IMAGE ------------
'''
Cut the original image to the shape of the ROI
'''

def cut2ROI(file_name):
    img = cv2.imread("data/images/" + file_name)
    img_h, img_w, c = img.shape

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


        roi = img[max(pt1_y,0):max(pt2_y,0), max(pt1_x,0):max(pt2_x,0)]

    return roi


# ------------ Order and transform points ------------
'''
Get, order and transform 4 boundary points
'''


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


def rectification(roi_img):
    found = False
    i = 0
    params = [[10.0, -500, 60], [1.0, 0, 60], [10.0, -300, 100], [2.0, 0, 120], [7.0, -500, 90],
              [7.0, -500, 200], [1.0, 0, 150], [1.0, 0, 160], [1.0, 0, 120], [7.0, -800, 140]]

    while not found and i != 10:
        #print(i, 'did not work')
        mask, found = rectification_polygon(roi_img, params[i][0], params[i][1], params[i][2])
        i+=1

    method = 1

    if not found:
        method = 2
        try:
            mask, found = rectification_mask(roi_img)
        except:
            found = False

    if found:
        if DEBUG:
            print('Method:', method)
        corners_src, found = findCorners(mask)
        if found:
            corners_src = np.asarray(corners_src, dtype=np.int)
            result = four_point_transform(roi_img, corners_src)
            cv2.imshow('Perspective Transform', result)

            if DEBUG:
                cv2.waitKey(0)
            return result
        else:
            if DEBUG:
                print('No corners found')
    else:
        if DEBUG:
            print('No shape found')
    return None


if __name__ == '__main__':
    files = [f for f in listdir("data/images/") if isfile(join("data/images/", f))]
    files = sorted(files)

    for i, file_name in enumerate(files):
        print(file_name)
        roi_img = cut2ROI(file_name)
        cv2.imshow('ROI Image', roi_img)
        cv2.waitKey(0)

        rectification(roi_img)
