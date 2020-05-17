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


# ------------ CORNER DETECTION: 2 WAYS ------------
'''
Results are pretty similar, the first uses the goodFeaturesToTrack method, the second the Harris method
'''

def findCorners(src):
    out_corners = []
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    corners = cv2.goodFeaturesToTrack(gray, 4, 0.01, 5)
    try:
        corners = np.int0(corners)
    except:
        return out_corners, False

    for corner in corners:
        x, y = corner.ravel()
        out_corners.append([x, y])
        cv2.circle(src, (x, y), 3, 255, - 1)

    # check if it's barely regular

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
                print("Count:", count_corners, "Center coords: (", j, ",", i, ")")
                cv2.circle(src, (j, i), 5, 0, 2)

    cv2.imshow("Corners", src)
    cv2.waitKey(0)
    return corners


# ------------ FIND A POLYGON IN THE IMAGE ------------
'''
This code takes an input image (RGB, already cut in ROI shape), detects inside one polygon 
with 4 edges and draws the result on a blank image which is then returned
'''

def drawPolygon2(src_img):
    new_image = np.zeros(src_img.shape, src_img.dtype)
    blank = 255 * np.ones_like(src_img)
    alpha = 4.0  # Simple contrast control
    beta = 8  # Simple brightness control
    new_image[:, :, :] = np.clip(alpha * src_img[:, :, :] + beta, 0, 255)
    cv2.imshow('Contrast', new_image)
    cv2.waitKey(0)

    gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Gray Image', gray)
    cv2.waitKey(0)

    flt = cv2.bilateralFilter(gray, 11, 17, 17)
    cv2.imshow('Filtered Image', flt)
    cv2.waitKey(0)

    _, threshold = cv2.threshold(flt, 60, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow('Threshold Image', threshold)
    cv2.waitKey(0)

    #edged = cv2.Canny(flt, 60, 200)
    #cv2.imshow('Canny Image', edged)
    #cv2.waitKey(0)

    cnts = cv2.findContours(threshold.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in cnts:
        area = cv2.contourArea(cnt)

        # Shortlist based on area --> doesn't detect too small areas
        if area > 100:
            # Find Polygons shapes
            approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)
            if (len(approx) == 4):  # select only if the polygon has 4 edges
                cv2.drawContours(blank, [approx], 0, (0, 0, 255), 2)  # draw the contour on the blank image
                cv2.imshow('Contoured Blank Image', blank)

    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    screenCnt = None
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.015 * peri, True)
        # if our approximated contour has four points, then
        # we can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            cv2.drawContours(src_img, [screenCnt], -1, (0, 255, 0), 3)
    cv2.imshow("Draw Image", src_img)
    cv2.waitKey(0)



def drawPolygon(src_img):
    gray_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    blank = 255 * np.ones_like(src_img)

    # flt = cv2.GaussianBlur(gray_img, (5, 5), 0)
    # flt = cv2.cv2.bilateralFilter(gray_img, 11, 17, 17)
    # cv2.imshow('Filtered Image', flt)
    # cv2.waitKey(0)

    # Thresholding --> binary image
    _, threshold = cv2.threshold(gray_img, 60, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow('Threshold Image', threshold)
    cv2.waitKey(0)

    # Find contours on the thresholded image
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Shortlist based on area --> doesn't detect too small areas
        if area > 400:
            # Find Polygons shapes
            approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)
            if (len(approx) == 4):  # select only if the polygon has 4 edges
                cv2.drawContours(blank, [approx], 0, (0, 0, 255), 2)  # draw the contour on the blank image
                cv2.imshow('Contoured Blank Image', blank)
                cv2.waitKey(0)
                return blank, True  # break to the first found


    _, threshold = cv2.threshold(gray_img, 120, 255, cv2.THRESH_BINARY)
    # cv2.imshow('Threshold Image', threshold)
    # cv2.waitKey(0)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:
            approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)
            if (len(approx) == 4):
                cv2.drawContours(blank, [approx], 0, (0, 0, 255), 2)
                cv2.imshow('Contoured Blank Image', blank)
                return blank, True

    # Exiting the windows if 'q' is pressed on the keyboard
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

    return blank, False



def drawPolygon_params(src_img, alpha, beta, bil_param1, bil_param2, threshold, approx):
    new_image = np.zeros(src_img.shape, src_img.dtype)
    blank = 255 * np.ones_like(src_img)

    new_image[:, :, :] = np.clip(alpha * src_img[:, :, :] + beta, 0, 255)
    cv2.imshow('Contrast', new_image)
    cv2.waitKey(0)

    gray_img = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)

    # flt = cv2.GaussianBlur(gray_img, (5, 5), 0)
    flt = cv2.bilateralFilter(gray_img, bil_param1, bil_param2, bil_param2)
    cv2.imshow('Filtered Image', flt)
    cv2.waitKey(0)

    # Thresholding --> binary image
    _, threshold = cv2.threshold(gray_img, threshold, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow('Threshold Image', threshold)
    cv2.waitKey(0)

    # Find contours on the thresholded image
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 400:
            try:
                approx = cv2.approxPolyDP(cnt, approx * cv2.arcLength(cnt, True), True)
            except:
                return blank, False
            if (len(approx) == 4):
                cv2.drawContours(blank, [approx], 0, (0, 0, 255), 2)
                cv2.imshow('Contoured Blank Image', blank)
                cv2.waitKey(0)
                return blank, True  # break to the first found

    return blank, False


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


        roi = img[max(pt1_y,0):max(pt2_y,0), max(pt1_x,0):max(pt2_x,0)]

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


if __name__ == '__main__':
    files = [f for f in listdir("data/images/") if isfile(join("data/images/", f))]
    files = sorted(files)

    for i, file_name in enumerate(files):
        print(file_name)
        roi_img = cut2ROI(file_name)
        cv2.imshow('ROI Image', roi_img)
        cv2.waitKey(0)
        found = False

        #drawPolygon2(roi_img)  # blank image with the rectangle
        # blank_pol, found = drawPolygon_prova(roi_img)
        # draPolygon_params(src_img, alpha, beta, bil_param1, bil_param2, threshold, approx)

        blank_pol, found = drawPolygon_params(roi_img, 10.0, -500, 17, 17, 60, 0.009)  # blocco 0, 1, 3
        if not found:
            print('1 didnt work')
            blank_pol, found = drawPolygon_params(roi_img, 1.0, 0, 17, 17, 60, 0.009)  # img chiare
            if not found:
                print('2 didnt work')
                blank_pol, found = drawPolygon_params(roi_img, 10.0, -300, 11, 50, 100, 0.009)  # dorate
                if not found:
                    print('3 didnt work')
                    blank_pol, found = drawPolygon_params(roi_img, 1.0, 0, 17, 17, 120, 0.009)  # img chiare
                    if not found:
                        print('4 didnt work')
                        blank_pol, found = drawPolygon_params(roi_img, 7.0, -500, 17, 17, 80, 0.009)  # grigie
                        if not found:
                            print('5 didnt work')
                            blank_pol, found = drawPolygon_params(roi_img, 7.0, -500, 11, 17, 200, 0.009)  # extra
                            if not found:
                                blank_pol, found = drawPolygon(roi_img)


        if found:
            corners_src, found = findCorners(blank_pol)
            if found:
                corners_src = np.asarray(corners_src, dtype=np.int)
                result = four_point_transform(roi_img, corners_src)
                # cv2.imshow('ROI Image', roi_img)
                cv2.imshow('Perspective Transform', result)
                cv2.waitKey(0)
            else:
                print('No corners found')
        else:
            print('No shape found')


        '''
        
        # ---------- OLD LOLLY CODE ----------
        x, y, w, h = cv2.getWindowImageRect('ROI Image')
        corners_dst = [[w,h], [0,h], [w,0], [0,0]]
        print(corners_dst)
        pts1 = np.float32(corners_src)
        pts2 = np.float32(corners_dst)
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        result = cv2.warpPerspective(roi_img, matrix, (w, h))
        
        
        blank_pol, found = draPolygon_params(roi_img, 10.0, -500, 17, 17, 60, 0.009)  # blocco 0, 1, 3
        if not found:
            print('1 didnt work')
            blank_pol, found = draPolygon_params(roi_img, 1.0, 0, 17, 17, 60, 0.009)  # img chiare
            if not found:
                print('2 didnt work')
                blank_pol, found = draPolygon_params(roi_img, 10.0, -200, 11, 50, 120, 0.009)  # dorate
                if not found:
                    print('3 didnt work')
                    blank_pol, found = draPolygon_params(roi_img, 1.0, 0, 17, 17, 120, 0.009)  # img chiare
                    if not found:
                        print('4 didnt work')
                        blank_pol, found = draPolygon_params(roi_img, 7.0, -500, 17, 17, 80, 0.009)  # grigie

        '''