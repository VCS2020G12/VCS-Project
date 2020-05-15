import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from scipy import signal
# from sklearn.preprocessing import normalize
from skimage.transform import resize

source_window = 'Source image'
corners_window = 'Corners detected'
max_thresh = 255


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
        exit("No corner found. Try with another image!")

    for corner in corners:
        x, y = corner.ravel()
        out_corners.append([x, y])
        cv2.circle(src, (x,y), 3, 255, - 1)

    cv2.imshow('Corner', src)
    cv2.waitKey(0)

    return out_corners


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
            if int(dst_norm[i,j]) > thresh:
                corners.append([j, i])
                count_corners += 1
                print("Count:", count_corners, "Center coords: (", j, ",", i, ")")
                cv2.circle(src, (j,i), 5, 0, 2)

    cv2.imshow("Corners", src)
    cv2.waitKey(0)
    return corners


# ------------ FIND A POLYGON IN THE IMAGE ------------
'''
This code takes an input image (RGB, already cut in ROI shape), detects inside one polygon 
with 4 edges and draws the result on a blank image which is then returned
'''
def drawPolygon(src_img):
    gray_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    blank = 255 * np.ones_like(src_img)
    cv2.imshow('White Image', blank)
    cv2.waitKey(0)

    # Median blur makes it worse in this case
    # Canny makes it worse in this case

    # Thresholding --> binary image
    _, threshold = cv2.threshold(gray_img, 60, 255, cv2.THRESH_BINARY_INV)
    # _, threshold = cv2.threshold(gray_img, 120, 255, cv2.THRESH_BINARY_INV) # detects distorted but is less precise
    cv2.imshow('Threshold Image', threshold)
    cv2.waitKey(0)

    # Find contours on the thresholded image
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Shortlist based on area --> doesn't detect areas too small
        if area > 100:
            # Find Polygons shapes
            approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)
            if (len(approx) == 4):  # select only if the polygon has 4 edges
                cv2.drawContours(blank, [approx], 0, (0, 0, 255), 2)  # draw the contour on the blank image
                cv2.imshow('Contoured Blank Image', blank)
                break  # break to the first found

    # Exiting the windows if 'q' is pressed on the keyboard
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

    return blank


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
            rcenter_X_px = int(float(rcenter_X) * img_w)
            rcenter_Y_px = int(float(rcenter_Y) * img_h)
            rwidth_px = int(float(rwidth) * img_w)
            rheight_px = int(float(rheight) * img_h)

            pt1_x = int(rcenter_X_px - (float(rwidth_px) / 2))
            pt1_y = int(rcenter_Y_px - (float(rheight_px) / 2))
            pt2_x = int(pt1_x + float(rwidth_px))
            pt2_y = int(pt1_y + float(rheight_px))

        roi = img[pt1_y:pt2_y, pt1_x:pt2_x]

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


'''
Order corners: bottom-right, bottom-left, top-right, top-left
'''
def orderCorners(src_corners):
    xy_sums = []
    x_values = []

    pts = []

    for point in src_corners:
        xy_sums.append(point[0] + point[1])
        x_values.append(point[0])

    # pt0
    pts.append(src_corners[xy_sums.index(max(xy_sums))])
    del x_values[xy_sums.index(max(xy_sums))]
    del src_corners[xy_sums.index(max(xy_sums))]
    del xy_sums[xy_sums.index(max(xy_sums))]

    # pt3
    pt3 = src_corners[xy_sums.index(min(xy_sums))]
    del x_values[xy_sums.index(min(xy_sums))]
    del src_corners[xy_sums.index(min(xy_sums))]
    del xy_sums[xy_sums.index(min(xy_sums))]

    # pt1
    pts.append(src_corners[x_values.index(min(x_values))])

    # pt2
    pts.append(src_corners[x_values.index(max(x_values))])

    # Append also pt3
    pts.append(pt3)

    return pts


if __name__ == '__main__':
    files = [f for f in listdir("data/images/") if isfile(join("data/images/", f))]
    files = sorted(files)

    for i, file_name in enumerate(files):

        roi_img = cut2ROI(file_name)
        cv2.imshow('ROI Image', roi_img)
        cv2.waitKey(0)

        blank_pol = drawPolygon(roi_img)  # blank image with the rectangle
        corners_src = findCorners(blank_pol)

        corners_src = orderCorners(corners_src)

        print(corners_src)
        # print(cornerHarris(blank_pol, 200))

        # Get window dims --> they'll be our dest points

        x, y, w, h = cv2.getWindowImageRect('ROI Image')

        corners_dst = [[w,h], [0,h], [w,0], [0,0]]
        print(corners_dst)

        # ----------- missing function to order points --------------

        pts1 = np.float32(corners_src)
        pts2 = np.float32(corners_dst)
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        result = cv2.warpPerspective(roi_img, matrix, (w, h))

        cv2.imshow('ROI Image', roi_img)
        cv2.imshow('Perspective Transform', result)
        cv2.waitKey(0)


