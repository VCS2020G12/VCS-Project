import cv2
from os import listdir
from os.path import isfile, join

files = [f for f in listdir("../images/") if isfile(join("../images/", f))]
files = sorted(files)

dim = len(files)

n_files = 0

for i, file_name in enumerate(files):
    img = cv2.imread("../images/" + file_name)
    img_height, img_width, channels = img.shape

    with open("../labels/" + file_name.split('.')[0] + ".txt", 'r') as file:
        for row in file:
            robj_class, rcenter_X, rcenter_Y, rwidth, rheight = row.split()
            rcenter_X_px = int(float(rcenter_X) * img_width)
            rcenter_Y_px = int(float(rcenter_Y) * img_height)
            rwidth_px = int(float(rwidth) * img_width)
            rheight_px = int(float(rheight) * img_height)

            pt1_x = int(rcenter_X_px - (float(rwidth_px) / 2))
            pt1_y = int(rcenter_Y_px - (float(rheight_px) / 2))
            pt2_x = int(pt1_x + float(rwidth_px))
            pt2_y = int(pt1_y + float(rheight_px))

            cv2.rectangle(img, (pt1_x, pt1_y), (pt2_x, pt2_y), (255,0,0), 2)
            print("Progress:", str(round(((i*100)/dim), 2)) + "%", "Nome file:", file_name.split('.')[0], "rcenter_X:", rcenter_X, "rcenter_Y:", rcenter_Y, "rwidth:", rwidth, "rheight:", rheight)

        cv2.imshow("img", img)
        cv2.waitKey(0)
    n_files += 1

print("Processed files:", n_files)
