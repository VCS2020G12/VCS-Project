import glob
import shutil
import os
import sys
from os import listdir
from os.path import isfile, join

src_img_dir = "../images"
dst_img_dir = "../newimages"

src_lbl_dir = "../labels"
dst_lbl_dir = "../newlabels"

try:
    os.mkdir(dst_img_dir)
except OSError:
    print ("Creation of the directory %s failed" % dst_img_dir)
else:
    print ("Successfully created the directory %s " % dst_img_dir)

try:
    os.mkdir(dst_lbl_dir)
except OSError:
    print ("Creation of the directory %s failed" % dst_lbl_dir)
else:
    print ("Successfully created the directory %s " % dst_lbl_dir)

reduce_factor = int(sys.argv[1])

index = 1
count = 0

files = [f for f in listdir("../images/") if isfile(join("../images/", f))]
dim = len(files)

for i, jpgfile in enumerate(glob.iglob(os.path.join(src_img_dir, "*.jpg"))):

    if index == 5:
        shutil.copy(jpgfile, dst_img_dir)  # copy image
        shutil.copy(jpgfile.replace(src_img_dir, src_lbl_dir).replace(".jpg", ".txt"), dst_lbl_dir)  # copy label

        count += 1
        index = 1
    else:
        index += 1

    print("Progress:", i*100/dim)

print("Total files:", str(count))
