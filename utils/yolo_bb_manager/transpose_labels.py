import cv2
import os
from os import listdir
from os.path import isfile, join

input = "../labels/"
destination = "../newlabels/"

files = [f for f in listdir(input) if isfile(join(input, f))]
files = sorted(files)

try:
    os.mkdir(destination)
except OSError:
    print ("Creation of the directory %s failed" % destination)
else:
    print ("Successfully created the directory %s " % destination)

dim = len(files)
for i, file_name in enumerate(files):
    print("Progress:", str(round(((100 * i) / dim), 2)) + "%")

    with open(input + file_name, 'r') as file:
        out_file = open(destination + file_name, "w")
        n_row = 0
        for row in file:
            if n_row > 0:
                out_file.write("\n")
            robj_class, rcenter_X, rcenter_Y, rwidth, rheight = row.split()
            out_file.write(robj_class + " " + rcenter_Y + " " + rcenter_X + " " + rheight + " " + rwidth)
            n_row += 1

        out_file.close()

print("Done")