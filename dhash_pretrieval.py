import dhash
from PIL import Image
from os import listdir
from os.path import isfile, join
import pandas as pd
import time


class Painting:
    def __init__(self, file_name, title, author, room, hash):
        self.file_name = file_name
        self.title = title
        self.author = author
        self.room = room
        self.hash = hash


# Set parameters
data_folder = "./utils/"
db_folder = "./paintings_db/"
painting_db = []


def match_painting(img):
    img_row, img_col = dhash.dhash_row_col(Image.open(img))
    img_hash = dhash.format_hex(img_row, img_col)
    img_hash = int(img_hash, 16)
    differences = []

    # Check difference between img and painting_db
    for painting in painting_db:
        differences.append(dhash.get_num_bits_different(img_hash, painting.hash))

    if min(differences) < 20:
        return painting_db[differences.index(min(differences))]
    else:
        return None


if __name__ == '__main__':
    # SETUP PHASE #
    data = pd.read_csv(data_folder + "data.csv")
    images = [f for f in listdir(db_folder) if isfile(join(db_folder, f))]
    # Fill painting_db list with paintings infos
    for file_name in images:
        row, col = dhash.dhash_row_col(Image.open(db_folder + file_name))
        painting = Painting(file_name,
                            data[data["Image"] == file_name]["Title"].item(),
                            data[data["Image"] == file_name]["Author"].item(),
                            data[data["Image"] == file_name]["Room"].item(),
                            int(dhash.format_hex(row, col), 16))
        painting_db.append(painting)

    # CHECK A NEW FRAME #
    tic = time.time() # Just for measuring time
    painting = match_painting('011.png')
    toc = time.time()
    if painting:
        print("Execution time:", toc-tic, "s")
        print(painting.file_name)
        print(painting.title)
        print(painting.author)
