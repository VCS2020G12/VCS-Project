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


def setup():
    """
    Fill painting_db with paintings data contained in data.csv and compute their hashes
    :return: None.
    """

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


def match_painting(img):
    """
    Compute hash bit differences between img and paintings in painting_db
    :param img: input image to use.
    :return: Matching painting (the one having less differences with img and below threshold)
            or None if all differences are above treshold.
    """

    threshold = 20

    img_row, img_col = dhash.dhash_row_col(img)
    img_hash = dhash.format_hex(img_row, img_col)
    img_hash = int(img_hash, 16)
    differences = []

    # Check difference between img and painting_db
    for painting in painting_db:
        differences.append(dhash.get_num_bits_different(img_hash, painting.hash))

    if min(differences) < threshold:
        return painting_db[differences.index(min(differences))]
    else:
        return None


if __name__ == '__main__':
    # SETUP PHASE #
    setup()

    # CHECK A NEW FRAME #
    tic = time.time() # Just for measuring time
    painting = match_painting(Image.open("011.png"))
    toc = time.time()
    if painting is not None:
        print("Execution time:", toc-tic, "s")
        print(painting.file_name)
        print(painting.title)
        print(painting.author)
