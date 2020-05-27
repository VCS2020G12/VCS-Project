import os
import cv2
import detection.darknet as darknet
from pathlib import Path
import rectification
import retrieval
import face_detect
from PIL import Image
import time
import keyboard

# Parameters
RUNNING = True  # Used to pause or resume the program


def convert_back(x, y, w, h):
    """
    Get min and max x, y values of detection.
    :param x: pt x.
    :param y: pt y.
    :param w: width.
    :param h: height.
    :return: mix and max x, y.
    """
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cv_draw_boxes(detections, img):
    """
    Draw rectangle regions contained in detections on the image
    :param detections: regions to draw.
    :param img: image to draw on.
    :return: image with detections drawn.
    """
    for detection in detections:
        x, y, w, h = detection[2][0], detection[2][1], detection[2][2], detection[2][3]
        xmin, ymin, xmax, ymax = convert_back(float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
        cv2.putText(img, detection[0].decode() + " [" + str(round(detection[1] * 100, 2)) + "]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 255, 0], 2)
    return img


def cv_cut_boxes(detections, img):
    """
    Crop image using detections regions.
    :param detections: regions to use.
    :param img: image to crop.
    :return: a list containing all cropped regions in image.
    """

    imgs = []

    for detection in detections:
        x, y, w, h = detection[2][0], detection[2][1], detection[2][2], detection[2][3]
        xmin, ymin, xmax, ymax = convert_back(float(x), float(y), float(w), float(h))

        crop = img[ymin:ymax, xmin:xmax].copy()
        if crop.size != 0:
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            imgs.append(crop)

    return imgs


netMain = None
metaMain = None
altNames = None


def pause():
    """
    Pause the elaboration pressing <SPACE> key. Resume pressing the same key. (sudo is required in order to work)
    :return: None.
    """

    global RUNNING
    RUNNING = not RUNNING
    time.sleep(0.3)


def process_video(video_file, JUMP=0, MAX_FPS = 0, OUTPUT = False):
    """
    Process video_file applying all the processing workflow starting from painting detection.
    :param video_file: input file to use.
    :param JUMP: jump factor to use. Default: 0, no jump.
    :param MAX_FPS: Maximum FPS desired by the user (0 means no limit).
    :param OUTPUT: Used to generate or not (True / False) an output file.
    :return: None.
    """

    if os.getuid() == 0:  # Enable pause elaboration with key press
        keyboard.on_press_key("SPACE", lambda _: pause())

    global metaMain, netMain, altNames
    config_path = "./detection/yolo-obj.cfg"
    weight_path = str(Path.home()) + "/yolo-obj_final.weights"
    meta_path = "./detection/obj.data"
    if not os.path.exists(config_path):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(config_path)+"`")
    if not os.path.exists(weight_path):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weight_path)+"`")
    if not os.path.exists(meta_path):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(meta_path)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(config_path.encode("ascii"),
                                          weight_path.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(meta_path.encode("ascii"))
    if altNames is None:
        try:
            with open(meta_path) as metaFH:
                meta_contents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", meta_contents, re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            names_list = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in names_list]
                except TypeError:
                    pass
        except Exception:
            pass

    cap = cv2.VideoCapture(video_file)
    cap.set(3, 1280)
    cap.set(4, 720)
    if OUTPUT:
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        size = (darknet.network_width(netMain), darknet.network_height(netMain))
        out = cv2.VideoWriter()
        out.open("out.mp4", fourcc, 30, size, True)

    # Image to be reused at each detection
    darknet_image = darknet.make_image(darknet.network_width(netMain), darknet.network_height(netMain), 3)

    jump_index = 0  # Tracks the jump factor
    old_rois = 0  # Tracks the number of roi windows in the previous iteration
    room_id = None  # Defines id room in the video

    # Setup Painting retrieval data
    retrieval.setup()

    # Setup Face Detection
    face_detect.setup()

    while True:
        while not RUNNING:
            time.sleep(0.1)

        tic = time.time()  # Keep track of FPS

        ret, frame_read = cap.read()

        # If the frame was not grabbed, then we have reached the end of the stream
        if not ret:
            break

        if jump_index < JUMP:
            jump_index += 1
        else:
            jump_index = 0
            frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (darknet.network_width(netMain), darknet.network_height(netMain)),
                                       interpolation=cv2.INTER_LINEAR)

            darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())

            detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)

            painting_detections = [detection for detection in detections if detection[0] == b'painting']
            people_detections = [detection for detection in detections if detection[0] == b'person']

            image = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            rois = cv_cut_boxes(painting_detections, frame_resized)
            actual_frame = image.copy()
            image = cv_draw_boxes(detections, image)

            cv2.imshow('Painting Detection', image)
            cv2.moveWindow('Painting Detection', 0, 0)

            if OUTPUT:
                out.write(image)

            # Show every painting detected
            actual_rois = 0
            position = 100
            for i, roi in enumerate(rois):
                cv2.imshow('ROI ' + str(i+1), roi)
                cv2.moveWindow("ROI " + str(i + 1), image.shape[1] + position, 0)

                # Call rectification on roi
                rectified_image = rectification.rectification(roi)

                if rectified_image is not None:
                    match_painting = retrieval.match_painting(Image.fromarray
                                                                     (rectified_image.astype('uint8'), 'RGB'))

                    if match_painting is not None:
                        room_id = match_painting.room
                        print("..................................Painting detected..................................")
                        print("Title:   ", match_painting.title)
                        print("Author:  ", match_painting.author)
                        print("Room:    ", match_painting.room)
                        print("Db image:", match_painting.file_name)
                        print(".....................................................................................")

                position += roi.shape[1] + 20
                actual_rois += 1

            # Delete useless old rois
            if actual_rois < old_rois:
                for i in range(actual_rois, old_rois):
                    cv2.destroyWindow("ROI " + str(i+1))
            old_rois = actual_rois

            # Print detected people
            if len(people_detections) > 0:
                faces = face_detect.get_faces(actual_frame)
                print("..................................People detected..................................")
                print("Number of people:     ", len(people_detections))
                if len(people_detections) - faces >= 0:
                    print("Looking at a painting:", len(people_detections) - faces)
                else:
                    print("Looking at a painting: 0")
                if room_id is None:
                    print("Room id:               To be determined")
                else:
                    print("Room id:              ", room_id)
                print(".....................................................................................")

            cv2.waitKey(2)

            iteration_time = (time.time() - tic)
            wait_time = 0
            if MAX_FPS != 0:
                if 1 / MAX_FPS > iteration_time:
                    wait_time = (1 / MAX_FPS) - iteration_time
                    time.sleep(wait_time)
            # Keep track of performance
            print("FPS:", round(1 / (wait_time + iteration_time), 2))

    cap.release()
    cv2.destroyAllWindows()
    if OUTPUT:
        out.release()
