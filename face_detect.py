import numpy as np
import cv2


net = None


def setup():
    """
    Load the model.
    :return: None.
    """

    global net
    net = cv2.dnn.readNetFromCaffe("./detection/face_detection/deploy.prototxt.txt",
                                   "./detection/face_detection/res10_300x300_ssd_iter_140000.caffemodel")


def get_faces(image):
    """
    Detect faces in image.
    :param image: image to apply the detection.
    :return: Number of faces detected.
    """

    faces = 0
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            faces += 1

    return faces
