from imutils.video import FPS
import numpy as np
import imutils
import cv2


def people_detection(video, prototxt, model):
    # Output frames list
    out_img = []

    # Initialize the color of rectangles
    color = np.random.uniform(0, 255, size=(1, 3))

    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(prototxt, model)

    print("[INFO] starting video processing...")
    vs = cv2.VideoCapture(video)
    fps = FPS().start()

    # Loop over video frames
    while True:
        frame = vs.read()[1]

        # Exit loop if end of file is reached
        if frame is None:
            break

        frame = imutils.resize(frame, width=1280)
        # Convert the frame to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        # pass the blob through the network and obtain the detections and predictions
        net.setInput(blob)
        detections = net.forward()

        # Loop over detections
        for i in np.arange(0, detections.shape[2]):
            # Extract the confidence of the detection
            confidence = detections[0, 0, i, 2]

            # Extract the detection class
            idx = int(detections[0, 0, i, 1])

            # Filter out weak detection that are lower than a fixed value
            # and draw only people detections filtering only idx 15
            if confidence > 0.2 and idx == 15:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Construct the string label and the rectangle
                label = "{}: {:.2f}%".format("Person", confidence * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color[0], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color[0], 2)

        # Save the output frame
        out_img.append(frame)

        # show the output frame
        cv2.imshow("Preview", frame)
        cv2.waitKey(1)
        # update the FPS counter
        fps.update()

    # Stop the timer and display info
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # Save the output video (out_name, type, fps, frame_size)
    out = cv2.VideoWriter('../out.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (out_img[0].shape[1], out_img[0].shape[0]))

    for i in range(len(out_img)):
        out.write(out_img[i])
    out.release()

    # Clean cv2 windows
    cv2.destroyAllWindows()


if __name__ == "__main__":

    # Arguments set-up
    video = "../Project material/videos/011/1.mp4"
    prototxt = "MobileNetSSD_deploy.prototxt.txt"
    model = "MobileNetSSD_deploy.caffemodel"

    people_detection(video, prototxt, model)
