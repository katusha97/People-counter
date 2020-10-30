from centroidtracker import CentroidTracker
from trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import os


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
                help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
                help="path to Caffe pre-trained model")
ap.add_argument("-i", "--input", type=str,
                help="path to optional input video file")
ap.add_argument("-o", "--output", type=str,
                help="path to optional output video file")
ap.add_argument("-c", "--confidence", type=float, default=0.4,
                help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip-frames", type=int, default=10,
                help="# of skip frames between detections")
args = vars(ap.parse_args())

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])


writer = None
ct = CentroidTracker(maxDisappeared=40)
trackers = []
trackableObjects = {}
totalFrames = 0
totalDown = 0
totalUp = 0
fps = FPS().start()


def create_writer(W, H, args):
    if args["output"] is not None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        return cv2.VideoWriter(args["output"], fourcc, 30,
                                 (W, H), True)
    return None


def create_trackers(detections, trackers):
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > args["confidence"]:
            idx = int(detections[0, 0, i, 1])
            if CLASSES[idx] != "person":
                continue

            box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")
            tracker = dlib.correlation_tracker()
            rect = dlib.rectangle(startX, startY, endX, endY)
            tracker.start_track(rgb, rect)
            trackers.append(tracker)  # compute the (x, y)-coordinates of the bounding box
            box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")
            tracker = dlib.correlation_tracker()
            rect = dlib.rectangle(startX, startY, endX, endY)
            tracker.start_track(rgb, rect)
            trackers.append(tracker)
    return trackers


def follow_object(objectID, centroid, totalUp, totalDown):
    to = trackableObjects.get(objectID, None)
    if to is None:
        to = TrackableObject(objectID, centroid)
    else:
        y = [c[1] for c in to.centroids]
        direction = centroid[1] - np.mean(y)
        to.centroids.append(centroid)
        if not to.counted:
            if direction < 0 and centroid[1] < H // 2:
                totalUp += 1
                to.counted = True
            elif direction > 0 and centroid[1] > H // 2:
                totalDown += 1
                to.counted = True
    trackableObjects[objectID] = to
    return totalUp, totalDown


def set_info(im, totalUp, totalDown, status):
    info = [
        ("Up", totalUp),
        ("Down", totalDown),
        ("Status", status),
    ]
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


def print_centroid(frame, totalUp, totalDown):
    text = "ID {}".format(objectID)
    cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)


def get_images_from_video(args):
    names = os.listdir(args.get("input"))
    names = sorted(names)
    for name in names:
        frame = cv2.imread(os.path.join(args.get("input"), name))
        frame = np.swapaxes(frame, 0, 1)
        yield frame
    return None


def get_images_from_camera():
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    while True:
        yield vs.read()


if args.get("input", False):
    frame_gen = get_images_from_video(args)
else:
    frame_gen = get_images_from_camera()

for frame in frame_gen:
    if args["input"] is not None and frame is None:
        break
    frame = imutils.resize(frame, width=500)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    (H, W) = frame.shape[:2]

    status = "Waiting"
    rects = []
    if totalFrames % args["skip_frames"] == 0:
        status = "Detecting"
        trackers = []
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
        net.setInput(blob)
        detections = net.forward()

        trackers = create_trackers(detections, trackers)


    else:
        for tracker in trackers:
            status = "Tracking"
            tracker.update(rgb)
            pos = tracker.get_position()
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())
            rects.append((startX, startY, endX, endY))

    cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)
    objects = ct.update(rects)

    for (objectID, centroid) in objects.items():
        (f, s) = follow_object(objectID, centroid, totalUp, totalDown)
        totalUp = f
        totalDown = s
        print_centroid(frame, totalUp, totalDown)
    set_info(cv2, totalUp, totalDown, status)

    if writer is None:
        writer = create_writer(W, H, args)
    if writer is not None:
        writer.write(frame)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    totalFrames += 1
    fps.update()

fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
if writer is not None:
    writer.release()
if not args.get("input", False):
    vs.stop()
else:
    vs.release()
cv2.destroyAllWindows()
