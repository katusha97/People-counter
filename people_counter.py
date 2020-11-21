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
import tensorflow as tf
import inspect


ap = argparse.ArgumentParser()
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
ap.add_argument("-v", "--input_video", type=str,
                help="path to video file")
args = vars(ap.parse_args())

CLASSES = ["person", "cat", "tv", "car", "meatballs", "marinara sauce",
            "tomato soup", "chicken noodle soup", "french onion soup",
            "chicken breast", "ribs", "pulled pork", "hamburger", "cavity", "PeopleTopView"]

print("[INFO] loading model...")

net = tf.saved_model.load(args['model'])


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
    confidences = detections['detection_scores'].numpy()[0]
    indices = detections['detection_classes'].numpy()[0]
    boxes = detections['detection_boxes'].numpy()[0]
    for i in np.arange(0, int(detections['num_detections'].numpy()[0])):
        confidence = confidences[i]
        if confidence < args["confidence"]:
            continue

        idx = int(indices[i]) - 1
        if idx < len(CLASSES):
            print(f'{CLASSES[idx]} class')
        else:
            print(f'{idx} not in CLASSES')
            continue

        box = [boxes[i][1], boxes[i][0], boxes[i][3], boxes[i][2]]
        box = box * np.array([W, H, W, H])
        (startX, startY, endX, endY) = box.astype("int")
        tracker = dlib.correlation_tracker()
        rect = dlib.rectangle(startX, startY, endX, endY)
        tracker.start_track(rgb, rect)
        trackers.append(tracker)
    print('---------------------------------------------------')
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


def set_info(frame, totalUp, totalDown, status):
    info = [
        ("Up", totalUp),
        ("Down", totalDown),
        ("Status", status),
    ]
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


def draw_rect(frame, rect):
    cv2.rectangle(frame, rect[0:2], rect[2:4], (255, 0, 0), thickness=3)


def print_centroid(frame, totalUp, totalDown):
    text = "ID {}".format(objectID)
    cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)


def get_images_from_images(args):
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

vs = cv2.VideoCapture()

def get_image_from_video(args):
    vs = cv2.VideoCapture(args.get("input_video"))
    while vs.isOpened():
        ret, frame = vs.read()
        if not ret:
            yield None
        yield frame


if args.get("input", False):
    frame_gen = get_images_from_images(args)
elif args.get("input_video", False):
    frame_gen = get_image_from_video(args)
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
        detections = net.signatures['serving_default'](tf.convert_to_tensor([frame], dtype='uint8'))
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

    for rect in rects:
        draw_rect(frame, rect)

    set_info(frame, totalUp, totalDown, status)

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
# if not args.get("input", False):
#     vs.stop()
# else:
vs.release()
cv2.destroyAllWindows()
