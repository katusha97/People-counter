import cv2
import numpy as np
import argparse


def read_video(path: str) -> list:
    cap = cv2.VideoCapture(path)
    video = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        video.append(frame)
    return video


def draw_rectangle(frame: np.array, point_one: tuple, point_two: tuple) -> np.array:
    new_frame = np.copy(frame)
    cv2.rectangle(new_frame, point_one, point_two, (255, 0, 0), thickness=10)
    return new_frame


def draw_rect_on_frames(frames: list, point_one: tuple, point_two: tuple) -> list:
    ans = []
    for frame in frames:
        new_frame = draw_rectangle(frame, point_one, point_two)
        ans.append(new_frame)
    return ans


def save_frames(frames: list, path: str):
    for i, frame in enumerate(frames):
        cv2.imwrite(path + '/' + str(i) + '.jpg', frame)


def coords(s):
    try:
        x, y = map(int, s.split(','))
        return x, y
    except:
        raise argparse.ArgumentTypeError("Coordinates must be x,y")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Work with video')
    parser.add_argument('--input', type=str)

    parser.add_argument("--leftUp", help="Coordinates", type=coords)
    parser.add_argument("--rightDown", help="Coordinates", type=coords)
    parser.add_argument("--numberOfFrames", help="How many frames do you want to save", type=int)

    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    video = read_video(args.input)
    result = draw_rect_on_frames(video[:args.numberOfFrames], args.leftUp, args.rightDown)
    save_frames(result, args.output)
