import cv2
import json
import math
import numpy as np


def read_scores(scores_path):
    with open(scores_path) as fp:
        scores = json.load(fp)
        return scores


def read_video_meta(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    return frame_count, frame_width, frame_height, fps


def read_video_sample(video_path, sample_period):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sample_frame_count = math.ceil(frame_count / sample_period)
    buf = np.empty((sample_frame_count, frame_height, frame_width, 3), np.dtype('uint8'))
    fc = 0
    sc = 0
    ret = True
    while fc < frame_count and ret:
        if fc % sample_period == 0:
            ret, buf[sc] = cap.read()
            sc += 1
        fc += 1
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    return buf, fps


def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    buf = np.empty((frame_count, frame_height, frame_width, 3), np.dtype('uint8'))
    fc = 0
    ret = True
    while fc < frame_count  and ret:
        ret, buf[fc] = cap.read()
        fc += 1
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    return buf, fps


def read_write_selected_video_frames(video_path, output_path, boolean_frames):
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    buf = np.empty((frame_height, frame_width, 3), np.dtype('uint8'))
    fc = 0
    ret = True

    frame_size = (frame_width, frame_height)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"XVID"), fps, frame_size, True)

    while fc < frame_count and ret:
        ret, buf = cap.read()
        if boolean_frames[fc]:
            writer.write(buf)
        fc += 1
    cap.release()
    writer.release()


def read_frames(video_path, frames):
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    buf = np.empty((frame_height, frame_width, 3), np.dtype('uint8'))
    fc = 0
    ret = True

    frame_size = (frame_width, frame_height)
    frame_data = []
    while fc < frame_count and ret:
        ret, buf = cap.read()
        if fc in frames:
            frame_data.append(buf)
        fc += 1
    cap.release()
    return frame_data


def read_video_meta(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return frame_count, frame_width, frame_height, fps