import matplotlib.pyplot as plt
import json
import cv2
import os
import numpy as np
import h5py
from tqdm import tqdm
from shutil import copyfile


def extract_frames_by_boolean(boolean_frames, video_data):
    return video_data[boolean_frames]


def map_to_original_boolean_frames(sample_frames, frame_per_sample, video_len):
    original_frames = []
    for i in sample_frames:
        for j in range(frame_per_sample):
            original_frames.append(i == 1.0 or i == True)
    original_frames = original_frames[:video_len]
    return original_frames


def extract_boolean_frames(scores, summary_len, return_with_frames=False):
    print(summary_len)
    items = list(map(lambda x: (x[0], x[1]), enumerate(scores)))
    items.sort(key=lambda x: x[1], reverse=True)
    prediction_summary_frames = items[:summary_len]
    prediction_summary_frames.sort(key=lambda x: x[0])
    prediction_summary_frames = list(map(lambda x: x[0], prediction_summary_frames))
    prediction_boolean_frames = []
    for i in range(len(scores)):
        if i in prediction_summary_frames:
            prediction_boolean_frames.append(True)
        else:
            prediction_boolean_frames.append(False)
    if return_with_frames:
        return prediction_boolean_frames, prediction_summary_frames
    else:
        return prediction_boolean_frames


def extract_prediction_summary_boolean(meta):
    summary_len = meta.gt_summary[meta.gt_summary == 1].shape[0]
    return extract_boolean_frames(meta.prediction_scores, summary_len)


def save_video(output_path, video_frames, frame_per_second=29.97, desc=""):
    img_size = (video_frames.shape[2], video_frames.shape[1])
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"XVID"), frame_per_second, img_size, True)
    for i in range(video_frames.shape[0]):
        frame = video_frames[i, :, :, :]
        writer.write(frame)
    writer.release()


def extract_summary_videos(meta, video_data, frame_per_sample=15):
    prediction_summary_boolean = extract_prediction_summary_boolean(meta)
    original_prediction_summary_boolean = map_to_original_boolean_frames(prediction_summary_boolean, frame_per_sample, video_data.shape[0])
    original_groundtruth_summary_boolean = map_to_original_boolean_frames(meta.gt_summary, frame_per_sample, video_data.shape[0])
    prediction_summary_frames = extract_frames_by_boolean(original_prediction_summary_boolean, video_data)
    groundtruth_summary_frames = extract_frames_by_boolean(original_groundtruth_summary_boolean, video_data)
    return groundtruth_summary_frames, prediction_summary_frames
