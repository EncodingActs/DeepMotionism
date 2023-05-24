import tensorflow as tf
import numpy as np
import cv2 as cv
import math
import argparse
import sys

#This represents the body parts indices to compare
parts_to_compare = [(5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (11, 12), (5, 11), (6, 12), (11, 13), (12, 14), (13, 15),
                    (14, 16)]

# Initiate the parser
parser = argparse.ArgumentParser()
parser.add_argument("-V", "--video_path", help="path to the video",
                    default='data/posecuts_frontview/poseface00091219.mov')
parser.add_argument("-M", "--model_path", help="path to the tflite model",
                    default="models/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite")

# Read arguments from the command line
args = parser.parse_args()


def get_frames(video_path, save_frame=False):
    """
    :param video_path: str, path of the video to retrieve frames
    :param save_frame: bool, save on disk
    :return: list of frames
    """
    vidcap = cv.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    ret_frames = []

    while success:
        if save_frame:
            cv.imwrite("frame{}.jpg".format(count), image)  # save frame as JPEG file
        ret_frames.append(image)
        success, image = vidcap.read()
        count += 1
    return ret_frames


def resize_frame(frame, w, h):
    """
    Resize frame to match it the model input dimension
    :param frame: frame tobe resized
    :return: resized frame
    """
    return cv.resize(frame, (w, h))


def regularize(frame):
    return (np.float32(frame) - 127.5) / 127.5


def parse_output(heatmap_data, offset_data, threshold):
    """
    Input:
      heatmap_data - heatmaps for an image. Three dimension array
      offset_data - offset vectors for an image. Three dimension array
      threshold - probability threshold for the key points. Scalar value
    Output:
      array with coordinates of the key points and flags for those that have
      low probability
    """

    # nr of features
    joint_num = heatmap_data.shape[-1]
    pose_kps = np.zeros((joint_num, 3), np.uint32)

    for i in range(joint_num):
        joint_heatmap = heatmap_data[..., i]
        max_val_pos = np.squeeze(np.argwhere(joint_heatmap == np.max(joint_heatmap)))
        # remap to 0..257 from 0..8(ranges)
        remap_pos = np.array(max_val_pos / 8 * 257, dtype=np.int32)

        # add offsets to the box coords
        pose_kps[i, 0] = int(remap_pos[0] + offset_data[max_val_pos[0], max_val_pos[1], i])
        pose_kps[i, 1] = int(remap_pos[1] + offset_data[max_val_pos[0], max_val_pos[1], i + joint_num])
        max_prob = np.max(joint_heatmap)

        if max_prob > threshold:
            if pose_kps[i, 0] < 257 and pose_kps[i, 1] < 257:
                pose_kps[i, 2] = 1

    return pose_kps


def run_inference(interpreter, input_index, output_indexes, frame):
    # Process image
    interpreter.set_tensor(input_index, frame)

    # Runs the computation
    interpreter.invoke()

    # Extract output data from the interpreter
    template_output_data = interpreter.get_tensor(output_indexes[0])
    template_offset_data = interpreter.get_tensor(output_indexes[1])

    # Getting rid of the extra dimension
    return np.squeeze(template_output_data), np.squeeze(template_offset_data)


def angle_length(p1, p2):
    """
    Input:
      p1 - coordinates of point 1. List
      p2 - coordinates of point 2. List
    Output:
      Tuple containing the angle value between the line formed by two input points
      and the x-axis as the first element and the length of this line as the second
      element
    """
    angle = math.atan2(- int(p2[0]) + int(p1[0]), int(p2[1]) - int(p1[1])) * 180.0 / np.pi
    length = math.hypot(int(p2[1]) - int(p1[1]), - int(p2[0]) + int(p1[0]))

    return [round(angle), round(length)]


def get_features(kps, parts=parts_to_compare):
    features = []
    for part in parts:
        features.append(angle_length(kps[part[0]][:2], kps[part[1]][:2]))
    return features


def get_video_features(model_path, video_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    frames = get_frames(video_path)

    angles_per_frame = []
    coords_per_frame = []
    for frame in frames:
        resized_frame = np.expand_dims(resize_frame(frame, width, height), axis=0)
        reg_frame = regularize(resized_frame)

        in_index = input_details[0]['index']
        out_indexes = (output_details[0]['index'], output_details[1]['index'])

        # get outputs
        heatmaps, offsets = run_inference(interpreter, in_index, out_indexes, reg_frame)

        kps = parse_output(heatmaps, offsets, 0.3)

        features = get_features(kps)
        angles = [angle for angle, length in features]

        coords = [[x, y] for x, y, prob in kps]

        angles_per_frame.append(angles)
        coords_per_frame.append(coords)

    return angles_per_frame, coords_per_frame


if __name__ == '__main__':
    a, c = get_video_features(args.model_path, args.video_path)
    print(a)
    print(c)
