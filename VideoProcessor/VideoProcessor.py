import json
import math

import cv2
import numpy as np
import os


class VideoProcessor:
    def __init__(self):
        self.filepath = "media/video.mp4"
        self.frames = np.array([])
        self.angle_data = []
        self.sync_frames_idx = []
        self.sync_angle_data_idx = []
        self.cnt = 0

    def set_filepath(self, filepath):
        self.filepath = filepath

    # def set_angle_data(self, angle_data):
    #     self.angle_data = angle_data

    def set_angle_data(self, angle_data):
        with open('media/angle_data.json', 'r') as json_file:
            self.angle_data = json.load(json_file)

    def get_frames(self):
        return self.frames

    def get_angle_data(self):
        return self.angle_data

    def process(self):
        video = cv2.VideoCapture(self.filepath)

        if not video.isOpened():
            print("Could not open :", self.filepath)
            exit(1)

        length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video.get(cv2.CAP_PROP_FPS)

        print("length :", length)
        print("width :", width)
        print("height :", height)
        print("fps :", fps)

        try:
            if not os.path.exists(self.filepath[:-4]):
                os.makedirs(self.filepath[:-4])
        except OSError:
            print("Error : Cannot create directory. " + self.filepath[:-4])

        frames = []

        for i in range(length):
            ret, frame = video.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                frames.append(frame)
                # image_path = f"media/frames/frame{i}.png"
                # cv2.imwrite(image_path, frame)

        self.frames = np.array(frames)

        video.release()

    def read(self, filepath, angle_data, angle_interval):
        self.set_filepath(filepath)
        self.set_angle_data(angle_data)
        self.process()
        self.synchronize_data(angle_interval)
        print(f"### Frame extraction done - {filepath} ###")
        print(f"Frames shape: {self.frames.shape}")
        print(f"Angle data length: {len(self.angle_data)}")

    def synchronize_data(self, angle_interval):
        print("### synchronize data ###")

        frames_length = len(self.frames)
        angle_data_length = len(self.angle_data)
        timestamp_max = self.angle_data[-1]['timestamp']

        desired_angle = 0.0
        pre_angle_data = self.angle_data[0]
        for angle_data_idx, angle_data in enumerate(self.angle_data):
            if angle_data['angleY'] <= desired_angle:
                # target_idx_ratio = angle_data_idx / (angle_data_length - 1)
                if angle_data['angleY'] == desired_angle:
                    target_idx_ratio = angle_data['timestamp'] / timestamp_max
                else:
                    cur_dist = abs(desired_angle - angle_data['angleY'])
                    pre_dist = abs(desired_angle - pre_angle_data['angleY'])
                    time_interval = angle_data['timestamp'] - pre_angle_data['timestamp']
                    target_timestamp = pre_angle_data['timestamp'] + time_interval * (pre_dist / (pre_dist + cur_dist))
                    target_idx_ratio = target_timestamp / timestamp_max
                    print(f"--------({angle_data_idx}/{angle_data_length})--------")
                    print(f"desired angle : {desired_angle}")
                    print(f"pre angle : {pre_angle_data['angleY']} ({pre_angle_data['timestamp']})")
                    print(f"post angle : {angle_data['angleY']} ({angle_data['timestamp']})")
                    print(f"interpolated : {desired_angle} ({target_timestamp})")
                    print(f"ratio : {target_idx_ratio}")
                frame_idx = round(frames_length * target_idx_ratio)
                print(f"result frame idx : {frame_idx}")
                if frame_idx == frames_length:
                    frame_idx -= 1
                self.sync_angle_data_idx.append(angle_data_idx)
                self.sync_frames_idx.append(frame_idx)
                desired_angle -= angle_interval
                pre_angle_data = angle_data

    def get_rotate_matrix(self, idx: int):
        angle_data = self.angle_data[idx]
        angleX = 180 * (math.pi / 180)
        # angleY = -angle_data['angleY']
        angleY = -(360 / 40) * (math.pi / 180) * self.cnt
        angleZ = 0

        self.cnt += 1

        rx = np.array([[1, 0, 0],
                       [0, np.cos(angleX), -np.sin(angleX)],
                       [0, np.sin(angleX), np.cos(angleX)]])

        ry = np.array([[np.cos(angleY), 0, np.sin(angleY)],
                       [0, 1, 0],
                       [-np.sin(angleY), 0, np.cos(angleY)]])

        rz = np.array([[np.cos(angleZ), -np.sin(angleZ), 0],
                       [np.sin(angleZ), np.cos(angleZ), 0],
                       [0, 0, 1]])

        rotation_matrix = np.dot(np.dot(rx, ry), rz)
        rotation_matrix = np.pad(rotation_matrix, ((0, 1), (0, 1)), mode='constant', constant_values=0)
        rotation_matrix[3, 3] = 1

        return rotation_matrix

    def normalize_2d_boundary(self, vertices, idx):
        angle_data = self.angle_data[idx]
        angle = angle_data['angleX']
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)
        rotation_matrix = np.array([[cos_theta, sin_theta],
                                    [-sin_theta, cos_theta]])
        vertices = np.dot(vertices, rotation_matrix)

        # angle = angle_data['angleZ']
        angle = 0
        cos_theta = np.cos(angle)
        shear_matrix = np.array([[1, 0],
                                 [0, (1 / cos_theta)]])
        return np.dot(vertices, shear_matrix)
