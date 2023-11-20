import cv2
import numpy as np
import os


class VideoProcessor:
    def __init__(self):
        self.filepath = ""
        self.frames = np.array([])

    def set_filepath(self, filepath):
        self.filepath = filepath

    def get_frames(self):
        return self.frames

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

    def read(self, filepath):
        self.set_filepath(filepath)
        self.process()
        print(f"### Frame extraction done - {filepath} ###")
        print(f"Frames shape : {self.frames.shape}")


# filepath = '../media/figure.mp4'
# videoProcessor = VideoProcessor()
# videoProcessor.read(filepath)
# print(videoProcessor.frames.shape)
