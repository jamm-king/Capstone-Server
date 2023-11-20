import cv2
import numpy as np
from MobileSAM.app import MobileSAM


class ImageProcessor:
    MAX_SIZE = 1024

    def __init__(self):
        self.image = None
        self.annotation = None
        self.MobileSAM = MobileSAM()
        self.point_coords = []
        self.point_labels = []
        self.object_center = (0, 0)
        self.object_bbox = (0, 0, 0, 0)

    # def update(self, new_image):

    def set_image(self, image):
        self.image = image

    def get_annotation(self):
        return self.annotation

    def get_edge(self):
        return self.edge

    def add_point(self, coord, label):
        self.point_coords.append(coord)
        self.point_labels.append(label)

    def reset_point(self):
        self.point_coords = []
        self.point_labels = []

    def segmentate(self):
        self.annotation = self.MobileSAM.segment_with_points(self.image, self.point_coords, self.point_labels)

    # def extract_edge(self):
    # # segmentation = self.annotation['segmentation']
    # # point_coords = (int(self.annotation['point_coords'][0][0]), int(self.annotation['point_coords'][0][1]))
    # print("### extract edge process start")
    # segmentation = self.annotation[0]
    # point_coords = (self.point_coords[0][0], self.point_coords[0][1])
    # shape = segmentation.shape
    # edges = []
    # print(f"start point : ({point_coords[0]}, {point_coords[1]})")
    # print(f"segmentation shape : {shape}")
    # print(shape[0], shape[1])
    #
    # checked = np.zeros(shape)
    # queue = [point_coords]
    # checked[point_coords[1]][point_coords[0]] = 1
    # dx = [1, 0, -1, 0]
    # dy = [0, 1, 0, -1]
    # while len(queue) != 0:
    #     y, x = queue.pop()
    #     print(f"--- ({y}, {x}) ---")
    #     appended = False
    #     for i in range(4):
    #         nx = x + dx[i]
    #         ny = y + dy[i]
    #         if nx >= shape[1] or nx < 0 or ny >= shape[0] or ny < 0:
    #             continue
    #         if checked[ny][nx] == 1:
    #             continue
    #         if not segmentation[ny][nx]:
    #             if not appended:
    #                 edges.append((x, y))
    #                 appended = True
    #             continue
    #         queue.append((nx, ny))
    #         checked[ny][nx] = 1
    #
    # return np.array(edges)

    def extract_boundary(self):
        segmentation = self.annotation[0].T
        boundary_array = segmentation.copy()
        kernel = np.array([[1, 1, 1],
                           [1, 1, 1],
                           [1, 1, 1]], dtype=np.uint8)
        boundary_array = cv2.erode(boundary_array.astype(np.uint8), kernel, iterations=1)
        boundary_array = segmentation ^ boundary_array
        boundary_coords = np.argwhere(boundary_array)

        return boundary_coords

    def normalize(self):
        # Calculate ratio for MAX_SIZE
        width, height = self.image.size
        scale = self.MAX_SIZE / max(width, height)
        new_point_coords = []

        # Calculate new length, coords
        new_width = int(width * scale)
        new_height = int(height * scale)
        for point in self.point_coords:
            new_point = (int(point[0] * scale), int(point[1] * scale))
            new_point_coords.append(new_point)

        # Update for scaled constants
        self.image = self.image.resize((new_width, new_height))
        self.point_coords = new_point_coords

    def update_object(self):
        object_mask = self.annotation[0].astype(np.uint8) * 255
        M = cv2.moments(object_mask)
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])
        object_center = (center_x, center_y)
        bbox = cv2.boundingRect(object_mask)
        self.object_center = object_center
        self.object_bbox = bbox

    def update_prompts(self):
        object_mask = self.annotation[0].astype(np.uint8) * 255
        M = cv2.moments(object_mask)
        current_cx = self.object_center[0]
        current_cy = self.object_center[1]
        current_bbox = self.object_bbox
        next_cx = int(M["m10"] / M["m00"])
        next_cy = int(M["m01"] / M["m00"])
        next_center = (next_cx, next_cy)
        next_bbox = cv2.boundingRect(object_mask)

        dx = next_cx - current_cx
        dy = next_cy - current_cy
        sx = next_bbox[2] / current_bbox[2]
        sy = next_bbox[3] / current_bbox[3]

        for i, point in enumerate(self.point_coords):
            x = point[0]
            y = point[1]
            current_dist_x = current_cx - x
            current_dist_y = current_cy - y
            next_dist_x = current_dist_x * sx
            next_dist_y = current_dist_y * sy
            next_x = next_cx + next_dist_x
            next_y = next_cy + next_dist_y
            self.point_coords[i] = (next_x, next_y)

    def smooth(self, filter_size=5, threshold=0.9):
        segmentation = self.annotation[0]
        height, width = segmentation.shape
        padding = filter_size // 2
        for i in range(padding, height - padding):
            for j in range(padding, width - padding):
                if not segmentation[i][j]:
                    cnt = 0
                    for k in range(i - padding, i + padding + 1):
                        for l in range(j - padding, j + padding + 1):
                            if segmentation[k][l]:
                                cnt += 1
                    if (cnt / filter_size ** 2) >= threshold:
                        for k in range(i - padding, i + padding + 1):
                            for l in range(j - padding, j + padding + 1):
                                if not segmentation[k][l]:
                                    segmentation[k][l] = True

    def print_info(self):
        print("### image processor info ###")
        print(f"image : {type(self.image)} {self.image.size}")
        print(f"annotation : {type(self.annotation)} {self.annotation.shape}")
        print(self.annotation)
        print(f"point coords : {self.point_coords}")
        print(f"point labels : {self.point_labels}")
