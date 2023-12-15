import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from PIL import Image
import numpy as np
import math
from VideoProcessor import VideoProcessor
from ImageProcessor import ImageProcessor
from Model import Model

PI = math.pi


def draw_model_point(model):
    glPointSize(1.0)
    glBegin(GL_POINTS)
    for vertex in model.vertices:
        glColor3fv((0.8, 0.8, 0.8))
        glVertex3fv(vertex[:3])
        print(vertex)
    glEnd()


def main(angle_data):

    # PROCESS VIDEO
    print("### processing video ###")

    file_path = 'media/video.mp4'
    # file_path = 'media/figure.mp4'
    video_processor = VideoProcessor()
    video_processor.read(file_path, angle_data, 2 * PI / 40)

    frames = video_processor.get_frames()

    # POINT PROMPT INTERFACE
    MAX_SIZE = 600

    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    position = None

    image = Image.fromarray(frames[0], mode='RGB')
    width, height = image.size
    scale = MAX_SIZE / max(width, height)
    new_width = int(width * scale)
    new_height = int(height * scale)

    point_coords = []
    point_labels = []

    pygame.init()

    screen = pygame.display.set_mode((new_width, new_height))

    pygame.display.set_caption('SAMpler interface')

    done = False
    clock = pygame.time.Clock()

    print("### interface screen init ###")
    print(f"screen size : {pygame.display.get_window_size()}")

    while not done:

        clock.tick()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.MOUSEBUTTONUP:
                point_coords.append(event.pos)
                # left click - add mask
                if event.button == 1:
                    point_labels.append(1)
                # right click - remove area
                elif event.button == 3:
                    point_labels.append(0)

        # UPDATE THE STATE

        # DRAW THE SCENE
        first_frame = np.transpose(frames[0], (1, 0, 2))
        first_frame = pygame.surfarray.make_surface(first_frame)
        first_frame = pygame.transform.scale(first_frame, (new_width, new_height))
        screen.blit(first_frame, (0, 0))
        for i, label in enumerate(point_labels):
            coord = point_coords[i]
            if label:
                pygame.draw.circle(screen, WHITE, coord, 10)
            else:
                pygame.draw.circle(screen, BLACK, coord, 10)

        pygame.display.flip()

    pygame.quit()

    # PROCESS FRAMES
    # scaled_x, scaled_y = position                       # prompt coord is valid on scaled interface screen
    # position = (scaled_x / scale, scaled_y / scale)     # inverse scale for applying on original image size

    pygame.init()

    display_width = 800
    display_height = 600
    display_shape = (display_width, display_height)

    pygame.display.set_caption('SAMpler interface')
    clock = pygame.time.Clock()
    display = pygame.display.set_mode(display_shape, DOUBLEBUF | OPENGL)

    gluPerspective(45, (display_width/display_height), 0.1, 50.0)
    glTranslate(0.0, 0.0, -30.0)

    image_processor = ImageProcessor()
    model = Model()
    for i, label in enumerate(point_labels):
        coord = (int(point_coords[i][0] / scale), int(point_coords[i][1] / scale))
        image_processor.add_point(coord, label)

    boundary_len_sum = 0
    average_boundary_len = 0
    boundary_cnt = 0
    bbox_left_avg = 0
    bbox_top_avg = 0
    bbox_width_avg = 0
    bbox_height_avg = 0
    bbox_left_sum = 0
    bbox_top_sum = 0
    bbox_width_sum = 0
    bbox_height_sum = 0
    bbox_cnt = 0
    init_bbox = {}

    for i in range(len(video_processor.sync_frames_idx)):
        frame_idx = video_processor.sync_frames_idx[i]
        angle_data_idx = video_processor.sync_angle_data_idx[i]
        frame = frames[frame_idx]
        angle_data = video_processor.angle_data[angle_data_idx]

        print(f'frames : {frame_idx} / {len(frames)}')
        print(f'angle data : {angle_data_idx} / {len(video_processor.angle_data)}')

        image = Image.fromarray(frame)
        image_processor.set_image(image)
        image_processor.normalize()
        if i != 0:
            image_processor.update_prompts()
        image_processor.segmentate()
        image_processor.smooth()
        image_processor.update_object()

        print(f'object_center : {image_processor.object_center}')
        print(f'object_bbox : {image_processor.object_bbox}')
        print(f'angle_data: ({angle_data["angleX"]}, {angle_data["angleY"]}, {angle_data["angleZ"]})')

        boundary = image_processor.extract_boundary()
        normalized_boundary = video_processor.normalize_2d_boundary(boundary, angle_data_idx)
        # model.add_vertices(normalized_boundary, video_processor.get_rotate_matrix(angle_data_idx))
        boundary_len = len(normalized_boundary)
        bbox = image_processor.object_bbox
        bbox_left = bbox[0]
        bbox_top = bbox[1]
        bbox_width = bbox[2]
        bbox_height = bbox[3]
        if i == 0:
            init_bbox = bbox
            boundary_len_sum += boundary_len
            boundary_cnt += 1
            average_boundary_len = boundary_len
            bbox_cnt += 1
            bbox_left_avg = bbox_left
            bbox_top_avg = bbox_top
            bbox_width_avg = bbox_width
            bbox_height_avg = bbox_height
            bbox_left_sum += bbox_left
            bbox_top_sum += bbox_top
            bbox_width_sum += bbox_width
            bbox_height_sum += bbox_height
            model.add_vertices(normalized_boundary, video_processor.get_rotate_matrix(angle_data_idx))
        else:
            if abs(boundary_len / average_boundary_len) > 1.5:
                continue
            if abs(bbox_left - bbox_left_avg) > 150 or abs(bbox_top - bbox_top_avg) > 150:
                continue
            else:
                if bbox_width / bbox_width_avg > 1.5 or bbox_height / bbox_height_avg > 1.5:
                    target_width = bbox_width * 0.0 + bbox_width_avg * 1.0
                    target_height = bbox_height * 0.0 + bbox_height_avg * 1.0
                    ratio_w = target_width / bbox_width
                    ratio_h = target_height / bbox_height
                    M = np.array([[ratio_h, 0],
                                  [0, ratio_w]])
                    normalized_boundary = np.dot(normalized_boundary, M.T)
                    bbox_cnt += 1
                    bbox_left_sum += bbox_left
                    bbox_height_sum += bbox_top_sum
                    bbox_width_sum += bbox_width
                    bbox_height_sum += bbox_height
                    bbox_left_avg = bbox_left_sum / bbox_cnt
                    bbox_top_avg = bbox_top_sum / bbox_cnt
                    bbox_width_avg = bbox_width_sum / bbox_cnt
                    bbox_height_avg = bbox_height_sum / bbox_cnt

                boundary_len_sum += boundary_len
                boundary_cnt += 1
                average_boundary_len = boundary_len_sum / boundary_cnt
                model.add_vertices(normalized_boundary, video_processor.get_rotate_matrix(angle_data_idx))

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        model.draw()
        oc = image_processor.object_center
        glPointSize(5.0)
        glBegin(GL_POINTS)
        glColor3fv((0., 0., 1.0))
        glVertex3fv((0, 0, 0))
        for i, point in enumerate(image_processor.point_coords):
            if image_processor.point_labels[i]:
                glColor3fv((0., 1.0, 0.))
            else:
                glColor3fv((1.0, 0., 0.))
            dx = -image_processor.object_center[0]
            dy = -image_processor.object_center[1]
            glVertex3fv(((point[0] + dx) / 50, (point[1] + dy) / 50, 0))
        glEnd()
        pygame.display.flip()
        # pygame.time.wait(10)
        # model.reset()

    pygame.quit()

    # image_processor = ImageProcessor()
    # model = Model()
    # for i, label in enumerate(point_labels):
    #     coord = (int(point_coords[i][0] / scale), int(point_coords[i][1] / scale))
    #     image_processor.add_point(coord, label)
    #
    # # boundary_len_sum = 0
    # # average_boundary_len = 0
    # # boundary_cnt = 0
    #
    # bbox_width_avg = 0
    # bbox_height_avg = 0
    # bbox_width_sum = 0
    # bbox_height_sum = 0
    # bbox_cnt = 0
    #
    # for i in range(len(video_processor.sync_frames_idx)):
    #     frame_idx = video_processor.sync_frames_idx[i]
    #     angle_data_idx = video_processor.sync_angle_data_idx[i]
    #     frame = frames[frame_idx]
    #     angle_data = video_processor.angle_data[angle_data_idx]
    #
    #     print(f'frames : {frame_idx} / {len(frames)}')
    #     print(f'angle data : {angle_data_idx} / {len(video_processor.angle_data)}')
    #
    #     image = Image.fromarray(frame)
    #     image_processor.set_image(image)
    #     image_processor.normalize()
    #     if i != 0:
    #         image_processor.update_prompts()
    #     image_processor.segmentate()
    #     image_processor.smooth()
    #     image_processor.update_object()
    #
    #     print(f'object_center : {image_processor.object_center}')
    #     print(f'object_bbox : {image_processor.object_bbox}')
    #     print(f'angle_data: ({angle_data["angleX"]}, {angle_data["angleY"]}, {angle_data["angleZ"]})')
    #
    #     boundary = image_processor.extract_boundary()
    #     normalized_boundary = video_processor.normalize_2d_boundary(boundary, i)
    #     # boundary_len = len(normalized_boundary)
    #     # if i == 0:
    #     #     boundary_len_sum += boundary_len
    #     #     boundary_cnt += 1
    #     #     average_boundary_len = boundary_len
    #     # else:
    #     #     if abs(boundary_len / average_boundary_len) > 2:
    #     #         continue
    #     #     else:
    #     #         boundary_len_sum += boundary_len
    #     #         boundary_cnt += 1
    #     #         average_boundary_len = boundary_len_sum / boundary_cnt
    #     #         model.add_vertices(normalized_boundary, video_processor.get_rotate_matrix(angle_data_idx))
    #     bbox = image_processor.object_bbox
    #     bbox_width = bbox[2]
    #     bbox_height = bbox[3]
    #     if i == 0:
    #         bbox_cnt += 1
    #         bbox_width_avg = bbox_width
    #         bbox_height_avg = bbox_height
    #         bbox_width_sum += bbox_width
    #         bbox_height_sum += bbox_height
    #         model.add_vertices(normalized_boundary, video_processor.get_rotate_matrix(angle_data_idx))
    #     else:
    #         if bbox_width / bbox_width_avg > 1.5 or bbox_height / bbox_height_avg > 1.5:
    #             continue
    #         else:
    #             bbox_cnt += 1
    #             bbox_width_sum += bbox_width
    #             bbox_height_sum += bbox_height
    #             bbox_width_avg = bbox_width_sum / bbox_cnt
    #             bbox_height_avg = bbox_height_sum / bbox_cnt
    #             model.add_vertices(normalized_boundary, video_processor.get_rotate_matrix(angle_data_idx))
    #     print(f'object_bbox_avg : {bbox_width_avg} {bbox_height_avg}')

    pygame.init()

    display_width = 800
    display_height = 600
    display_shape = (display_width, display_height)

    pygame.display.set_caption('SAMpler interface')
    clock = pygame.time.Clock()
    display = pygame.display.set_mode(display_shape, DOUBLEBUF | OPENGL)

    gluPerspective(45, (display_width/display_height), 0.1, 50.0)
    glTranslate(0.0, 0.0, -30.0)

    done = False

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                pygame.quit()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glRotate(15, 0.0, 1.0, 0.0)
        # draw_model_point(model)
        model.draw()
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main(0)
