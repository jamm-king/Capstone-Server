import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from transformations import *
from PIL import Image
import numpy as np
import math
from VideoProcessor import VideoProcessor
from ImageProcessor import ImageProcessor
from Model import Model

PI = math.pi

colors = [
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1)
]
vertices = np.array([])
triangles = np.array([])


def set_cube():
    global vertices
    global triangles

    vertices = np.array([
        (1, 1, 1, 1),
        (1, 1, -1, 1),
        (1, -1, 1, 1),
        (-1, 1, 1, 1),
        (1, -1, -1, 1),
        (-1, 1, -1, 1),
        (-1, -1, 1, 1),
        (-1, -1, -1, 1)
    ])
    triangles = np.array([
        (0, 2, 1),
        (1, 2, 4),
        (0, 1, 3),
        (1, 5, 3),
        (0, 3, 2),
        (2, 3, 6),
        (3, 5, 6),
        (5, 7, 6),
        (2, 6, 7),
        (2, 7, 4),
        (1, 4, 7),
        (1, 7, 5)
    ])


# def set_model(filepath):
#     global vertices
#
#     image = Image.open(filepath)
#     annotations = segment_everything(image)
#     longest = 0
#     for i, annotation in enumerate(annotations):
#         edges = extract_edge(annotation)
#         if len(edges) > longest:
#             longest = len(edges)
#             vertices = np.array(edges)
#             vertices = np.c_[vertices, np.zeros(vertices.shape[0])]
#             vertices = np.c_[vertices, np.ones(vertices.shape[0])]
#             vertices = vertices / 100
#     print(vertices)


def draw_cube():
    set_cube()
    draw_model()


def draw_model(model):
    global vertices
    global triangles

    glBegin(GL_TRIANGLES)
    for triangle in triangles:
        for vertex in triangle:
            glColor3fv((0.8, 0.8, 0.8))
            glVertex3fv(vertices[vertex][:3])
    glEnd()


def draw_model_point(model):
    global vertices

    glPointSize(1.0)
    glBegin(GL_POINTS)
    for vertex in model.vertices:
        glColor3fv((0.8, 0.8, 0.8))
        glVertex3fv(vertex[:3])
        print(vertex)
    glEnd()


def rotate(angle, direction, point=None):
    global vertices

    M = rotation_matrix(angle, direction, point)
    vertices = np.dot(M, vertices.T).T


def main():

    # PROCESS VIDEO
    print("### processing video ###")

    file_path = 'media/figure.mp4'
    video_processor = VideoProcessor()
    video_processor.read(file_path)

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
    for i, frame in enumerate(frames):
        image = Image.fromarray(frame)
        image_processor.set_image(image)
        image_processor.normalize()
        image_processor.segmentate()
        # image_processor.smooth()
        boundary = image_processor.extract_boundary()
        # vertices = []
        # for j, row in enumerate(image_processor.annotation[0]):
        #     for k, pix in enumerate(row):
        #         if pix:
        #             vertices.append((k, j))
        # model.add_vertices(np.array(vertices))
        model.add_vertices(boundary)
        image_processor.print_info()
        image_processor.update_object()
        image_processor.update_prompts()

        print(image_processor.object_center)
        print(image_processor.object_bbox)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        model.draw()
        oc = image_processor.object_center
        glPointSize(5.0)
        glBegin(GL_POINTS)
        glColor3fv((0., 0., 1.0))
        glVertex3fv((oc[0] / 50, oc[1] / 50, 0))
        for i, point in enumerate(image_processor.point_coords):
            if image_processor.point_labels[i]:
                glColor3fv((0., 1.0, 0.))
            else:
                glColor3fv((1.0, 0., 0.))
            glVertex3fv((point[0] / 50, point[1] / 50, 0))
        glEnd()
        pygame.display.flip()
        pygame.time.wait(10)
        model.reset()

    pygame.quit()


    # pygame.init()
    #
    # display_width = 800
    # display_height = 600
    # display_shape = (display_width, display_height)
    #
    # pygame.display.set_caption('SAMpler interface')
    # clock = pygame.time.Clock()
    # display = pygame.display.set_mode(display_shape, DOUBLEBUF | OPENGL)
    #
    # gluPerspective(45, (display_width/display_height), 0.1, 50.0)
    # glTranslate(0.0, 0.0, -30.0)
    #
    # done = False
    #
    # while not done:
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             done = True
    #             pygame.quit()
    #
    #     glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    #     # rotate(PI / 192, (0, 1, 0), (0, 0, 0))
    #     draw_model_point(model)
    #     pygame.display.flip()
    #     pygame.time.wait(10)
    #
    # pygame.quit()


main()
