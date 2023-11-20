import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from transformations import *
from PIL import Image
import numpy as np
import math
from MobileSAM.app.app import segment_everything, extract_edge

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


def set_model(filepath):
    global vertices

    image = Image.open(filepath)
    annotations = segment_everything(image)
    longest = 0
    for i, annotation in enumerate(annotations):
        edges = extract_edge(annotation)
        if len(edges) > longest:
            longest = len(edges)
            vertices = np.array(edges)
            vertices = np.c_[vertices, np.zeros(vertices.shape[0])]
            vertices = np.c_[vertices, np.ones(vertices.shape[0])]
            vertices = vertices / 100
    print(vertices)


def draw_cube():
    set_cube()
    draw_model()


def draw_model():
    global vertices
    global triangles

    glBegin(GL_TRIANGLES)
    for triangle in triangles:
        for vertex in triangle:
            glColor3fv((0.8, 0.8, 0.8))
            glVertex3fv(vertices[vertex][:3])
    glEnd()


def draw_model_point():
    global vertices

    glPointSize(5.0)
    glBegin(GL_POINTS)
    for vertex in vertices:
        glColor3fv((0.8, 0.8, 0.8))
        glVertex3fv(vertex[:3])
        print(vertex)
    glEnd()


def rotate(angle, direction, point=None):
    global vertices

    M = rotation_matrix(angle, direction, point)
    vertices = np.dot(M, vertices.T).T


def myOpenGL():
    # pygame.init()
    #
    # display_width = 800
    # display_height = 600
    # display_shape = (display_width, display_height)
    #
    # pygame.display.set_caption('SAMpler interface')
    # clock = pygame.time.Clock()
    # # display = pygame.display.set_mode(display_shape, DOUBLEBUF | OPENGL)
    # display = pygame.display.set_mode(display_shape)
    #
    # # gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
    # # glTranslate(0.0, 0.0, -20.0)
    #
    # # draw_cube()
    # # filepath = './figure/frame63.jpg'
    # # set_model(filepath)
    # first_frame = pygame.image.load('./figure/sample.png')
    #
    #
    # while True:
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             pygame.quit()
    #
    #     # glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    #     # # rotate(PI / 192, (0, 1, 0), (0, 0, 0))
    #     # # draw_model()
    #     # draw_model_point()
    #     # pygame.display.flip()
    #     # pygame.time.wait(10)
    #
    #     display.fill((255, 255, 255))
    #     display.blit(first_frame, (0, 0))
    #     pygame.display.update()
    #     clock.tick(60)
    width = 800
    height = 600
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    position = None

    image = Image.open('./figure/frame63.jpg')
    w, h = image.size
    scale = 600 / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    image = image.resize((new_w, new_h))

    pygame.init()

    screen = pygame.display.set_mode((new_w, new_h))

    pygame.display.set_caption('SAMpler interface')

    done = False
    clock = pygame.time.Clock()

    while not done:

        clock.tick()

        for event in pygame.event.get():  # User did something
            if event.type == pygame.QUIT:  # If user clicked close
                done = True  # Flag that we are done so we exit this loop
            elif event.type == pygame.MOUSEBUTTONUP:
                position = event.pos
                mouses = pygame.mouse.get_pressed()

        # UPDATE THE STATE

        milliseconds = clock.get_time()
        keys = pygame.key.get_pressed()

        # if mouses[pygame.MOUSEBUTTONUP]:
        #     position = pygame.mouse.get_pos()
        #     pygame.draw.circle(screen, BLACK, position, 1)


        # DRAW THE SCENE
        first_frame = pygame.image.load('./figure/frame63.jpg')
        first_frame = pygame.transform.scale(first_frame, (new_w, new_h))
        screen.blit(first_frame, (0, 0))
        if position:
            pygame.draw.circle(screen, BLACK, position, 20)
        pygame.display.flip()

    pygame.quit()


myOpenGL()
