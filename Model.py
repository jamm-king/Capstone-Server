from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np


class Model:

    def __init__(self):
        self.vertices = []
        self.planes = []
        self.vertex_num = 0

    def add_vertex(self, vertex):
        vertex = np.array([vertex[0], vertex[1], 0, 1]) / 50
        self.vertices.append(vertex)
        self.vertex_num += 1

    def add_vertices(self, vertices):
        vertices = vertices / 50
        vertices = np.hstack((vertices, np.zeros((vertices.shape[0], 2)), np.ones((vertices.shape[0], 1))))
        for vertex in vertices:
            self.vertices.append(vertex)
        self.vertex_num += len(vertices)

    def draw(self):
        glPointSize(1.0)
        glBegin(GL_POINTS)
        for vertex in self.vertices:
            glColor3fv((0.8, 0.8, 0.8))
            glVertex3fv(vertex[:3])
        glEnd()

    def triangular(self):
        """
        triangular implementation
        """

    def reset(self):
        self.vertices = []
        self.planes = []
