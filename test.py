import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from transformations import *
import cv2
from PIL import Image
import numpy as np
import math
from VideoProcessor import VideoProcessor
from ImageProcessor import ImageProcessor
from Model import Model


def app():

    file_path = 'media/figure.mp4'
    video_processor = VideoProcessor()
    video_processor.read(file_path)

    frames = video_processor.get_frames()
    num_frames = len(frames)

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

    gluPerspective(45, (display_width / display_height), 0.1, 50.0)
    glTranslate(0.0, 0.0, -30.0)

    image_processor = ImageProcessor()
    model = Model()
    for i, label in enumerate(point_labels):
        coord = (int(point_coords[i][0] / scale), int(point_coords[i][1] / scale))
        image_processor.add_point(coord, label)

    for frame_index, frame in enumerate(frames):
        image = Image.fromarray(frame)
        image_processor.set_image(image)
        image_processor.normalize()
        image_processor.segmentate()
        # image_processor.smooth()
        silhouette = image_processor.extract_boundary()
        if frame_index > 0:
            # Perform feature matching and camera pose estimation
            points1, points2, transformation_matrix = estimate_camera_pose(prev_frame, frame)

            # 3. Convert 2D Points to 3D
            points2d = np.float32(silhouette.reshape(-1, 2))  # Reshape silhouette to 2D points
            points3d = cv2.perspectiveTransform(np.array([points2d]), np.linalg.inv(transformation_matrix))[0]

            # 4. Aggregate 3D Points
            if frame_index == 1:
                point_cloud = points3d
            else:
                point_cloud = np.concatenate((point_cloud, points3d), axis=0)
        # Save current frame for the next iteration
        prev_frame = frame
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

    # Iterate over frames
    for frame_index in range(num_frames):
        # 1. Extract Silhouette
        silhouette = extract_silhouette(frame)  # Implement your silhouette extraction function

        # 2. Camera Pose Estimation
        if frame_index > 0:
            # Perform feature matching and camera pose estimation
            points1, points2, transformation_matrix = estimate_camera_pose(prev_frame, frame)

            # 3. Convert 2D Points to 3D
            points2d = np.float32(silhouette.reshape(-1, 2))  # Reshape silhouette to 2D points
            points3d = cv2.perspectiveTransform(np.array([points2d]), np.linalg.inv(transformation_matrix))[0]

            # 4. Aggregate 3D Points
            if frame_index == 1:
                point_cloud = points3d
            else:
                point_cloud = np.concatenate((point_cloud, points3d), axis=0)

        # Save current frame for the next iteration
        prev_frame = frame

    # The 'point_cloud' variable now contains the aggregated 3D points representing the object.

def extract_keypoints_and_descriptors(image):
    # Use a feature detection method (e.g., ORB)
    orb = cv2.ORB_create()

    # Find keypoints and descriptors
    keypoints, descriptors = orb.detectAndCompute(image, None)

    return keypoints, descriptors


def match_keypoints(descriptors1, descriptors2):
    # Use a feature matching method (e.g., Brute Force Matcher)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(descriptors1, descriptors2)

    # Sort matches based on distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Choose top N matches (you may adjust N based on your needs)
    N = 10
    good_matches = matches[:N]

    return good_matches


def estimate_camera_pose(image1, image2):
    # Extract keypoints and descriptors from both images
    keypoints1, descriptors1 = extract_keypoints_and_descriptors(image1)
    keypoints2, descriptors2 = extract_keypoints_and_descriptors(image2)

    # Match keypoints between the two images
    good_matches = match_keypoints(descriptors1, descriptors2)

    # Extract matched keypoints
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Use the PnP algorithm to estimate camera pose
    _, rotation_vector, translation_vector, _ = cv2.solvePnPRansac(points1, points2, camera_matrix, dist_coeffs)

    # Create the transformation matrix (4x4)
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    transformation_matrix = np.column_stack((rotation_matrix, translation_vector))
    transformation_matrix = np.vstack((transformation_matrix, [0, 0, 0, 1]))

    return points1, points2, transformation_matrix


# # Example usage
# image1 = cv2.imread('frame1.jpg')
# image2 = cv2.imread('frame2.jpg')
#
# camera_matrix = np.array([[focal_length_x, 0, principal_point_x],
#                           [0, focal_length_y, principal_point_y],
#                           [0, 0, 1]])
#
# dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
#
# points1, points2, transformation_matrix = estimate_camera_pose(image1, image2)
#
# # Use 'transformation_matrix' for further processing, such as transforming 2D points to 3D.
