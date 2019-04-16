"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import cv2
import logging

logger = logging.Logger(__name__)
import numpy as np
from scipy.spatial.distance import pdist
from scipy.interpolate import interp1d

from hips_detectors.interest_point import InterestPoint
from hips_detectors.maze_grid_code import MazeGridCode

# because np.sqrt is slower when we do it on small arrays
def reversedEnumerate(l):
    return zip(range(len(l) - 1, -1, -1), reversed(l))


from math import sqrt, atan, pi, hypot

sqrt_2 = sqrt(2)


######################################################################


# SIFT initialization
# upper_left = cv2.imread(r'C:\work\maze-nc-int-1.jpg', cv2.IMREAD_GRAYSCALE)
#
# sift = cv2.xfeatures2d.SIFT_create()
# upper_left_key_points, upper_left_descriptors = sift.detectAndCompute(upper_left, None)
#
# upper_left = cv2.drawKeypoints(upper_left, upper_left_key_points, upper_left)
# # cv2.imshow("Reference", upper_left)
#
# index_params = dict(algorithm=0, trees=5)
# search_params = dict()
# flann = cv2.FlannBasedMatcher(index_params,search_params)


# def homography(gray_frame):
#     gray_frame_key_points, gray_frame_descriptors = sift.detectAndCompute(gray_frame, None)
#     # gray_frame = cv2.drawKeypoints(gray_frame, gray_frame_key_points, gray_frame)
#     # cv2.imshow("Test", gray_frame)
#
#     matches = flann.knnMatch(upper_left_descriptors, gray_frame_descriptors, k=2)
#     good_matches = []
#     for m, n in matches:
#         if m.distance < n.distance * 0.8:
#             good_matches.append(m)
#
#     # im3 = cv2.drawMatches(upper_left, upper_left_key_points, gray_frame, gray_frame_key_points, good_matches, gray_frame)
#     # cv2.imshow("Matches", im3)
#
#     if len(good_matches) > 10:
#         query_points = np.float32([upper_left_key_points[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
#         train_points = np.float32([gray_frame_key_points[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
#
#         matrix, mask = cv2.findHomography(query_points, train_points, cv2.RANSAC, 5.0)
#         matches_mask = mask.ravel().tolist()
#         if matrix is not None:  # and matrix.any():
#             h, w, _ = upper_left.shape
#             pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
#             dst = cv2.perspectiveTransform(pts, matrix)
#
#             homography_blah = cv2.polylines(gray_frame, [np.int32(dst)], True, (255, 0, 0), 3)
#             cv2.imshow("Matches", homography_blah)
#         else:
#             cv2.imshow("Matches", gray_frame)
#     else:
#         cv2.imshow("Matches", gray_frame)


def distance(a, b):
    return sqrt(pow(b[0] - a[0], 2) + pow(b[1] - a[1], 2))


def slope_in_degrees(slope):
    return atan(slope) * 360.0 / (2.0 * pi)


class PartialLine:
    def __init__(self, x1, x2, y1, y2):
        self.start = (x1, y1)
        self.end = (x2, y2)
        self.slope = (y2 - y1) / (x2 - x1)
        self.line_length = distance(self.start, self.end)

    def is_same_slope(self, other_line, threshold_in_degrees=1):
        alpha_this = slope_in_degrees(self.slope)
        alpha_other = slope_in_degrees(other_line.slope)
        difference = abs(alpha_this - alpha_other)
        return difference <= threshold_in_degrees


def line_detection(gray_img, img):
    #  https://www.youtube.com/watch?v=KEYzUP7-kkU
    # edges = cv2.Canny(gray_img, 75, 150)
    # edges = cv2.Canny(gray_img, 50, 200, None, 3)
    # lines = cv2.HoughLinesP(gray_img, rho=0.1, theta=np.pi / 90, threshold=30, minLineLength=50, maxLineGap=30)
    # lines = cv2.HoughLinesP(edges, 2, np.pi / 90, 30)

    edges = cv2.Canny(gray_img, 100, 170, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 190)

    if lines is not None:
        # partial_lines = []
        # for line in lines:
        #     x1, y1, x2, y2 = line[0]
        #     # partial_line = PartialLine(x1, y1, x2, y2)
        #     # partial_lines.append(partial_line)
        #     cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3, cv2.LINE_AA)
        for rho, theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    cv2.imshow("Lines", img)


def corner_detection(gray_img, img):
    #  https://www.youtube.com/watch?v=ROjDoDqsnP8
    corners = cv2.goodFeaturesToTrack(gray_img, 1000, 0.1, 10)
    if corners is not None:
        corners = np.int0(corners)

        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
    cv2.imshow("Lines", img)


def opencv_testing(gray_img, img, frame):
    #harris corner detec + subpix
    pass


def line_detector(frame, img, gray_img):
    # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html
    # Thresholds a gray image into a black and white image.

    # source image, threshold value for pixels (below it goes black, above it goes white), color value given to pixels above the threshold (in this case, white)
    retval, threshold_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY_INV)
    # threshold_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    # invert threshold_img. Use when using cv2.THRESH_BINARY instead of cv2.THRESH_BINARY_INV
    # threshold_img = cv2.bitwise_not(threshold_img)

    corners = cv2.goodFeaturesToTrack(gray_img, 1000, 0.1, 10)
    if corners is not None:
        corners = np.int0(corners)

        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)

    # show results
    cv2.imshow("Lines", img)


# https://stackoverflow.com/questions/45531074/how-to-merge-lines-after-houghlinesp

# https://github.com/opencv/opencv/blob/1913482cf5d257d7da292bb2c15f22d0588d34dc/modules/calib3d/src/calibinit.cpp
# def find_maze_corners(frame, gray_img, pattern_size=(4,4), corners_list=[], flags=0):
#     is_found = False
#
#     MIN_DILATIONS = 0
#     MAX_DILATIONS = 7
#
#     # img must be 8-bit grayscale or color, assuming grayscale for now
#
#     pattern_width = pattern_size[0]
#     pattern_height = pattern_size[1]
#     if pattern_width <=2 or pattern_height <= 2:
#         print("ERROR: Both width and height of the pattern should be bigger than 2.")
#
#     out_corners = []
#     previous_square_size = 0
#     threshold_image = image_histogram_based_thresholding(gray_img)
#
#     # TODO: a fast check for a maze/chessboard
#     # https://github.com/opencv/opencv/blob/1913482cf5d257d7da292bb2c15f22d0588d34dc/modules/calib3d/src/calibinit.cpp#L523
#
#     detector = MazeDetector(pattern_size)
#
#     image_height, image_width = threshold_image.shape
#
#     for i in range(MIN_DILATIONS, MAX_DILATIONS):
#
#
#         # So we can find rectangles that go to the edge, we draw a white line around the image edge.
#         # Otherwise, find_contours will miss those clipped rectangle contours.
#         # The border color will be the image mean, because otherwise we risk screwing up filters like cv2.smooth
#         cv2.rectangle(threshold_image, (0,0), (image_width, image_height), (255,255,255), 3, cv2.LINE_8)
#
#         detector.reset()
#
#         detector.generate_quads(frame, threshold_image, flags)
#
#     # cv2.imshow("ChessBoardTest", threshold_image)

# Found at bottom of calibinit.cpp; findCirclesGrid(params)
def find_maze_corners(frame, gray_img, feature_detector, pattern_size=(4, 4), out_centers=[], flags=0):
    # feature_detector is an interface to swap freely between ORB, SIFT, SURF, etc.
    img = frame.img

    centers = []

    # One flag is either SYMMETRIC_GRID or ASYMMETRIC_GRID. We only need SYMMETRIC_GRID.

    # example feature_detector
    # orb = cv2.ORB()
    # orb.detect()

    # really slow on each frame. run on edges?
    # keypoints = feature_detector.detect(gray_img, None) # img, mask
    # points = [x.pt for x in keypoints]

    # good features to track much faster and finds better points, but need to adjust settings to account for lighting
    points = []
    corners = cv2.goodFeaturesToTrack(gray_img, 1000, 0.1, 10)
    if corners is not None:
        corners = np.int0(corners)
        for corner in corners:
            point = corner.ravel()
            points.append(point)

    for point in points:
        x, y = point
        x = round(x)
        y = round(y)
        cv2.circle(img, (x, y), 3, (0,255,0))

    # flag here for CALIB_CB_CLUSTERING; completely separate grid finder that returns, ignoring everything below

    ATTEMPTS = 2
    MIN_HOMOGRAPHY_POINTS = 4

    # for i in range(ATTEMPTS):
    #     out_centers.clear()
    #     box_finder = CirclesGridFinder(pattern_size, points)
    #     is_found = False
    #     try:
    #         is_found = box_finder.find_holes(frame)
    #     except:
    #         raise
    #
    #     if is_found:
    #         box_finder.get_holes(out_centers)


    cv2.imshow("ChessBoardTest", img)


# GoodFeaturesToTrack params
gftt_params = dict(maxCorners=500, qualityLevel=0.1, minDistance=10)
# maxCorners=1000, 500
# qualityLevel=0.1, 0.01
# minDistance=10
# blockSize=None


def custom_finder(frame, gray_img):
    img = frame.img

    corners = cv2.goodFeaturesToTrack(gray_img, **gftt_params)
    # Convert GFTTs to boxes and display
    points = [p[0] for p in corners]
    # grid_size = InterestPoint.calculate_grid_size(points)
    # half_grid = int(grid_size / 2)
    i_points = []
    for c in corners:
        pt = c[0]
        i_point = InterestPoint(pt)
        i_points.append(i_point)

    # for i_point in i_points:
    #     matching_i_points = MazeGridCode.test_find(gray_img, i_point, i_points, grid_size)
    #     if matching_i_points:
    #         print("**********************MATCH FOUND************************")
    #         cv2.imshow("Find QR", gray_img)

    MazeGridCode.scale_test_find(img, gray_img, i_points)

    # points = []
    # corners = cv2.goodFeaturesToTrack(gray_img, 1000, 0.1, 10)
    # if corners is not None:
    #     corners = np.int0(corners)
    #     for corner in corners:
    #         point = corner.ravel()
    #         points.append(point)
    #
    # for point in points:
    #     x, y = point
    #     x = round(x)
    #     y = round(y)
    #     cv2.circle(img, (x, y), 5, (0, 255, 0), -1)

    # threshold_image = image_histogram_based_thresholding(gray_img)
    #
    # cv2.imshow("Custom", threshold_image)


def image_histogram_based_thresholding(img):
    # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html
    # Using Otsu's Binarization to start.
    # If this fails, directly implement icvBinarizationHistogramBased() seen here: https://github.com/opencv/opencv/blob/1913482cf5d257d7da292bb2c15f22d0588d34dc/modules/calib3d/src/calibinit.cpp#L520
    # Source Code here: https://github.com/opencv/opencv/blob/1913482cf5d257d7da292bb2c15f22d0588d34dc/modules/calib3d/src/calibinit.cpp#L342

    blurred_img = cv2.GaussianBlur(img, (5,5), 0)
    ret, otsu_threshold_img = cv2.threshold(blurred_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)   # definition has a problem, especially when hand is added
    return otsu_threshold_img


# https://github.com/opencv/opencv/blob/1913482cf5d257d7da292bb2c15f22d0588d34dc/modules/calib3d/src/calibinit.cpp#L224
class MazeDetector:
    def __init__(self, pattern_size):
        self.pattern_size = pattern_size
        self.all_quads = []
        self.all_corners = []
        self.all_quads_count = 0

    def reset(self):
        self.all_quads = []
        self.all_corners = []
        self.all_quads_count = 0

    def generate_quads(self, frame, img, flags):
        binarized_img = img # does this copy img to binarized_img?

        # Is this necessary?
        self.all_quads = []
        self.all_corners = []
        self.all_quads_count = 0

        # Empiric bound for minimal allowed perimeter for squares
        MIN_SIZE = 25 # cv2.round(width*height*0.03*0.01*0.92)

        filter_quads = False # set by flags, ignore for now

        _img2, contours, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        hierarchy = hierarchy[0]


        if not contours:
            print("No Contours")
            return

        color_img = frame.img
        cv2.drawContours(color_img, contours, -1, (0, 255, 0), 3)

        contour_child_counter = []
        contour_quads = []
        board_index = -1

        no_hole_contours = []
        for i in range(len(contours) - 1, -1, -1):
            hierarchy_of_contour = hierarchy[i]
            parent_index = hierarchy_of_contour[3]
            # Filters out holes when looking for chessboard, but not necessary for grid finding
            # if hierarchy[i][2] != -1 or parent_index == -1:
            #     continue

            contour = contours[i]
            x,y,w,h = cv2.boundingRect(contour)
            cv2.rectangle(color_img, (x,y), (x+w, y+h), (0,0,255), 3)


        # cv2.drawContours(color_img, no_hole_contours, -1, (0, 0, 255), 3)

        cv2.imshow("ChessBoardTest", color_img)


# modules/calib3d/src/circlesgrid.cpp/hpp
class CirclesGridFinder:
    def __init__(self, pattern_size, points):
        self.pattern_size = pattern_size
        self.points = points
        self.parameters = GridFinderParameters()
        self.large_holes = 0
        self.small_holes = 0

    def find_holes(self, frame):
        vectors = []
        filtered_vectors = []
        basis = []
        basis_graphs = []
        graph = CirclesGraph(len(self.points))
        self.compute_graph(graph, vectors, frame)
        #self.filter_outliers_by_density(vectors, filtered_vectors, frame)
        self.find_basis(vectors, basis, basis_graphs, frame)

    def compute_graph(self, graph, vectors, frame):
        vectors.clear()
        points_size = len(self.points)

        img = frame.img

        for i in range(points_size):
            point_1 = self.points[i]
            x1, y1 = point_1
            for j in range(points_size):
                if i == j:
                    continue

                point_2 = self.points[j]
                x2, y2 = point_2
                distance = hypot(x2-x1, y2-y1)

                is_neighbors = True
                for k in range(points_size):
                    if k == i or k == j:
                        continue

                    point_3 = self.points[k]
                    x3, y3 = point_3
                    distance1 = hypot(x3 - x1, y3 - y1)
                    distance2 = hypot(x3 - x2, y3 - y2)
                    if distance1 < distance and distance2 < distance:
                        is_neighbors = False
                        break

                if is_neighbors:
                    graph.add_edge(i, j)
                    vector = tuple(np.subtract(point_1, point_2))
                    vectors.append(vector)
        #             cv2.line(img, (x1, y1), (x2, y2), (0,255,0), 2)
        # cv2.imshow("ChessBoardTest", img)

    # if this is to work, parameters.neighborhood_size and min_density need to be adjusted
    def filter_outliers_by_density(self, vectors, filtered_vectors, frame):
        if not vectors:
            print("ERROR: vectors is empty")
            return

        filtered_vectors.clear()
        for vector in vectors:
            h, w = self.parameters.density_neighborhood_size
            neighborhood_position = np.subtract(vector, self.parameters.density_neighborhood_size)
            x, y = neighborhood_position
            x = x / 2
            y = y / 2
            neighborhood = (x, y, h, w)

            neighbors_count = 0
            for sample in vectors:
                if contains(neighborhood, sample):
                    neighbors_count = neighbors_count + 1
            if neighbors_count >= self.parameters.min_density:
                filtered_vectors.append(vector)

        if not filtered_vectors:
            print("ERROR: filtered_vectors is empty")

    def find_basis(self, vectors, basis, basis_graphs, frame):
        basis.clear()

        samples = np.asarray(vectors, dtype=int)
        CLUSTERS_COUNT = 4
        best_labels = []
        criteria = (cv2.TERM_CRITERIA_EPS)
        compactness, labels, centers = cv2.kmeans(samples.reshape(1, 0), CLUSTERS_COUNT, best_labels, None, self.parameters.k_means_attempts, cv2.KMEANS_RANDOM_CENTERS)


    def get_holes(self, centers):
        pass


def contains(rectangle, point):
    x_pos, y_pos = point
    x, y, h, w = rectangle

    return x <= x_pos and x_pos < x + w and y <= y_pos and y_pos < y + h


class CirclesGraph:
    def __init__(self, size):
        self.vertices = {}
        for i in range(size):
            self.add_vertex(i)

    def does_vertex_exist(self, id):
        return id in self.vertices

    def add_vertex(self, id):
        assert not self.does_vertex_exist(id)
        self.vertices[id] = Vertex()

    def add_edge(self, id1, id2):
        assert self.does_vertex_exist(id1)
        assert self.does_vertex_exist(id2)

        self.vertices[id1].neighbors.add(id2)
        self.vertices[id2].neighbors.add(id1)


class Vertex:
    def __init__(self):
        self.neighbors = set()


class GridFinderParameters:
    def __init__(self):
        self.min_density = 10
        self.density_neighborhood_size = (16,16)
        self.min_distance_to_add_keypoint = 20
        self.k_means_attempts = 100
        self.convex_hull_factor = 1.1
        self.keypoint_scale = 1

        self.min_graph_confidence = 9
        self.vertex_gain = 1
        self.vertex_penalty = -0.6
        self.edge_gain = 1
        self.edge_penalty = 0.6
        self.existing_vertex_gain = 10000

        self.min_graph_edge_switch_distance = 5
        self.square_size = 1
        self.max_rectified_distance = self.square_size / 2



#####################################################################


def get_close_markers(markers, centroids=None, min_distance=20):
    if centroids is None:
        centroids = [m["centroid"] for m in markers]
    centroids = np.array(centroids)

    ti = np.triu_indices(centroids.shape[0], 1)

    def full_idx(i):
        # get the pair from condensed matrix index
        # defindend inline because ti changes every time
        return np.array([ti[0][i], ti[1][i]])

    # calculate pairwise distance, return dense distace matrix (upper triangle)
    distances = pdist(centroids, "euclidean")

    close_pairs = np.where(distances < min_distance)
    return full_idx(close_pairs)


def decode(square_img, grid_side_cell_count):
    step = square_img.shape[0] / grid_side_cell_count
    start = step / 2
    # look only at the center point of each grid cell
    # msg = square_img[start::step,start::step]

    # resize to grid size
    msg = cv2.resize(square_img, (grid, grid), interpolation=cv2.INTER_LINEAR)
    msg = msg > 50  # threshold

    # resample to 4 pixel per gridcell. using linear interpolation
    soft_msg = cv2.resize(square_img, (grid * 2, grid * 2), interpolation=cv2.INTER_LINEAR)
    # take the area mean to get a soft msg bit.
    soft_msg = cv2.resize(soft_msg, (grid, grid), interpolation=cv2.INTER_AREA)

    # border is: first row - last row and  first column - last column
    if msg[0:: grid - 1, :].any() or msg[:, 0:: grid - 1].any():
        # logger.debug("This is not a valid marker: \n %s" %msg)
        return None
    # strip border to get the message
    msg = msg[1:-1, 1:-1]
    soft_msg = soft_msg[1:-1, 1:-1]

    # out first bit is encoded in the orientation corners of the marker:
    #               MSB = 0                   MSB = 1
    #               W|*|*|W   ^               B|*|*|B   ^
    #               *|*|*|*  / \              *|*|*|*  / \
    #               *|*|*|*   |  UP           *|*|*|*   |  UP
    #               B|*|*|W   |               W|*|*|B   |
    # 0,0 -1,0 -1,-1, 0,-1
    # angles are counter-clockwise rotation
    corners = msg[0, 0], msg[-1, 0], msg[-1, -1], msg[0, -1]

    if sum(corners) == 3:
        msg_int = 0
    elif sum(corners) == 1:
        msg_int = 1
        corners = tuple([1 - c for c in corners])  # just inversion
    else:
        # this is no valid marker but maybe a maldetected one? We return unknown marker with None rotation
        return None

    # read rotation of marker by now we are guaranteed to have 3w and 1b
    # angle is number of 90deg rotations
    if corners == (0, 1, 1, 1):
        angle = 3
    elif corners == (1, 0, 1, 1):
        angle = 0
    elif corners == (1, 1, 0, 1):
        angle = 1
    else:
        angle = 2

    msg = np.rot90(msg, -angle - 2).transpose()
    soft_msg = np.rot90(soft_msg, -angle - 2).transpose()
    # Marker Encoding
    #  W |LSB| W      ^
    #  1 | 2 | 3     / \ UP
    # MSB| 4 | W      |
    # print angle
    # print msg    #the message is displayed as you see in the image

    msg = msg.tolist()
    msb = msg_int

    # strip orientation corners from marker
    del msg[0][0]
    del msg[0][-1]
    del msg[-1][0]
    del msg[-1][-1]
    # flatten list
    msg = [item for sublist in msg for item in sublist]
    while msg:
        # [0,1,0,1] -> int [MSB,bit,bit,...,LSB], note the MSB is definde above
        msg_int = (msg_int << 1) + msg.pop()

    # do the same for the soft msg image
    msg = soft_msg.tolist()
    msg_img = soft_msg
    # strip orientation corners from marker
    del msg[0][0]
    del msg[0][-1]
    del msg[-1][0]
    del msg[-1][-1]

    soft_msg = [item / 255.0 for sublist in msg for item in sublist] + [float(msb)]
    return angle, msg_int, soft_msg, msg_img


def correct_gradient(gray_img, r):
    # used just to increase speed - this simple check is still way to slow
    # lets assume that a marker has a black border
    # we check two pixels one outside, one inside both close to the border
    p1, _, p2, _ = r.reshape(4, 2).tolist()
    vector_across = p2[0] - p1[0], p2[1] - p1[1]
    ratio = 5.0 / sqrt(vector_across[0] ** 2 + vector_across[1] ** 2)  # we want to measure 5px away from the border
    vector_across = int(vector_across[0] * ratio), int(vector_across[1] * ratio)
    # indecies are flipped because numpy is row major
    outer = p1[1] - vector_across[1], p1[0] - vector_across[0]
    inner = p1[1] + vector_across[1], p1[0] + vector_across[0]
    try:
        gradient = int(gray_img[outer]) - int(gray_img[inner])
        return gradient > 20  # at least 20 shades darker inside
    except:
        # px outside of img frame, let the other method check
        return True


def detect_hips_markers(frame, gray_img, grid_side_cell_count=4, min_marker_perimeter=40, aperture=11, visualize=False):
    color_img = frame.img

    GRID_CELL_SIDE_LENGTH = 50   # I assume this is pixels
    grid_side_length = GRID_CELL_SIDE_LENGTH * grid_side_cell_count

    # top left,bottom left, bottom right, top right in image
    mapped_space = np.array(((0, 0), (grid_side_length, 0), (grid_side_length, grid_side_length), (0, grid_side_length)), dtype=np.float32).reshape(4, 1, 2)


    custom_finder(frame, gray_img)

    # TODO: See if better features detected with this instead of utso
    # edges = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, aperture, 9)


    #Get Corners
    # corners = cv2.goodFeaturesToTrack(gray_img, **gftt_params)
    #
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
    # cv2.cornerSubPix(gray_img, r, (3, 3), (-1, -1), criteria)

    markers = []

    # for r in rect_cand:
    #     if correct_gradient(gray_img, r):
    #         r = np.float32(r)
    #         # define the criteria to stop and refine the marker verts
    #         criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
    #         cv2.cornerSubPix(gray_img, r, (3, 3), (-1, -1), criteria)
    #
    #         M = cv2.getPerspectiveTransform(r, mapped_space)
    #         flat_marker_img = cv2.warpPerspective(gray_img, M,
    #                 (size, size))  # [, dst[, flags[, borderMode[, borderValue]]]])
    #         # Otsu documentation here :
    #         # https://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html#thresholding
    #         _, otsu = cv2.threshold(flat_marker_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #
    #         # getting a cleaner display of the rectangle marker
    #         kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    #         cv2.erode(otsu, kernel, otsu, iterations=3)
    #         # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    #         # cv2.dilate(otsu,kernel,otsu, iterations=1)
    #         marker = decode(otsu, grid_side_cell_count)
    #         if marker is not None:
    #             angle, msg, soft_msg, msg_img = marker
    #
    #             centroid = r.sum(axis=0) / 4.0
    #             centroid.shape = 2
    #             # angle is number of 90deg rotations
    #             # roll points such that the marker points correspond with oriented marker
    #             # rolling may not make the verts appear as you expect,
    #             # but using m_screen_to_marker() will get you the marker with proper rotation.
    #             r = np.roll(r, angle + 1, axis=0)  # np.roll is not the fastest when using these tiny arrays...
    #
    #             # id_confidence = 2*np.mean (np.abs(np.array(soft_msg)-.5 ))
    #             id_confidence = 2 * min(np.abs(np.array(soft_msg) - 0.5))
    #
    #             marker = {
    #                 "id": msg,
    #                 "id_confidence": id_confidence,
    #                 "verts": r.tolist(),
    #                 "soft_id": soft_msg,
    #                 "perimeter": cv2.arcLength(r, closed=True),
    #                 "centroid": centroid.tolist(),
    #                 "frames_since_true_detection": 0, }
    #             if visualize:
    #                 marker["otsu"] = np.rot90(otsu, -angle - 2).transpose()
    #                 marker["img"] = cv2.resize(msg_img, (20 * grid_size, 20 * grid_size),
    #                         interpolation=cv2.INTER_NEAREST, )
    #             if (marker["id"] != 32):  # marker 32 sucks because its just a single white spec.
    #                 markers.append(marker)
    return markers


def draw_markers(img, markers):
    for m in markers:
        centroid = np.array(m["centroid"], dtype=np.float32)
        origin = np.array(m["verts"][0], dtype=np.float32)
        hat = np.array([[[0, 0], [0, 1], [0.5, 1.25], [1, 1], [1, 0]]], dtype=np.float32)
        hat = cv2.perspectiveTransform(hat, m_marker_to_screen(m))
        if m["id_confidence"] > 0.9:
            cv2.polylines(img, np.int0(hat), color=(0, 0, 255), isClosed=True)
        else:
            cv2.polylines(img, np.int0(hat), color=(0, 255, 0), isClosed=True)
        # cv2.polylines(img,np.int0(centroid),color = (255,255,int(255*m['id_confidence'])),isClosed=True,thickness=2)
        m_str = "id: {:d}".format(m["id"])
        org = origin.copy()
        # cv2.rectangle(img, tuple(np.int0(org+(-5,-13))[0,:]), tuple(np.int0(org+(100,30))[0,:]),color=(0,0,0),thickness=-1)
        cv2.putText(img, m_str, tuple(np.int0(org)[0, :]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4,
                color=(0, 0, 255), )
        if "id_confidence" in m:
            m_str = "idc: {:.3f}".format(m["id_confidence"])
            org += (0, 12)
            cv2.putText(img, m_str, tuple(np.int0(org)[0, :]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4,
                    color=(0, 0, 255), )
        if "loc_confidence" in m:
            m_str = "locc: {:.3f}".format(m["loc_confidence"])
            org += (0, 12)
            cv2.putText(img, m_str, tuple(np.int0(org)[0, :]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4,
                    color=(0, 0, 255), )
        if "frames_since_true_detection" in m:
            m_str = "otf: {}".format(m["frames_since_true_detection"])
            org += (0, 12)
            cv2.putText(img, m_str, tuple(np.int0(org)[0, :]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4,
                    color=(0, 0, 255), )
        if "opf_vel" in m:
            m_str = "otf: {}".format(m["opf_vel"])
            org += (0, 12)
            cv2.putText(img, m_str, tuple(np.int0(org)[0, :]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4,
                    color=(0, 0, 255), )


def m_marker_to_screen(marker):
    # verts need to be sorted counterclockwise stating at bottom left
    # marker coord system:
    # +-----------+
    # |0,1     1,1|  ^
    # |           | / \
    # |           |  |  UP
    # |0,0     1,0|  |
    # +-----------+
    mapped_space_one = np.array(((0, 0), (1, 0), (1, 1), (0, 1)), dtype=np.float32)
    return cv2.getPerspectiveTransform(mapped_space_one, np.array(marker["verts"], dtype=np.float32))


def m_screen_to_marker(marker):
    # verts need to be sorted counterclockwise stating at bottom left
    # marker coord system:
    # +-----------+
    # |0,1     1,1|  ^
    # |           | / \
    # |           |  |  UP
    # |0,0     1,0|  |
    # +-----------+
    mapped_space_one = np.array(((0, 0), (1, 0), (1, 1), (0, 1)), dtype=np.float32)
    return cv2.getPerspectiveTransform(np.array(marker["verts"], dtype=np.float32), mapped_space_one)


# persistent vars for detect_markers_robust
lk_params = dict(winSize=(45, 45), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.03), )

prev_img = None
tick = 0


def detect_markers_robust(gray_img, grid_size, prev_markers, min_marker_perimeter=40, aperture=11, visualize=False,
        true_detect_every_frame=1, invert_image=False, ):
    global prev_img

    if invert_image:
        gray_img = 255 - gray_img

    global tick
    if not tick:
        tick = true_detect_every_frame
        new_markers = detect_hips_markers(gray_img, grid_size, min_marker_perimeter, aperture, visualize)
    else:
        new_markers = []
    tick -= 1

    if prev_img is not None and prev_img.shape == gray_img.shape and prev_markers:

        new_ids = [m["id"] for m in new_markers]

        # any old markers not found in the new list?
        not_found = [m for m in prev_markers if m["id"] not in new_ids and m["id"] >= 0]
        if not_found:
            prev_pts = np.array([np.array(m["verts"], dtype=np.float32) for m in not_found])
            prev_pts = np.vstack(prev_pts)
            new_pts, flow_found, err = cv2.calcOpticalFlowPyrLK(prev_img, gray_img, prev_pts, None,
                    minEigThreshold=0.01, **lk_params)
            for marker_idx in range(flow_found.shape[0] // 4):
                m = not_found[marker_idx]
                m_slc = slice(marker_idx * 4, marker_idx * 4 + 4)
                if flow_found[m_slc].sum() >= 4:
                    found, _ = np.where(flow_found[m_slc])
                    # calculate differences
                    old_verts = prev_pts[m_slc][found, :]
                    new_verts = new_pts[m_slc][found, :]
                    vert_difs = new_verts - old_verts
                    # calc mean dif
                    mean_dif = vert_difs.mean(axis=0)
                    # take n-1 closest difs
                    dist_variance = np.linalg.norm(mean_dif - vert_difs, axis=1)
                    if max(np.abs(dist_variance).flatten()) > 5:
                        m["frames_since_true_detection"] = 100
                    else:
                        closest_mean_dif = np.argsort(dist_variance, axis=0)[:-1, 0]
                        # recalc mean dif
                        mean_dif = vert_difs[closest_mean_dif].mean(axis=0)
                        # apply mean dif
                        proj_verts = prev_pts[m_slc] + mean_dif
                        m["verts"] = new_verts.tolist()
                        m["centroid"] = new_verts.sum(axis=0) / 4.0
                        m["centroid"].shape = 2
                        m["centroid"] = m["centroid"].tolist()
                        m["frames_since_true_detection"] += 1  # m['opf_vel'] = mean_dif
                else:
                    m["frames_since_true_detection"] = 100

        # cocatenating like this will favour older markers in the doublication deletion process
        markers = [m for m in not_found if m["frames_since_true_detection"] < 5] + new_markers
        if markers:  # del double detected markers
            min_distace = min([m["perimeter"] for m in markers]) / 4.0
            # min_distace = 50
            if len(markers) > 1:
                remove = set()
                close_markers = get_close_markers(markers, min_distance=min_distace)
                for f, s in close_markers.T:
                    # remove the markers further down in the list
                    remove.add(s)
                remove = list(remove)
                remove.sort(reverse=True)
                for i in remove:
                    del markers[i]
    else:
        markers = new_markers

    prev_img = gray_img.copy()
    return markers


# def bench(folder):
#     from os.path import join
#     from video_capture.av_file_capture import File_Capture
#     cap = File_Capture(join(folder,'marker-test.mp4'))
#
#     tracker = MarkerTracker()
#     detected_count = 0
#     for x in range(500):
#         frame = cap.get_frame()
#         img = frame.img
#         gray_img = cv2.cvtColor(img, cv2.cv.CV_BGR2GRAY)
#         markers = tracker.track_in_frame(gray_img,5,visualize=True)
#         draw_markers(img, markers)
#         cv2.imshow('Detected Markers', img)
#         if cv2.waitKey(1) == 27:
#            break
#         detected_count += len(markers)
#
#     print detected_count #3106 #3226


def bench(folder):
    from os.path import join
    from video_capture.av_file_capture import File_Capture

    cap = File_Capture(join(folder, "marker-test.mp4"))
    markers = []
    detected_count = 0

    for x in range(500):
        frame = cap.get_frame()
        img = frame.img
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        markers = detect_markers_robust(gray_img, 5, prev_markers=markers, true_detect_every_frame=1, visualize=True)

        draw_markers(img, markers)
        cv2.imshow("Detected Markers", img)

        # for m in markers:
        #     if 'img' in m:
        #         cv2.imshow('id %s'%m['id'], m['img'])
        #         cv2.imshow('otsu %s'%m['id'], m['otsu'])
        if cv2.waitKey(1) == 27:
            break
        detected_count += len(markers)
    print(detected_count)  # 2900 #3042 #3021


if __name__ == "__main__":
    folder = "/Users/mkassner/Desktop/"
    import cProfile, subprocess, os

    cProfile.runctx("bench(folder)", {"folder": folder}, locals(), os.path.join(folder, "world.pstats"), )
    loc = os.path.abspath(__file__).rsplit("pupil_src", 1)
    gprof2dot_loc = os.path.join(loc[0], "pupil_src", "shared_modules", "gprof2dot.py")
    subprocess.call(
            "cd {} ; python {} -f pstats world.pstats | dot -Tpng -o world_cpu_time.png".format(folder, gprof2dot_loc),
            shell=True, )
    print("created  time graph for  process. Please check out the png next to this file")
