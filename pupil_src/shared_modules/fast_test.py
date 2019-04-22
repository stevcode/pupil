import cv2
import uvc
import numpy as np
import math

from hips_detectors.interest_point import InterestPoint
from hips_detectors.maze_grid_code import MazeGridCode

from camera_models import Radial_Dist_Camera


# region Pupil World Camera Initialization

uvc_capture = None
devices = uvc.Device_List()
devices_by_name = {dev["name"]: dev for dev in devices}

name = "Pupil Cam1 ID2"
preferred_names = (name,)

# try to init by name
for name in preferred_names:
    for d_name in devices_by_name.keys():
        if name in d_name:
            uid_for_name = devices_by_name[d_name]["uid"]
            try:
                uvc_capture = uvc.Capture(uid_for_name)
            except uvc.OpenError:
                print("{} matches {} but is already in use or blocked.".format(uid_for_name, name))
            except uvc.InitError:
                print("Camera failed to initialize.")
            else:
                break

cap = None
if uvc_capture is None:
    cap = cv2.VideoCapture(0)   # Capture from webcam instead.
else:
    uvc_capture.frame_size = (1280, 720)
    uvc_capture.frame_rate = 60

# endregion

# region Camera Intrinsics Setup

camera_matrix = [[829.3510515270362, 0.0, 659.9293047259697],
                 [0.0, 799.5709408845464, 373.0776462356668],
                 [0.0, 0.0, 1.0], ]

dist_coefs = [[-0.43738542863224966, 0.190570781428104, -0.00125233833830639, 0.0018723428760170056, -0.039219091259637684,]]

camera_model = Radial_Dist_Camera(camera_matrix, dist_coefs, (1280, 720), name)

# endregion


def image_histogram_based_thresholding(img):
    # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html
    # Using Otsu's Binarization to start.
    # If this fails, directly implement icvBinarizationHistogramBased() seen here: https://github.com/opencv/opencv/blob/1913482cf5d257d7da292bb2c15f22d0588d34dc/modules/calib3d/src/calibinit.cpp#L520
    # Source Code here: https://github.com/opencv/opencv/blob/1913482cf5d257d7da292bb2c15f22d0588d34dc/modules/calib3d/src/calibinit.cpp#L342

    blurred_img = cv2.GaussianBlur(img, (5, 5), 0)
    ret, otsu_threshold_img = cv2.threshold(blurred_img, 0, 255,
                                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # definition has a problem, especially when hand is added
    return otsu_threshold_img


def locating(color_img, gray_img, is_drawing=False):
    # threshold_image = image_histogram_based_thresholding(gray_img)
    #
    # # So we can find rectangles that go to the edge, we draw a white line around the image edge.
    # # Otherwise, find_contours will miss those clipped rectangle contours.
    # # The border color will be the image mean, because otherwise we risk screwing up filters like cv2.smooth
    # image_height, image_width = threshold_image.shape
    # cv2.rectangle(threshold_image, (0,0), (image_width, image_height), (0,0,0), 5, cv2.LINE_8)
    #
    # _img2, contours, hierarchy = cv2.findContours(threshold_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # hierarchy = hierarchy[0]
    #
    # if not contours:
    #     print("No Contours")
    #     return
    #
    # no_parent_contours = []
    # for i in range(len(contours) - 1, -1, -1):
    #     if hierarchy[i][2] == -1:
    #         continue
    #
    #     contour = contours[i]
    #     no_parent_contours.append(contour)
    #
    #     # x, y, w, h = cv2.boundingRect(contour)
    #     # cv2.rectangle(color_img, (x, y), (x + w, y + h), (0, 0, 255), 3)
    #
    # if not no_parent_contours:
    #     print("No Parentless Contours, falling back to all contours")
    #     no_parent_contours = contours
    #
    # largest_contours = sorted(no_parent_contours, key=lambda c: cv2.contourArea(c), reverse=True)
    # largest_contours = largest_contours[:1]
    # largest_contour = largest_contours[0]
    #
    # if is_drawing:
    #     cv2.drawContours(color_img, [largest_contour], -1, (0, 255, 0), 3)
    # # cv2.imshow("Location Testing", color_img)
    #
    # return largest_contour

    threshold_image = image_histogram_based_thresholding(gray_img)
    edges = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 9)
    # Initiate STAR detector
    orb = cv2.ORB_create()
    fast = cv2.FastFeatureDetector_create()

    # Initiate BRIEF extractor
    # brief = cv2.DescriptorExtractor_create("BRIEF")

    # find the keypoints with STAR
    kp = fast.detect(gray_img, None)
    cv2.drawKeypoints(color_img, kp, color_img)
    cv2.imshow("Location Testing", color_img)

    # compute the descriptors with BRIEF
    # kp, des = brief.compute(img, kp)


# region Feature Globals

# For Featuring
prev_points = []  # Points at t-1
curr_points = []  # Points at t
lines = []  # To keep all the lines overtime

track_len = 10
detect_interval = 5

prev_gray_img = None
width, height = (1280, 720)

# GoodFeaturesToTrack params
gftt_params = dict( maxCorners=1000, qualityLevel=0.1, minDistance=10)
# maxCorners=1000, 500
# qualityLevel=0.1, 0.01
# minDistance=10
# blockSize=None

# Lucas kanade params
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# winSize=(10,10)
# maxLevel=3 or 4
# criteria=(cv2.CV_TERMCRIT_ITER | cv2.CV_TERMCRIT_EPS, 20, 0.03)

# endregion


def featuring(color_img, gray_img):
    global prev_gray_img
    global prev_points
    global curr_points
    global lines
    global track_len
    global detect_interval

    if prev_gray_img is None:
        prev_gray_img = image_histogram_based_thresholding(gray_img)
        return

    if len(lines) > 0:
        img0, img1 = prev_gray_img, image_histogram_based_thresholding(gray_img)
        prev_points = np.float32([line[-1] for line in lines]).reshape(-1, 1, 2)
        curr_points, status, err = cv2.calcOpticalFlowPyrLK(img0, img1, prev_points, None, **lk_params)
        prev_points_reversed, status, err = cv2.calcOpticalFlowPyrLK(img1, img0, curr_points, None, **lk_params)
        dist = abs(prev_points - prev_points_reversed).reshape(-1, 2).max(-1)
        is_good = dist < 1
        new_lines = []
        for line, (x, y), good_flag in zip(lines, curr_points.reshape(-1, 2), is_good):
            if not good_flag:
                continue
            line.append((x, y))
            if len(line) > track_len:
                del line[0]
            new_lines.append(line)
            cv2.circle(color_img, (x, y), 2, (0, 255, 0), -1)
        lines = new_lines
        cv2.polylines(color_img, [np.int32(line) for line in lines], False, (0, 255, 0))

    if detect_interval == 5:
        mask = np.zeros_like(gray_img)
        mask[:] = 255
        for x, y in [np.int32(line[-1]) for line in lines]:
            cv2.circle(mask, (x,y), 5, 0, -1)
        points = cv2.goodFeaturesToTrack(prev_gray_img, mask=mask, **gftt_params)
        # points = refined_points(frame, color_img, gray_img, points)
        if points is not None:
            for x, y in np.float32(points).reshape(-1, 2):
                lines.append([(x, y)])

    detect_interval -= 1
    if detect_interval <= 0:
        detect_interval = 5

    prev_gray_img = image_histogram_based_thresholding(gray_img)

    cv2.imshow("Feature Movement", color_img)


def refined_points(color_img, gray_img, points):
    largest_contour = locating(color_img, gray_img)
    x, y, w, h = cv2.boundingRect(largest_contour)
    # cv2.rectangle(color_img, (x, y), (x + w, y + h), (0, 0, 255), 3)
    rps = []
    for p in points:
        g = p[0]
        x = g[0]
        y = g[1]
        pt = (x, y)
        if cv2.pointPolygonTest(largest_contour, pt, False) >= 0:
            rps.append(p)

    return rps


def qr(color_img, gray_img):
    corners = cv2.goodFeaturesToTrack(gray_img, **gftt_params)
    # cv2.drawKeypoints(color_img, corners, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    points = [p[0] for p in corners]
    grid_size = InterestPoint.calculate_grid_size(points)
    half_grid = int(grid_size / 2)
    i_points = []
    for c in corners:
        pt = c[0]
        i_point = InterestPoint(pt)
        i_points.append(i_point)
        cv2.rectangle(color_img, (i_point.x - half_grid, i_point.y - half_grid),
                      (i_point.x + half_grid, i_point.y + half_grid),
                      (255, 255, 255))  # cv2.rectangle(color_img, (x - 10, y - 10), (x + 10, y + 10), (255, 255, 255))


def homography_test(color_img, gray_img):
    # print(MazeGridCode.get_angle_between_cells(0,14) * 180.0 / math.pi)
    is_displaying = True

    corners = cv2.goodFeaturesToTrack(gray_img, **gftt_params)

    # Display GFTTs
    if is_displaying:
        # Set the needed parameters to find the refined corners
        winSize = (5, 5)
        zeroZone = (-1, -1)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_COUNT, 40, 0.001)
        # Calculate the refined corner locations
        corners = cv2.cornerSubPix(gray_img, corners, winSize, zeroZone, criteria)

        kp_corners = [cv2.KeyPoint(c[0][0], c[0][1], 13) for c in corners]
        cv2.drawKeypoints(color_img, kp_corners, color_img, (0,255,0))

    # Convert GFTTs to boxes and display
    points = [p[0] for p in corners]
    # grid_cell_side_length = InterestPoint.calculate_grid_size(points)
    # half_grid_cell_side_length = math.ceil(grid_cell_side_length / 2)
    i_points = []
    first_corner = None
    for c in corners:
        pt = c[0]

        i_point = InterestPoint(pt)
        i_points.append(i_point)

        if first_corner is None:
            first_corner = i_point

    if first_corner is None:
        return

    size = 18
    NORMAL_WINDOW = (size, size)
    RESIZED_WINDOW = (600, 600)

    x1 = first_corner.x - size
    x2 = first_corner.x + size
    y1 = first_corner.y - size
    y2 = first_corner.y + size

    if x1 < 0 or y1 < 0:
        return

    roi = gray_img[y1:y2, x1:x2]
    # integral1_img, integral2_img, integral3_img = cv2.integral3(roi)
    _, thresh_img = cv2.threshold(roi, 100, 255, cv2.THRESH_BINARY)
    image_height, image_width = thresh_img.shape
    cv2.rectangle(thresh_img, (0,0), (image_width, image_height), (0,0,0), 1, cv2.LINE_8)
    _img2, contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # hierarchy = hierarchy[0]
    #
    # if not contours:
    #     print("No Contours")
    #     return
    #
    # no_parent_contours = []
    # for i in range(len(contours) - 1, -1, -1):
    #     if hierarchy[i][2] == -1:
    #         continue
    #
    #     contour = contours[i]
    #     no_parent_contours.append(contour)
    #
    #     x, y, w, h = cv2.boundingRect(contour)
    #     cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 0, 255), 3)

    for c in contours:
        M = cv2.moments(c)
        x = int(M["m10"] / M["m00"])
        y = int(M["m01"] / M["m00"])
        cv2.circle(thresh_img, (x, y), 7, (0, 0, 0), -1)

    cv2.namedWindow("zoom", cv2.WINDOW_NORMAL)
    cv2.imshow("zoom", thresh_img)


    cv2.resizeWindow("zoom", 600, 600)


    # Find top left feature
    # top_left = InterestPoint.top_left_point(i_points, grid_cell_side_length)
    # if is_displaying:
    #     cv2.rectangle(color_img, (top_left.x - half_grid_cell_side_length, top_left.y - half_grid_cell_side_length),
    #                   (top_left.x + half_grid_cell_side_length, top_left.y + half_grid_cell_side_length), (0, 255, 0))

    # matching_i_points = MazeGridCode.scale_test_find(color_img, gray_img, i_points)

def compute_gftt_descriptors(gray_img, corner):
    descriptors = []
    # Descriptor 0: connection_count
    # Number of lines connected to the corner
    # Values: 1, 2, 3, 4 (zero value means the corner was too close to the edge of the image)
    connection_count = 0
    size = 18

    x1 = corner.x - size
    x2 = corner.x + size
    y1 = corner.y - size
    y2 = corner.y + size

    if not (x1 < 0 or y1 < 0):
        roi = gray_img[y1:y2, x1:x2]
        _, thresh_img = cv2.threshold(roi, 100, 255, cv2.THRESH_BINARY)
        image_height, image_width = thresh_img.shape
        cv2.rectangle(thresh_img, (0, 0), (image_width, image_height), (0, 0, 0), 1, cv2.LINE_8)
        _img2, contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        connection_count = len(contours)
    descriptors.append(connection_count)

    # Descriptor 1: angle


orb = cv2.ORB_create()
fast = cv2.FastFeatureDetector_create()
brisk = cv2.BRISK_create()
gfft = cv2.GFTTDetector_create()
kaze = cv2.KAZE_create()
def create_surface(color_img, gray_img):
    kps = brisk.detect(gray_img, None)

    copy_img = color_img.copy()
    cv2.drawKeypoints(copy_img, kps, copy_img, (255,0,0))
    cv2.imshow("SurfaceDef", copy_img)
    cv2.setMouseCallback("SurfaceDef", define_quadrant, (color_img, gray_img, kps))

rx1 = 0
rx2 = 0
ry1 = 0
ry2 = 0
is_roi_ready = False
down_params = None
def define_quadrant(event, x, y, flags, params):
    global rx1, rx2, ry1, ry2
    global is_roi_ready
    global down_params

    if event == cv2.EVENT_LBUTTONDOWN:
        rx1 = x
        ry1 = y
        down_params = params
        print("down")
    if event == cv2.EVENT_LBUTTONUP:
        rx2 = x
        ry2 = y

        color_img, gray_img, key_points = down_params

        kp, desc = create_marker((rx1, ry1, rx2, ry2), color_img, gray_img, key_points)

        copy_img = color_img.copy()
        cv2.rectangle(copy_img, (rx1, ry1), (rx2, ry2), (0, 0, 255))
        cv2.drawKeypoints(copy_img, kp, copy_img, (255, 0, 0), 4)
        cv2.imshow("SurfaceDef", copy_img)
        is_roi_ready = True

# def create_surface(frames):
#     def create_marker():
#         pass

mgc = None
def create_marker(roi, color_img, gray_img, key_points):
    def find_closest(corner, key_points):
        x1, y1 = corner
        closest_distance = float('inf')
        closest_key_point = None
        for kp in key_points:
            x2, y2 = kp.pt
            manhattan_distance = abs(x2 - x1) + abs(y2 - y1)
            # if manhattan_distance >= 20:
            #     continue
            if closest_key_point is None or manhattan_distance < closest_distance:
                closest_key_point = kp
                closest_distance = manhattan_distance
        return closest_key_point

    global mgc
    kp = []
    for pt in key_points:
        x, y = pt.pt
        if MazeGridCode.is_point_in_rect(x, y, roi):
            kp.append(pt)

    corners = cv2.goodFeaturesToTrack(gray_img, **gftt_params)
    kps = set()
    for corner in corners:
        pt = corner[0]
        x1, y1 = pt
        for k in kp:
            x2, y2 = k.pt
            manhattan_distance = abs(x2 - x1) + abs(y2 - y1)
            if manhattan_distance <= 20:
                kps.add(k)

        # closest_kp = find_closest(pt, kp)
        # kps.append(closest_kp)

    print(str(len(corners)))
    print(str(len(kp)))
    print(str(len(kps)))


    # Add keypoints at corners to gray_img so that boundary keypoints aren't removed
    left_kp = cv2.KeyPoint()
    left_kp.pt = (0,0)
    kps.add(left_kp)

    left_kp = cv2.KeyPoint()
    left_kp.pt = (0, gray_img.size)
    kps.add(left_kp)

    right_kp = cv2.KeyPoint()
    right_kp.pt = (gray_img.size, gray_img.size)
    kps.add(right_kp)

    right_kp = cv2.KeyPoint()
    right_kp.pt = (gray_img.size, 0)
    kps.add(right_kp)

    print(str(len(kps)))

    kps_out, desc = brisk.compute(gray_img, list(kps))
    # if desc is None:
    #     desc = []

    mgc = MGC(kps_out, desc)

    print(str(len(kps_out)))
    print(str(len(desc)))

    return list(kps), desc


class MGC:
    def __init__(self, key_points, descriptors):
        self.key_points = key_points
        self.descriptors = descriptors
        self.brute_force_matcher = cv2.BFMatcher(cv2.NORM_L2SQR)

    def find_match_flann(self, kp, desc):
        if self.descriptors is None or desc is None:
            return []

        index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict()
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(self.descriptors, desc, k=2)
        good_points = []
        try:
            for m, n in [x for x in matches if len(x) >= 2]:
                # print("m.dist=" + str(m.distance) + ", n.distance=" + str(n.distance))
                if m.distance < 0.8 * n.distance:
                    good_points.append(m)
        except:
            pass
        finally:
            return good_points

    def find_match_brute(self, kp, desc):
        if self.descriptors is None or desc is None:
            return []


        # matches = self.brute_force_matcher.knnMatch(self.descriptors, desc, k=2)
        return self.brute_force_matcher.match(self.descriptors, desc)


# region Instant Testing

adv_left = cv2.imread("C:\work\pictures\ADV\ADV.png", 0)
#       # # O O
#       O O # O
#       O # # #
#       O # # O

corners = cv2.goodFeaturesToTrack(adv_left, **gftt_params)

# Display GFTTs
kp_corners = [cv2.KeyPoint(c[0][0], c[0][1], 13) for c in corners]
test = cv2.drawKeypoints(adv_left, kp_corners, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("Test", test)

# Convert GFTTs to boxes and display
# points = [p[0] for p in corners]
# corners_mask = np.zeros_like(adv_left)
# grid_size = InterestPoint.calculate_grid_size(points)
# half_grid = int(grid_size / 2)
# i_points = []
# for c in corners:
#     pt = c[0]
#     i_point = InterestPoint(pt)
#     i_points.append(i_point)
#     cv2.rectangle(corners_mask, (i_point.x - half_grid, i_point.y - half_grid),
#                   (i_point.x + half_grid, i_point.y + half_grid), (255, 255, 255), -1)
#
# cv2.imshow("Mask", corners_mask)


# Find mgc adv0
# top_left = InterestPoint.top_left_point(i_points, grid_size)
# # cv2.rectangle(adv_left, (top_left.x - half_grid, top_left.y - half_grid),
# #               (top_left.x + half_grid, top_left.y + half_grid), (255, 255, 255), -1)
#
# matching_i_points = MazeGridCode.test_find(adv_left, top_left, i_points, grid_size)

# # temp_copy = adv_left
# for i_point in i_points:
#     # temp_copy = adv_left.copy()
#     matching_i_points = MazeGridCode.test_find(adv_left, i_point, i_points, grid_size)
#     # cv2.imshow("Find QR", adv_left)
#     # key = cv2.waitKey(50)
#     # if key == 27:  # esc to quit
#     #     break
#     if matching_i_points:
#         break
# cv2.imshow("Find QR", adv_left)


# endregion


# region Program Loop

is_locating = False
is_featuring = False
is_qr = False
is_homography = True

frame_count = 0

while True:
    is_threshing = False

    if uvc_capture is None:
        _, frame = cap.read()
    else:
        frame = uvc_capture.get_frame()
    frame_count += 1
    color_img = frame if uvc_capture is None else frame.img
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if uvc_capture is None else frame.gray
    undistorted_img = color_img if uvc_capture is None else camera_model.undistort(color_img)
    undistorted_gray_img = gray_img if uvc_capture is None else camera_model.undistort(gray_img)
    if is_threshing:
        undistorted_gray_img = image_histogram_based_thresholding(undistorted_gray_img)


    # if is_locating and frame_count >= 10:
    #     frame_count = 0
    #     locating(frame, img, gray_img, True)
    # create_surface(undistorted_img, undistorted_gray_img)

    if mgc is not None:
        kp = brisk.detect(undistorted_gray_img, None)
        # cv2.drawKeypoints(color_img, kp, color_img, (255, 0, 0))
        kp, desc = brisk.compute(undistorted_gray_img, kp)
        matches = mgc.find_match_brute(kp, desc)
        matching_kp = [kp[m.trainIdx] for m in matches]
        # print("matches=" + str(len(matches)))

        cv2.drawKeypoints(undistorted_img, matching_kp, undistorted_img, (0, 0, 255), 4)

    if is_qr and frame_count >= 10:
        frame_count = 0
        qr(frame, undistorted_img, undistorted_gray_img)

    if is_featuring and frame_count >= 10:
        frame_count = 0
        featuring(undistorted_img, undistorted_gray_img)

    if is_homography and frame_count >= 10:
        frame_count = 10
        homography_test(undistorted_img, undistorted_gray_img)

    integral1_img, integral2_img, integral3_img = cv2.integral3(undistorted_img)
    cv2.imshow("Testing", undistorted_img)

    key = cv2.waitKey(1)
    if key == 27:    # esc to quit
        break
    if key == 108:    # l to locating
        is_locating = False
        is_featuring = False
        is_qr = False
        print("Locating")
        is_locating = True
    if key == 102:      # f to featuring
        is_locating = False
        is_featuring = False
        is_qr = False
        print("Featuring")
        is_featuring = True
    if key == 113:      # q to qr
        is_locating = False
        is_featuring = False
        is_qr = False
        print("Matching")
        is_qr = True
    if key == 104:  # h to homography_test
        is_locating = False
        is_featuring = False
        is_qr = False
        print("Matching")
        is_homography = True
    if key == 32:       # space to take pic
        # create_surface(color_img, gray_img)
        create_surface(undistorted_img, undistorted_gray_img)
        # cv2.imwrite("C:\work\pictures\captured\maze_color.png", img)
        # cv2.imwrite("C:\work\pictures\captured\maze_gray.png", gray_img)
        # cv2.imwrite("C:\work\pictures\captured\maze_thresh.png", image_histogram_based_thresholding(gray_img))

# endregion

# region Dispose of camera connection

devices.cleanup()
devices = None
if uvc_capture is None:
    uvc_capture.close()
    uvc_capture = None
else:
    cap.release()
cv2.destroyAllWindows()

# endregion



