import cv2
import math

import numpy as np

from hips_detectors.interest_point import InterestPoint


class MazeGridCode:
    CELL_COUNT = 4  # grid is 4x4
    REAL_CELL_SIZE = 7.9375  # in mm
    REAL_GRID_SIZE = REAL_CELL_SIZE * CELL_COUNT
    MAPPED_REAL_GRID_SPACE = np.array(((0, 0), (REAL_GRID_SIZE, 0), (REAL_GRID_SIZE, REAL_GRID_SIZE), (0, REAL_GRID_SIZE)), dtype=np.float32).reshape(4, 1, 2)


    def __init__(self):
        self.grid_step_size = None
        self.maze_name = None
        self.maze_code_section_id = None
        self.dominant_cell_id = None
        self.last_cell_id = None
        self.code = []                  # 0 = White Square Mask, 1 = Black Square Mask
        self.code_i_points = [None]*16
        self.homography_destination_points = []

    def set_far_cell_id(self):
        far_cell_id = 0
        far_cell_distance = 0
        y1, x1 = divmod(self.dominant_cell_id, MazeGridCode.CELL_COUNT)
        for cell_id in range(len(self.code)):
            if self.code[cell_id] == 1:
                continue
            y2, x2 = divmod(cell_id, MazeGridCode.CELL_COUNT)
            manhattan_distance = abs(x1 - x2) + abs(y1 - y2)
            if manhattan_distance >= far_cell_distance:
                far_cell_distance = manhattan_distance
                far_cell_id = cell_id
        self.last_cell_id = far_cell_id

    def set_homography_destination_points(self):
        destination_points = []
        cell_id = 0
        for cell in self.code:
            if cell == 1:
                cell_id += 1
                continue
            x_offset, y_offset = self.get_cell_offsets(cell_id, 100)
            destination_points.append((x_offset, y_offset))
            cell_id += 1
        self.homography_destination_points = np.array(destination_points)

    # def is_match(self, reference_code):
    #     is_matching = self.code == reference_code.code
    #     if is_matching:
    #         self.maze_name = reference_code.maze_name
    #         self.maze_code_section_id = reference_code.maze_code_section_id
    #         self.dominant_cell_id = reference_code.dominant_cell_id

    def is_match(self, color_img, gray_img, i_point, all_i_points, grid_cell_side_length, offset_angle):
        # grid_cell_side_length += 2
        x_offset, y_offset = MazeGridCode.get_cell_offsets(self.dominant_cell_id, grid_cell_side_length)
        x_start = i_point.x - x_offset
        y_start = i_point.y - y_offset

        matching_i_points = []
        is_a_match = True
        search_rects = []
        for cell_id in range(16):
            is_interest_point_location = self.code[cell_id] == 0
            x_cell_offset, y_cell_offset = MazeGridCode.get_cell_offsets(cell_id, grid_cell_side_length)
            x_grid = x_start + x_cell_offset
            y_grid = y_start + y_cell_offset
            half_grid = int(grid_cell_side_length / 2)
            search_rects.append(((x_grid - half_grid, y_grid - half_grid), (x_grid + half_grid, y_grid + half_grid)))
            # cv2.rectangle(gray_img, (x_grid - half_grid, y_grid - half_grid), (x_grid + half_grid, y_grid + half_grid), (255, 255, 255))

            is_a_point_in_cell = False
            for pt in all_i_points:
                x, y = pt.x, pt.y
                rect = (x - half_grid, y - half_grid, x + half_grid, y + half_grid)
                if MazeGridCode.is_point_in_rect(x_grid, y_grid, rect):
                    is_a_point_in_cell = True
                    if is_interest_point_location:
                        matching_i_points.append(pt)
                    break
            if is_a_point_in_cell and not is_interest_point_location:
                is_a_match = False
                break
            if is_interest_point_location and not is_a_point_in_cell:
                is_a_match = False
                break

        if is_a_match:
            for pt in matching_i_points:
                x, y = pt.x, pt.y
                cv2.rectangle(color_img, (x-half_grid, y-half_grid), (x+half_grid, y+half_grid), (255, 255, 255))
        else:
            matching_i_points = None

        # for rect in search_rects:
        #     cv2.rectangle(gray_img, rect[0], rect[1], (255, 255, 255))

        # cv2.imshow("Match Found", gray_img)
        return matching_i_points

    def is_match_fast(self, color_img, gray_img, i_point, all_i_points, grid_cell_side_length):
        # cv2.putText(color_img, str(grid_cell_side_length), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0))
        half_grid_cell_side_length = math.ceil(grid_cell_side_length / 2)

        # grid_cell_side_length += 2
        x_offset, y_offset = MazeGridCode.get_cell_offsets(self.dominant_cell_id, grid_cell_side_length)
        x_start = i_point.x - x_offset
        y_start = i_point.y - y_offset

        x_cell_offset, y_cell_offset = MazeGridCode.get_cell_offsets(self.last_cell_id, grid_cell_side_length)
        x_end = i_point.x + x_cell_offset
        y_end = i_point.y + y_cell_offset
        # cv2.rectangle(color_img, (x_end - half_grid_cell_side_length, y_end - half_grid_cell_side_length),
        #               (x_end + half_grid_cell_side_length, y_end + half_grid_cell_side_length), (255, 0, 0))

        diagonal_distance = math.hypot(x_end - x_start, y_end - y_start)
        angle = self.get_angle_between_cells(self.dominant_cell_id, self.last_cell_id)
        # cv2.circle(color_img, (x_start, y_start), int(diagonal_distance), (255, 0, 0), 1)

        diagonal_match = None
        closest_match_angle = 180
        inner_matches = []
        for pt in all_i_points:
            x, y = pt.x, pt.y
            distance = math.hypot(x - x_start, y - y_start)
            angle_difference = abs(angle - self.get_angle_between_points(i_point, pt))
            if diagonal_distance - half_grid_cell_side_length / 2 <= distance <= diagonal_distance + half_grid_cell_side_length / 2:
                if diagonal_match is None:
                    closest_match_angle = angle_difference
                    diagonal_match = pt
                elif angle_difference < closest_match_angle:
                    closest_match_angle = angle_difference
                    diagonal_match = pt

            if distance <= diagonal_distance + half_grid_cell_side_length / 2:
                inner_matches.append(pt)

        if diagonal_match is not None:
            x, y = diagonal_match.x, diagonal_match.y
            # cv2.rectangle(color_img, (x - half_grid_cell_side_length, y - half_grid_cell_side_length),
            #               (x + half_grid_cell_side_length, y + half_grid_cell_side_length), (0, 0, 255))

        return diagonal_match, inner_matches, closest_match_angle

    @staticmethod
    def is_point_in_rect(x, y, rect):
        x1, y1, x2, y2 = rect
        return x1 <= x <= x2 and y1 <= y <= y2

    @staticmethod
    def get_cell_offsets(cell_id, grid_cell_side_length):
        y, x = divmod(cell_id, MazeGridCode.CELL_COUNT)
        x_offset = x * grid_cell_side_length
        y_offset = y * grid_cell_side_length

        return x_offset, y_offset

    @staticmethod
    def get_angle_between_cells(cell_id1, cell_id2):
        y1, x1 = divmod(cell_id1, MazeGridCode.CELL_COUNT)
        y2, x2 = divmod(cell_id2, MazeGridCode.CELL_COUNT)
        return math.atan2(y2 - y1, x2 - x1) * 180 / math.pi

    @staticmethod
    def get_angle_between_points(i_pt1, i_pt2):
        return math.atan2(i_pt2.y - i_pt1.y, i_pt2.x - i_pt1.x) * 180 / math.pi

    @staticmethod
    def _force_adv_0():
        mg_code = MazeGridCode()
        mg_code.maze_name = "ADV-C"
        mg_code.maze_code_section_id = 0
        mg_code.dominant_cell_id = 0
        mg_code.code = [0, 0, 1, 1,
                        1, 1, 0, 1,
                        1, 0, 0, 0,
                        1, 0, 0, 1]
        mg_code.set_far_cell_id()
        mg_code.set_homography_destination_points()
        return mg_code

    @staticmethod
    def test_find(color_img, gray_img, top_left, i_points, grid_cell_side_length):
        mgc_adv0 = MazeGridCode._force_adv_0()

        mgc_adv0.is_match_fast(color_img, gray_img, top_left, i_points, grid_cell_side_length)

    @staticmethod
    def scale_test_find(color_img, gray_img, i_points):
        # top_left = InterestPoint.top_left_point(i_points, 10)
        # cv2.circle(color_img, (top_left.x, top_left.y), 10, (255, 0, 0), 1)
        #
        # mgc_adv0 = MazeGridCode._force_adv_0()
        # grid_cell_side_lengths = [15, 20, 25, 30, 35, 40, 45, 50]
        #
        # for grid_cell_side_length in grid_cell_side_lengths:
        #     fast_match = mgc_adv0.is_match_fast(color_img, gray_img, top_left, i_points, grid_cell_side_length)
        #     if fast_match is not None:
        #         # cv2.circle(color_img, (fast_match.x, fast_match.y), 10, (255, 0, 0), 1)
        #         match = mgc_adv0.is_match(color_img, gray_img, top_left, i_points, grid_cell_side_length)

        mgc_adv0 = MazeGridCode._force_adv_0()
        grid_cell_side_lengths = [25, 30, 35, 40, 45, 50]

        is_match_found = False
        matches = []
        for i_pt in i_points:
            if is_match_found:
                break
            for grid_cell_side_length in grid_cell_side_lengths:
                fast_match, inner_matches, closest_match_angle = mgc_adv0.is_match_fast(color_img, gray_img, i_pt, i_points, grid_cell_side_length)
                if fast_match is not None:
                    # cv2.circle(color_img, (fast_match.x, fast_match.y), 10, (255, 0, 0), 1)
                    matches = mgc_adv0.is_match(color_img, gray_img, i_pt, inner_matches, grid_cell_side_length,
                                                closest_match_angle)
                    if matches is not None:
                        is_match_found = True
                        break

        # if is_match_found:
        #     homography_source_points = np.array([(i_pt.x, i_pt.y) for i_pt in matches])
        #     H, status = cv2.findHomography(homography_source_points, mgc_adv0.homography_destination_points)
        #     img_out = cv2.warpPerspective(color_img, H, (300, 300))
        #     cv2.imshow("Homography", img_out)
