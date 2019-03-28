import cv2

import numpy as np


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
        self.code = []                  # 0 = White Square Mask, 1 = Black Square Mask
        self.code_i_points = [None]*16

    # def is_match(self, reference_code):
    #     is_matching = self.code == reference_code.code
    #     if is_matching:
    #         self.maze_name = reference_code.maze_name
    #         self.maze_code_section_id = reference_code.maze_code_section_id
    #         self.dominant_cell_id = reference_code.dominant_cell_id

    def is_match(self, gray_img, i_point, all_i_points, grid_size):
        grid_size += 2
        x_offset, y_offset = MazeGridCode.get_cell_offsets(self.dominant_cell_id, grid_size)
        x_start = i_point.x - x_offset
        y_start = i_point.y - y_offset

        matching_i_points = []
        is_a_match = True
        search_rects = []
        for cell_id in range(16):
            is_interest_point_location = self.code[cell_id] == 0
            x_cell_offset, y_cell_offset = MazeGridCode.get_cell_offsets(cell_id, grid_size)
            x_grid = x_start + x_cell_offset
            y_grid = y_start + y_cell_offset
            half_grid = int(grid_size / 2)
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
                cv2.rectangle(gray_img, (x-half_grid, y-half_grid), (x+half_grid, y+half_grid), (255, 255, 255))
        else:
            matching_i_points = None

        # for rect in search_rects:
        #     cv2.rectangle(gray_img, rect[0], rect[1], (255, 255, 255))

        # cv2.imshow("Match Found", gray_img)
        return matching_i_points

    @staticmethod
    def is_point_in_rect(x, y, rect):
        x1, y1, x2, y2 = rect
        return x1 <= x <= x2 and y1 <= y <= y2


    @staticmethod
    def get_cell_offsets(cell_id, grid_size):
        y, x = divmod(cell_id, MazeGridCode.CELL_COUNT)
        x_offset = x * grid_size
        y_offset = y * grid_size

        return x_offset, y_offset


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
        return mg_code

    @staticmethod
    def test_find(gray_img, top_left, i_points, grid_size):
        mgc_adv0 = MazeGridCode._force_adv_0()

        mgc_adv0.is_match(gray_img, top_left, i_points, grid_size)
