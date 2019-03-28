import cv2

import numpy as np


class InterestPoint:
    def __init__(self, sub_pixel_point):
        self.sub_pixel_x, self.sub_pixel_y = sub_pixel_point
        self.x = int(self.sub_pixel_x)
        self.y = int(self.sub_pixel_y)
        self.angle = None

    @staticmethod
    def calculate_grid_size(points):
        # Get smallest x and y distances be each point
        smallest_x_distances = []
        smallest_y_distances = []
        for i in range(len(points) - 1):
            smallest_x_distance = float('inf')
            smallest_y_distance = float('inf')
            x0, y0 = points[i]
            for j in range(i + 1, len(points)):
                x1, y1 = points[j]

                x_distance = abs(x0 - x1)
                if smallest_x_distance > x_distance > 6:
                    smallest_x_distance = x_distance

                y_distance = abs(y0 - y1)
                if smallest_y_distance > y_distance > 6:
                    smallest_y_distance = y_distance

            if smallest_y_distance < float('inf') and smallest_y_distance < float('inf'):

                smallest_x_distances.append(smallest_x_distance)
                smallest_y_distances.append(smallest_y_distance)

        # Average them for blank maze
        distances = smallest_x_distances + smallest_y_distances
        if not distances:
            return 1
        avg = sum(distances) / len(distances)
        if avg < float('inf'):
            grid_size = int(avg)
        else:
            grid_size = 1
        return grid_size

    # @staticmethod
    def top_left_point(i_points, grid_size):
        if not i_points:
            return

        most_left = sorted(i_points, key=lambda p: p.x)[0]
        all_left = [p for p in i_points if p.x < most_left.x + (grid_size)]
        top_left = sorted(all_left, key=lambda p: p.y)[0]
        return top_left

    # @staticmethod
    # def build_qr(i_points, grid_size):
    #     if not i_points:
    #         return
    #
    #     i_points = [InterestPoint(p) for p in points]
    #
    #     top_left = InterestPoint.top_left_point(i_points, grid_size)
    #
    #     corners_mask = np.zeros_like(gray_img)
    #     for c in corners:
    #         x, y = c[0]
    #         x = int(x)
    #         y = int(y)  # cv2.rectangle(color_img, (x - 10, y - 10), (x + 10, y + 10), (255, 255, 255), -1)