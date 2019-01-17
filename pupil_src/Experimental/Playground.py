import os.path
import cv2
import numpy as np
from scipy.spatial.distance import pdist
from scipy.interpolate import interp1d


# because np.sqrt is slower when we do it on small arrays
def reversedEnumerate(l):
    return zip(range(len(l) - 1, -1, -1), reversed(l))


from math import sqrt, fabs, log10

sqrt_2 = sqrt(2)

def matchShapes(contour, HM):
    anyA = False
    anyB = False
    ma = cv2.HuMoments(cv2.moments(contour)).flatten()
    mb = HM
    sma = 0
    smb = 0
    epsilon = 0.00001
    mmm = 0
    result = 0

    for i in range(0,7):
        ama = fabs(ma[i])
        amb = fabs(mb[i])

        if ama > 0:
            anyA = True
        if amb > 0:
            anyB = True

        if ma[i] > 0:
            sma = 1
        elif ma[i] < 0:
            sma = -1
        else:
            sma = 0

        if mb[i] > 0:
            smb = 1
        elif ma[i] < 0:
            smb = -1
        else:
            smb = 0

        if ama > epsilon and amb > epsilon:
            ama = 1.0 / (sma * log10(ama))
            amb = 1.0 / (smb * log10(amb))
            result = result + fabs(-ama + amb)

    if anyA != anyB:
        return float('inf')

    return result

if __name__ == '__main__':
    # start_box = cv2.imread(r'C:\work\start_box.bmp')
    # start_box_bw = cv2.cvtColor(start_box, cv2.COLOR_RGB2GRAY)
    # HM = cv2.HuMoments(cv2.moments(start_box_bw)).flatten()
    #
    # edges = cv2.imread(r'C:\work\edges.bmp')
    # im_bw = cv2.cvtColor(edges, cv2.COLOR_RGB2GRAY)
    # _img, contours, hierarchy = cv2.findContours(im_bw, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE,
    #                                              offset=(0, 0))  # TC89_KCOS
    # hierarchy = hierarchy[0]
    # contours = np.array(contours)
    #
    # # keep only contours                        with parents
    # contained_contours = contours[np.logical_and(hierarchy[:, 3] >= 0, True)]
    #
    #
    # aprox_contours = [cv2.approxPolyDP(c, epsilon=2.5, closed=True) for c in contained_contours]
    #
    # for c in aprox_contours:
    #     similarity = matchShapes(c, HM)
    #     if similarity <= 0.93:
    #         print(similarity)
    #         cv2.drawContours(edges, [c], -1, (0, 0, 255), 3)
    # cv2.imwrite(r'C:\work\humoments.bmp', edges)
    #
    #
    #
    # longest_contour = None
    # longest_contour_length = 0
    # for c in aprox_contours:
    #     if len(c) > longest_contour_length:
    #         longest_contour_length = len(c)
    #         longest_contour = c
    #
    # min_x = float('inf')
    # min_y = float('inf')
    # max_x = 0
    # max_y = 0
    #
    # for point in longest_contour:
    #     point = point[0]
    #     x = point[0]
    #     y = point[1]
    #     if x > max_x:
    #         max_x = x
    #     if x < min_x:
    #         min_x = x
    #     if y > max_y:
    #         max_y = y
    #     if y < min_y:
    #         min_y = y
    #
    # width = max_x - min_x
    # section_Width = width / 3
    # height = max_y - min_y
    # section_height = height / 3
    #
    # a = []
    #
    # for point in longest_contour:
    #     p = point[0]
    #     x = p[0]
    #     y = p[1]
    #
    #
    #     x_a_start = min_x
    #     x_a_end = min_x + section_Width
    #     y_a_start = min_y
    #     y_a_end = min_y + section_height
    #
    #     if x_a_start <= x < x_a_end and y_a_start <= y < y_a_end:
    #         a.append(point)
    #
    # test = np.array(a)
    # print(test)
    #
    # w, h = 1280, 720
    # blank_image = np.zeros((h, w, 3), np.uint8)
    # blank_image[:] = (255, 255, 255)
    #
    # cv2.drawContours(edges, test, -1, (0, 0, 255), 3)
    # cv2.imwrite(r'C:\work\edges_section.bmp', edges)
    #
    #
    # grid_size = 20
    # size = 20 * grid_size
    # mapped_space = np.array(((0, 0), (size, 0), (size, size), (0, size)), dtype=np.float32).reshape(4, 1, 2)

    ref = cv2.imread(r'C:\work\upperleft3.bmp')
    img = cv2.imread(r'C:\work\edges.bmp')

    orb = cv2.ORB_create()
    kp_ref, des_ref = orb.detectAndCompute(ref, None)
    kp_img, des_img = orb.detectAndCompute(img, None)

    # FLANN parameters
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6,  # 12
                        key_size=12,  # 20
                        multi_probe_level=1)  # 2
    search_params = dict(checks=150)  # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des_ref, des_img, k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]

    # ratio test as per Lowe's paper
    keypoints = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]
            keypoints.append(kp_img[matches[i][0].trainIdx])

    draw_params = dict(matchColor=(0, 255, 0), singlePointColor=(255, 0, 0), matchesMask=matchesMask, flags=0)

    all = cv2.drawMatchesKnn(ref, kp_ref, img, kp_img, matches, None, **draw_params)

    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # matches = bf.match(des_ref, des_img)
    # matches = sorted(matches, key = lambda x: x.distance)
    # all = cv2.drawMatches(ref, kp_ref, img, kp_img, matches[:10], None, flags=2)
    cv2.imshow("Image", all)


    # ref = cv2.drawKeypoints(ref, kp_ref, None)
    # img = cv2.drawKeypoints(img, kp_img, None)

#    kppp = cv2.drawKeypoints(img, keypoints, None)
#    cv2.imshow("Image", kppp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
