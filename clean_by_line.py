import math

import cv2
import numpy as np

import img_util


def midpoint(x1, y1, x2, y2):
    x_mid = int((x1 + x2) / 2)
    y_mid = int((y1 + y2) / 2)

    return x_mid, y_mid


# 清除图片文字
def clean_txt_of_img(txt_points, img_in, all_mask):
    mask = np.zeros(img_in.shape[:2], np.uint8)

    x0, y0 = txt_points[0]
    x1, y1 = txt_points[1]
    x2, y2 = txt_points[2]
    x3, y3 = txt_points[3]

    x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)

    x_mid1, y_mi1 = midpoint(x0, y0, x3, y3)

    thickness = int(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)*0.6)

    cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mi1), 255, thickness)

    img_in = cv2.inpaint(img_in, mask, 10, cv2.INPAINT_TELEA)

    all_mask = np.add(all_mask, mask)

    # 返回图像
    return img_in, all_mask


# 按列表清除图片上的文字
def clean(txt_info_list, img_in):
    img_out = img_in.copy()

    all_mask = np.zeros(img_in.shape[:2], np.uint8)

    # 遍历文字列表
    for txt_info in txt_info_list:
        # 执行清理
        img_out, all_mask = clean_txt_of_img(txt_info[0], img_out, all_mask)

    # 返回图像
    return img_out, all_mask
