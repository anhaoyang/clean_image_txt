import cv2
import numpy as np


# 获取图像蒙版背景
def get_img_mask_bg(img_in):
    # 创建图床
    img_mo_bg = np.ones(img_in.shape, dtype=np.uint8)

    # 黑色背景
    img_mo_bg = img_mo_bg * 0

    # 返回
    return img_mo_bg


# 获取图像指定区域
def get_img_for_points(img_in, txt_points):
    mask = get_img_mask_bg(img_in)

    # 获取指定区域蒙版
    fill_mask_for_points(mask, txt_points)

    # 获取需要处理的区域
    img_dealing = cv2.bitwise_and(img_in, mask)

    # 返回
    return img_dealing


# 获取图像蒙版
def fill_mask_for_points(mask, txt_points):
    # 转array
    pts = np.array(txt_points, np.int32)

    # 将指定区域填充为白色
    mask = cv2.fillPoly(mask, [pts], (255, 255, 255))

    # 返回蒙版
    return mask


# 划红线
def line_img(img_in, txt_points):
    # 转array
    pts = np.array(txt_points, np.int32)

    img_in = cv2.polylines(img_in, [pts], True, (0, 0, 255), 1)

    return img_in


# 重绘
def img_inpaint(img_in, img_gray):
    return cv2.inpaint(img_in, img_gray, 0, cv2.INPAINT_TELEA)
