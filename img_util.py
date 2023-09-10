import array

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


if __name__ == '__main__':
    arr = array.array('i', [-1] * 65)
    print(len(arr))


def remove_img_box_mask_line(mask, points):
    _minX, _maxX, _minY, _maxY = get_point_min_max_xy(points)

    # 获取指定区域坐标（冗余2像素，防止扫描不到）
    minX = int(max(0, _minX - 2))
    minY = int(max(0, _minY - 2))
    maxX = int(min(_maxX + 2, len(mask[0])))
    maxY = int(min(_maxY + 2, len(mask)))

    # 构建
    top = array.array('i', [-1] * len(mask[0]))
    bottom = array.array('i', [-1] * len(mask[0]))
    left = array.array('i', [-1] * len(mask))
    right = array.array('i', [-1] * len(mask))

    # 扫描每个像素蒙版
    for y in range(minY, maxY):
        for x in range(minX, maxX):
            # 判断是否存在蒙版（存在蒙版时为255）
            if mask[y][x] > 0:
                if minY > 0:
                    # 上边框不为0时收集（因为为0时不出框）
                    top[x] = y if top[x] < 0 else min(top[x], y)
                if maxY < len(mask):
                    # 下边框不为0时收集（因为为0时不出框）
                    bottom[x] = y if bottom[x] < 0 else max(bottom[x], y)
                if minX > 0:
                    # 左边框不为0时收集（因为为0时不出框）
                    left[y] = x if left[y] < 0 else min(left[y], x)
                if maxX < len(mask[0]):
                    # 右边框不为0时收集（因为为0时不出框）
                    right[y] = x if right[y] < 0 else max(right[y], x)

    # 清除扫描到的边框
    for y in range(minY, maxY):
        for x in range(minX, maxX):
            if top[x] > -1:
                mask[top[x]][x] = 0
            if bottom[x] > -1:
                mask[bottom[x]][x] = 0
        if left[y] > -1:
            mask[y][left[y]] = 0
        if right[y] > -1:
            mask[y][right[y]] = 0

    # 返回蒙版
    return mask


def get_point_min_max_xy(points):
    x1, y1 = points[0]
    x2, y2 = points[1]
    x3, y3 = points[2]
    x4, y4 = points[3]

    return min(x1, x2, x3, x4), max(x1, x2, x3, x4), min(y1, y2, y3, y4), max(y1, y2, y3, y4)


# 划红线
def line_img(img_in, txt_points):
    # 转array
    pts = np.array(txt_points, np.int32)

    img_in = cv2.polylines(img_in, [pts], True, (0, 0, 255), 1)

    return img_in


# 重绘
def img_inpaint(img_in, img_gray):
    return cv2.inpaint(img_in, img_gray, 10, cv2.INPAINT_TELEA)
