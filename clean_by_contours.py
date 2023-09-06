import cv2
import numpy as np

import img_util


# 图像寻边闭运算
def do_fill_contour(img_gray):
    # 找到边缘闭环
    contours, _ = cv2.findContours(img_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 循环遍历边缘闭环并填充
    for contour in contours:
        cv2.drawContours(img_gray, [contour], 0, (255, 255, 255), thickness=cv2.FILLED)

    return img_gray


# 清除图片文字
def clean_txt_of_img(txt_points, img_in, all_mask):
    # 获取指定区域
    img_dealing = img_util.get_img_for_points(img_in, txt_points)

    # 转为灰度图
    img_gray = cv2.cvtColor(img_dealing, cv2.COLOR_BGR2GRAY)

    # 边缘检测
    img_canny = cv2.Canny(img_gray, threshold1=0, threshold2=0)

    # 寻边检测
    img_contour = do_fill_contour(img_canny)

    # 重绘
    img_out = cv2.inpaint(img_in, img_contour, 20, cv2.INPAINT_NS)

    # 记录蒙版
    all_mask = np.add(all_mask, img_contour)

    # 返回图像
    return (img_out, all_mask)


# 按列表清除图片上的文字
def clean(txt_info_list, img_in):
    # 定义输出图像
    img_out = img_in.copy()

    # 记录蒙版
    all_mask = np.zeros(img_in.shape[:2], np.uint8)

    # 遍历文字列表
    for txt_info in txt_info_list:
        # 执行清理
        img_out, all_mask = clean_txt_of_img(txt_info[0], img_out, all_mask)

    # 返回图像
    return (img_out, all_mask)
