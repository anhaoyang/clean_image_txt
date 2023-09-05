import cv2
import numpy as np

import img_util


# 将 数组中的0 替换为 最小非零值
def replace_zeroes(data):
    min_nonzero = min(data[np.nonzero(data)])
    data[data == 0] = min_nonzero
    return data


# 图像增强算法
def ssr(src_img, size=3):
    # 对图像做高斯模糊，以降低图像中的噪声
    blur_img = cv2.GaussianBlur(src_img, (size, size), sigmaX=3)

    # 原图 - 像素0值替换
    no_zero_src_img = replace_zeroes(src_img)
    # 高斯图 - 像素0值替换
    no_zero_blur_img = replace_zeroes(blur_img)

    # 原图 - 像素值对数变换，用于得出像素点高度
    log_src_img = cv2.log(no_zero_src_img / 255.0)
    # 高斯图 - 像素值对数变换，用于得出像素点高度
    log_blur_img = cv2.log(no_zero_blur_img / 255.0)

    # 像素乘法，提高像素点差异性
    multiply_img = cv2.multiply(log_src_img, log_blur_img)

    # 像素减法，剔除原图值，只保留超出255的部分
    subtract_img = cv2.subtract(log_src_img, multiply_img)

    # 像素高度正常化，将像素值归化到[0,255]之间
    normalize_img = cv2.normalize(subtract_img, None, 0, 255, cv2.NORM_MINMAX)

    # 将浮点型结果转换为无符号8位整数
    uint8_img = cv2.convertScaleAbs(normalize_img)

    # 返回
    return uint8_img


# 图像增强
def ssr_img(img_in):
    # 获取图像RBG序列
    b_gray, g_gray, r_gray = cv2.split(img_in)

    # 蓝-通道图像增强
    b_gray = ssr(b_gray)
    # 绿-通道图像增强
    g_gray = ssr(g_gray)
    # 红-通道图像增强
    r_gray = ssr(r_gray)

    # 合为完整图像
    merged_gray = cv2.merge([b_gray, g_gray, r_gray])

    # 返回
    return merged_gray


# 图像形态闭运算
def do_morphologyEx(img_canny):
    k = np.ones((3, 3), np.uint8)
    img_mo = cv2.morphologyEx(img_canny, cv2.MORPH_CLOSE, k)
    return img_mo


# 清除图片文字
def clean_txt_of_img(txt_points, img_in, all_mask):
    # 获取指定区域
    img_dealing = img_util.get_img_for_points(img_in, txt_points)

    # 展示一下
    # cv2.imshow("img_dealing", img_dealing)

    # 获取增强后的图像
    img_ssr = ssr_img(img_dealing)

    # 转为灰度图
    img_gray = cv2.cvtColor(img_ssr, cv2.COLOR_BGR2GRAY)

    # 边缘检测
    img_canny = cv2.Canny(img_gray, threshold1=0, threshold2=255)
    # cv2.imshow("img_canny", img_canny)

    # 形态运算
    img_mo = do_morphologyEx(img_canny)

    # 展示
    # cv2.imshow("img_mo", img_mo)

    # 图像重绘
    img_inpaint = img_util.img_inpaint(img_in, img_mo)

    # 记录mask
    all_mask = np.add(all_mask, img_mo)

    # 返回图像
    return img_inpaint, all_mask


# 按列表清除图片上的文字
def clean(txt_info_list, img_in):
    # 定义输出图像
    img_out = img_in.copy()

    all_mask = np.zeros(img_in.shape[:2], np.uint8)

    # 遍历文字列表
    for txt_info in txt_info_list:
        # 执行清理
        img_out, all_mask = clean_txt_of_img(txt_info[0], img_out, all_mask)

    # 返回图像
    return img_out, all_mask
