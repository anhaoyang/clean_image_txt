import numpy as np
import cv2


def replaceZeroes(data):
    min_nonzero = min(data[np.nonzero(data)])
    data[data == 0] = min_nonzero
    return data


def SSR(src_img, size):
    L_blur = cv2.GaussianBlur(src_img, (size, size), sigmaX=3)
    img = replaceZeroes(src_img)
    L_blur = replaceZeroes(L_blur)
    dst_Img = cv2.log(img / 255.0)
    dst_Lblur = cv2.log(L_blur / 255.0)
    dst_IxL = cv2.multiply(dst_Img, dst_Lblur)
    log_R = cv2.subtract(dst_Img, dst_IxL)
    dst_R = cv2.normalize(log_R, None, 0, 255, cv2.NORM_MINMAX)
    log_uint8 = cv2.convertScaleAbs(dst_R)
    return log_uint8


if __name__ == '__main__':
    img = './3.png'
    size = 3
    src_img = cv2.imread(img)
    b_gray, g_gray, r_gray = cv2.split(src_img)
    b_gray = SSR(b_gray, size)
    g_gray = SSR(g_gray, size)
    r_gray = SSR(r_gray, size)
    result = cv2.merge([b_gray, g_gray, r_gray])
    # cv2.imshow('aaa', result)

    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    canny_img = cv2.Canny(gray, threshold1=0, threshold2=255)

    # sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    # sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    # canny_img = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    # 使用Scharr算子进行边缘检测
    # scharr_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
    # scharr_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
    # canny_img = np.sqrt(scharr_x ** 2 + scharr_y ** 2)

    cv2.imshow('canny_img', canny_img)

    k = np.ones((3, 3), np.uint8)
    img2 = cv2.morphologyEx(canny_img, cv2.MORPH_CLOSE, k)  # 闭运算
    # img2 = cv2.erode(cv2.dilate(canny_img, k), k)
    cv2.imshow('img2', img2)

    dst = cv2.inpaint(src_img, img2, 30000, cv2.INPAINT_TELEA)
    cv2.imshow('dst', dst)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
