import cv2

import clean_by_box
import clean_by_contours
import clean_by_line
import clean_by_morphology
import clean_by_keras_ocr
import ocr_by_easy_ocr as ocr

if __name__ == '__main__':
    # 1.图片地址
    img_path = "5.png"

    # 2.读取图片内容
    txt_info_list = ocr.read_txt(img_path)

    # 读取图像
    img_in = cv2.imread(img_path)

    # 3.清除图片文字

    # 使用简单多边形
    # img_box, mask_box = clean_by_box.clean(txt_info_list, img_in)
    # cv2.imshow("img_box", img_box)
    # cv2.imshow("mask_box", mask_box)
    #
    # # 使用线条
    # img_line, mask_line = clean_by_line.clean(txt_info_list, img_in)
    # cv2.imshow("img_line", img_line)
    # cv2.imshow("mask_line", mask_line)
    #
    # 边缘检测的
    img_morphology, mask_morphology = clean_by_morphology.clean(txt_info_list, img_in)
    img_morphology, mask_morphology = clean_by_morphology.clean(txt_info_list, img_morphology)
    img_morphology, mask_morphology = clean_by_morphology.clean(txt_info_list, img_morphology)
    img_morphology, mask_morphology = clean_by_morphology.clean(txt_info_list, img_morphology)
    cv2.imshow("mask_morphology", mask_morphology)
    img_line, mask_morphology = clean_by_line.clean(txt_info_list, img_morphology)
    img_contours, mask_morphology = clean_by_contours.clean(txt_info_list, img_line)
    cv2.imshow("img_morphology*4", img_morphology)
    cv2.imshow("+img_line", img_line)
    cv2.imshow("+img_contours", img_contours)
    # cv2.imshow("mask_morphology", mask_morphology)
    cv2.imshow("img_in", img_in)

    # 寻边检测的
    # img_contours, mask_contours = clean_by_contours.clean(txt_info_list, img_in)
    # cv2.imshow("img_contours", img_contours)
    # cv2.imshow("mask_contours", mask_contours)
    #
    # # 使用keras_ocr
    # img_keras_ocr = clean_by_keras_ocr.clean(img_path)
    # cv2.imshow("img_keras_ocr", img_keras_ocr)

    # 5.窗体展示停留
    cv2.waitKey(0)
    cv2.destroyAllWindows()
