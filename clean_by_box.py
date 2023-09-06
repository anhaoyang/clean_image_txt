import img_util
import cv2


# 清除图片文字
def clean_txt_of_img(txt_points, img_out, all_mask):
    # 获取蒙版
    txt_mask = img_util.get_img_mask_bg(img_out)

    # 记录到全局，方便分析问题
    img_util.fill_mask_for_points(all_mask, txt_points)

    # 填充当前区域
    img_util.fill_mask_for_points(txt_mask, txt_points)

    # 获取灰度图
    img_gray = cv2.cvtColor(txt_mask, cv2.COLOR_BGR2GRAY)

    # 返回图像
    return cv2.inpaint(img_out, img_gray, 20, cv2.INPAINT_TELEA)


# 按列表清除图片上的文字
def clean(txt_info_list, img_in):
    # 定义输出图像
    img_out = img_in.copy()

    all_mask = img_util.get_img_mask_bg(img_out)

    # 遍历文字列表
    for txt_info in txt_info_list:
        # 执行清理
        img_out = clean_txt_of_img(txt_info[0], img_out, all_mask)

    # 返回图像
    return img_out, all_mask
