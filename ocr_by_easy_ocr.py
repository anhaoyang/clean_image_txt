import easyocr

# 定义 ocr 识别器
OCR_READER = easyocr.Reader(['ch_sim', 'en'],
                            gpu=True,  # 是否使用GPU，默认False
                            model_storage_directory="model",  # 定义自动下载的OCR模型文件存放地址
                            download_enabled=True)  # 是否自动下载OCR模型文件


# 从图片中读取文字
def read_txt(img_path):
    # 读取图片
    info_list = OCR_READER.readtext(image=img_path)

    # 打印图片文字
    for txt_info in info_list:
        print(txt_info)

    # 返回
    return info_list
