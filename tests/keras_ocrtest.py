import math

import cv2
import keras_ocr
import numpy as np


def midpoint(x1, y1, x2, y2):
    x_mid = int((x1 + x2) / 2)
    y_mid = int((y1 + y2) / 2)

    return (x_mid, y_mid)


def inpaint_text(img_path, pipeline):
    # read image
    img = keras_ocr.tools.read(img_path)

    # generate (word, box) tuples 
    prediction_groups = pipeline.recognize([img])
    mask = np.zeros(img.shape[:2], dtype="uint8")

    for box in prediction_groups[0]:
        print(box)

        x0, y0 = box[1][0]
        x1, y1 = box[1][1]
        x2, y2 = box[1][2]
        x3, y3 = box[1][3]

        x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)

        x_mid1, y_mi1 = midpoint(x0, y0, x3, y3)

        thickness = int(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))

        cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mi1), 255, thickness)

        img = cv2.inpaint(img, mask, 7, cv2.INPAINT_NS)

    return (img)


if __name__ == '__main__':
    pipeline = keras_ocr.pipeline.Pipeline()

    img = '../1.png'
    out = inpaint_text(img, pipeline)
    cv2.imshow("test", out)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
