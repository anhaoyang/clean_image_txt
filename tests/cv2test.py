import cv2
import sys
import numpy as np

# 读取图像
image0 = cv2.imread('C:\\Users\\ChengHaoQian\\Desktop\\3.png', cv2.IMREAD_UNCHANGED)
image = cv2.imread('C:\\Users\\ChengHaoQian\\Desktop\\3.png', cv2.IMREAD_GRAYSCALE)

# 转换图像为灰度图
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# 使用Canny边缘检测
edges = cv2.Canny(image, threshold1=1000, threshold2=0)

# 找到边缘闭环
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 复制原始图像以保持原始像素
result = image0.copy()

ddd = (int(image0[0,0][0]),int(image0[0,0][1]),int(image0[0,0][2]))

# 循环遍历边缘闭环并填充
for contour in contours:
    fill_color = np.mean(edges[contour[0][:, 1], contour[0][:, 0]], axis=0)
    # fill_color = fill_color.astype(np.uint8)
    cv2.drawContours(result, [contour], 0, ddd, thickness=cv2.FILLED)


# dst = cv2.inpaint(result, edges, 10, cv2.INPAINT_TELEA)

# 显示结果图像
cv2.imshow('Filled Edges', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# for arr in edges:
#     for i in arr:
#         sys.stdout.write('\t')
#         sys.stdout.write(str(i))
#     sys.stdout.write("\r\n")