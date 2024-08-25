import cv2
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
# 灰度化
img = cv2.imread("lenna.png")  # cv2读取的是BGR
h, w = img.shape[:2]  # shape是一个元组，切片操作得到h，w
img_gray = np.zeros([h, w], img.dtype)  # 创建大小和数据类型相同的数组
for i in range(h):
    for j in range(w):
        m = img[i, j]  # R0.3 G0.59 B0.11
        img_gray[i, j] = int(m[0]*0.11+m[1]*0.59+m[2]*0.3)

print(img_gray)
cv2.imshow("win_img_gray", img_gray)

plt.subplot(2, 2, 1)
img = plt.imread("lenna.png")
plt.imshow(img)
# 灰度化
img_gray = rgb2gray(img)
plt.subplot(2, 2, 2)
plt.imshow(img_gray, cmap='gray')  # 默认是绿通，需要设置灰度

# 二值化
rows, cols = img_gray.shape
for i in range(rows):
    for j in range(cols):
        if(img_gray[i, j] <= 0.5):
            img_gray[i, j] = 0
        else:
            img_gray[i, j] = 1

img_binary = np.where(img_gray >= 0.5, 1, 0)
plt.subplot(2, 2, 3)
plt.imshow(img_binary, cmap='gray')
plt.show()