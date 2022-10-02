import cv2
import random
import numpy as np
import logging
img = cv2.imread("ggwp.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


segmentator = cv2.ximgproc.segmentation.createGraphSegmentation(sigma=0.5, k=290, min_size=4000)
src = cv2.imread("../ggwp.jpg")
segment = segmentator.processImage(src)
seg_image = np.zeros(src.shape, np.uint8)


for i in range(np.max(segment)):
  # 將第 i 個分割的座標取出
  y, x = np.where(segment == i)

  #logging.warning(src[y[0],x[0]])
  # 隨機產生顏色
  color = [random.randint(0, 255), random.randint(0, 255),random.randint(0, 255)]
  pixel_avg = [0,0,0]
  threshold = 0

  #計算道路色素平均值
  compair_pixel_avg = [133 ,136, 139]

  if(x.__len__()<30000):
    # 設定第 i 個分割區的顏色
    pixel_avg = [0,0,0]
    #計算區塊色素平均值
    for xi, yi in zip(x, y):
        #計算道路平均值
        pixel_avg = pixel_avg+ src[yi, xi]
    pixel_avg[0]/=x.__len__()
    pixel_avg[1]/=x.__len__()
    pixel_avg[2]/=x.__len__()

    pixel_avg = pixel_avg - compair_pixel_avg
    threshold = pixel_avg[0]+pixel_avg[1]+pixel_avg[2]


    if(threshold<30 and threshold>-30):
        logging.warning(i)
        logging.warning(threshold)
        for xi, yi in zip(x, y):
        #計算道路平均值
             seg_image[yi, xi] = color

  result = cv2.addWeighted(src, 0.3, seg_image, 0.7, 0)
# 顯示結果
  cv2.rectangle(result, (380, 250), (600, 380), (0, 255, 0), 1)
  cv2.imshow("Result", result)

# 將原始圖片與分割區顏色合併
result = cv2.addWeighted(src, 0.3, seg_image, 0.7, 0)
# 顯示結果
cv2.rectangle(result, (380, 250), (600, 380), (0, 255, 0), 1)
cv2.imshow("Result", result)
cv2.waitKey(0)


# kernel_size = 5
# blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

# low_threshold = 50
# high_threshold = 150
# edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

# rho = 2  # distance resolution in pixels of the Hough grid
# theta = np.pi / 180  # angular resolution in radians of the Hough grid
# threshold = 100  # minimum number of votes (intersections in Hough grid cell)
# min_line_length = 10  # minimum number of pixels making up a line
# max_line_gap = 10  # maximum gap in pixels between connectable line segments
# line_image = np.copy(img) * 0  # creating a blank to draw lines on

# Run Hough on edge detected image
# Output "lines" is an array containing endpoints of detected line segments
# lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
#                         min_line_length, max_line_gap)

# for line in lines:
#     for x1, y1, x2, y2 in line:
        
#         cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
# Draw the lines on the  image
# lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)

# cv2.imshow("img", lines_edges)
# cv2.waitKey(0)
