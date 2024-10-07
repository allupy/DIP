import cv2
import os
import matplotlib.pyplot as plt

from myNearest import nearest
from myBilinear import bilinear
from myBicubic import bicubic


# 请大家完成三个插值算法函数 myNearest.m、myBilinear.m、myCubic.m，然后将本主程序中的变量
# filename设置为不同值（'lenna'、'cameraman'、'building'），运行main.m，即可得到插值结果

ratio_1 = 0.4; # 缩放因子1
ratio_2 = 3;   # 缩放因子2
# filename = 'lenna'; # 测试图像1
# # filename = 'cameraman'; %测试图像2
# # filename = 'building'; %测试图像3

# 定义缩放因子
ratio_1 = 0.4  # 缩放因子1
ratio_2 = 3    # 缩放因子2

filenames = ['lenna','cameraman','building']
for filename in filenames:
    print(filename)

    # 读取图像
    im = cv2.imread(f'{filename}.jpg')
    
    
    if len(im.shape) == 2:  # 灰度图  
        print("0")  
    elif len(im.shape) == 3 and im.shape[2] == 1:  # 在某些情况下，OpenCV可能以(高度, 宽度, 1)的形式返回灰度图  
        print("2")  
    elif len(im.shape) == 3 and im.shape[2] == 3:  # RGB图  
        print("3")  
    else:  
        print("这是其他类型的图像")


    row, col,channel = im.shape  # 得到图像尺寸
    im_center = im[int(row*3/8):int(row*5/8), int(col*3/8):int(col*5/8):]  # 截取中间图像块，用于图像放大

    # 将图像长宽缩放为原图的 ratio_1 (<1)倍
    im1_n = nearest(im, ratio_1)
    im1_b = bilinear(im, ratio_1)
    im1_c = bicubic(im, ratio_1)

    # 将图像长宽缩放为原图的 ratio_2 (>1)倍
    im2_n = nearest(im_center, ratio_2)
    im2_b = bilinear(im_center, ratio_2)
    im2_c = bicubic(im_center, ratio_2)

    # 创建 result 文件夹
    if not os.path.exists('result'):
        os.makedirs('result')

    # 将结果保存到当前目录下的result文件夹下
    cv2.imwrite(f'result/_{filename}_{ratio_1:.1f}_3.jpg', im_center)  #放大的图像
    cv2.imwrite(f'result/_{filename}_{ratio_1:.1f}_n.jpg', im1_n)
    cv2.imwrite(f'result/_{filename}_{ratio_1:.1f}_b.jpg', im1_b)
    cv2.imwrite(f'result/_{filename}_{ratio_1:.1f}_c.jpg', im1_c)
    cv2.imwrite(f'result/_{filename}_{ratio_2:.1f}_n.jpg', im2_n)
    cv2.imwrite(f'result/_{filename}_{ratio_2:.1f}_b.jpg', im2_b)
    cv2.imwrite(f'result/_{filename}_{ratio_2:.1f}_c.jpg', im2_c)

    # # 显示结果
    # plt.figure(1)
    # plt.subplots_adjust(wspace=0.5, hspace=0.5)
    # plt.subplot(221); plt.imshow(im); plt.title('Orgin')
    # plt.subplot(222); plt.imshow(im1_n); plt.title('Nearest')
    # plt.subplot(223); plt.imshow(im1_b); plt.title('Bilinear')
    # plt.subplot(224); plt.imshow(im1_c); plt.title('Bicubic')

    # plt.figure(2)
    # plt.subplots_adjust(wspace=0.5, hspace=0.5)
    # plt.subplot(221); plt.imshow(im_center); plt.title('Orgin')
    # plt.subplot(222); plt.imshow(im2_n); plt.title('Nearest')
    # plt.subplot(223); plt.imshow(im2_b); plt.title('Bilinear')
    # plt.subplot(224); plt.imshow(im2_c); plt.title('Bicubic')

    # plt.show()