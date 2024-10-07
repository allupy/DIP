import numpy as np

def bilinear(img_1, n):
    # 假设 img_1 是已加载的图像 (使用 NumPy 或 OpenCV 加载)
    h_1, w_1,channel = img_1.shape  # 获取图像尺寸
    h_2 = int(h_1 * n)  # 计算缩放后的高度
    w_2 = int(w_1 * n)  # 计算缩放后的宽度

    # 创建一个大小为 h_2 x w_2 的空图像 (黑色，灰度图)
    img_2 = np.zeros((h_2, w_2))

    # 下面写双线性插值实现代码
    for i in range(h_2):  
        for j in range(w_2):  
            # 计算在原图中的对应位置（浮点数）  
            x = (j + 0.5) / n - 0.5  # 中心对齐，调整偏移  
            y = (i + 0.5) / n - 0.5 

            # 找到周围的四个点  
            x1 = int(np.floor(x))  
            x2 = min(x1 + 1, w_1 - 1)  # 防止索引越界  
            y1 = int(np.floor(y))  
            y2 = min(y1 + 1, h_1 - 1)  # 防止索引越界  
  
            # 提取这四个点的像素值  
            Q11 = img_1[y1, x1,0]  
            Q12 = img_1[y1, x2,0]  
            Q21 = img_1[y2, x1,0]  
            Q22 = img_1[y2, x2,0]  
  
            # 计算插值  
            wa = (x2 - x) * (y2 - y)  
            wb = (x - x1) * (y2 - y)  
            wc = (x2 - x) * (y - y1)  
            wd = (x - x1) * (y - y1)  
  
            # 计算目标像素值  
            img_2[i, j] = Q11 * wa + Q12 * wb + Q21 * wc + Q22 * wd




    return img_2
