import numpy as np



def nearest(img_1, n):
    # 假设 img_1 是已加载的图像 (使用 NumPy 或 OpenCV 加载)
    h_1, w_1,channel = img_1.shape  # 获取图像尺寸
    h_2 = int(h_1 * n)  # 计算缩放后的高度
    w_2 = int(w_1 * n)  # 计算缩放后的宽度


    # 创建一个大小为 h_2 x w_2 的空图像 (黑色，灰度图)
    img_2 = np.zeros((h_2, w_2))

    # 下面写最近邻插值实现代码

    for i in range(h_2):  
        for j in range(w_2):  
            # 计算在原图中的对应位置（浮点数）  
            src_x = (j + 0.5) / n - 0.5  # 中心对齐，调整偏移  
            src_y = (i + 0.5) / n - 0.5  
  
            # 向下取整找到最近的像素索引  
            src_x_int = int(np.floor(src_x))  
            src_y_int = int(np.floor(src_y))  
  
            # 确保索引在图像边界内  
            src_x_int = np.clip(src_x_int, 0, w_1 - 1)  
            src_y_int = np.clip(src_y_int, 0, h_1 - 1)  
  
            # 将原图像的像素值复制到目标图像  
            img_2[i, j] = img_1[src_y_int, src_x_int,0]  
  
    return img_2 






