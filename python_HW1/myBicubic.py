import numpy as np


# 定义双三次插值的基函数（通常使用Bicubic的a=-0.5）  
def cubic(r):  
    r = abs(r)  
    r2 = r ** 2  
    r3 = r ** 3  
    if r <= 1:  
        return 1.5 * r3 - 2.5 * r2 + 1  
    elif r < 2:  
        return -0.5 * r3 + 2.5 * r2 - 4 * r + 2  
    else:  
        return 0
        
def bicubic(img_1, n):
    # 假设 img_1 是已加载的图像 (使用 NumPy 或 OpenCV 加载)
    h_1, w_1,channel = img_1.shape  # 获取图像尺寸
    h_2 = int(h_1 * n)  # 计算缩放后的高度
    w_2 = int(w_1 * n)  # 计算缩放后的宽度

    # 创建一个大小为 h_2 x w_2 的空图像 (黑色，灰度图)
    img_2 = np.zeros((h_2, w_2))

    # 下面写立方插值实现代码    
      
    # 遍历目标图像的每个像素  
    for y in range(h_2):  
        for x in range(w_2):  
            # 计算在原图像中的对应浮点坐标  
            src_x = (x + 0.5) /n - 0.5  
            src_y = (y + 0.5) / n - 0.5  
  
            # 边界处理  
            xi = int(np.floor(src_x))  
            yi = int(np.floor(src_y))  
            xi = np.clip(xi, 0, w_1 - 1)  
            yi = np.clip(yi, 0, h_1 - 1)  
  
            # 计算权重 
            u = src_x - xi
            v = src_y - yi
            wx = [cubic(u+1),cubic(u),cubic(1-u),cubic(2-u)] 
            wy = [cubic(v+1),cubic(v),cubic(1-v),cubic(2-v)] 
            # wx = [cubic(src_x - xi - j) for j in range(4)]  
            # wy = [cubic(src_y - yi - k) for k in range(4)]   
  
            # 插值          
            z = 0 
            for i in range(4):  
                for j in range(4):  
                    zi = yi + i  
                    zj = xi + j  
                    zi = np.clip(zi, 0, h_1 - 1)  
                    zj = np.clip(zj, 0, w_1 - 1)  
                    z += img_1[zi, zj, 0] * wx[j] * wy[i]  
  
            # 将插值结果赋值给目标图像的相应位置  
            img_2[y, x] = z 





    return img_2



