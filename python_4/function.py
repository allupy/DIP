import os 
import cv2
import numpy as np 
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

def bilinear(img_1, n = 2):
    """
    对图像进行两倍上采样，使用双线性插值
    """   
    h_1, w_1 = img_1.shape  # 获取图像尺寸
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
            Q11 = img_1[y1, x1]  
            Q12 = img_1[y1, x2]  
            Q21 = img_1[y2, x1]  
            Q22 = img_1[y2, x2]  
  
            # 计算插值  
            wa = (x2 - x) * (y2 - y)  
            wb = (x - x1) * (y2 - y)  
            wc = (x2 - x) * (y - y1)  
            wd = (x - x1) * (y - y1)  
  
            # 计算目标像素值  
            img_2[i, j] = Q11 * wa + Q12 * wb + Q21 * wc + Q22 * wd
    return img_2

def downsample_image(image):
    """
    对图像进行下采样，支持多通道图像。
    
    :param image: 输入图像
    :return: 下采样后的图像
    """
    downsized_image = image[::2, ::2]  # 对于灰度图，或二维图像进行下采样
    
    return downsized_image

def drawing(image_approx,filename,flag):
    """
    输入：
        image_approx:存有图像金字塔的列表
        filename:图像文件名
        flag:图像金字塔类型,1表示近似金字塔,2表示残差金字塔
    """
    
    if flag == 0:
        string = 'Approximation_Pyramid'
    else:
        string = 'Resdual_Pyramid'

    # 创建 result 文件夹
    if not os.path.exists('result'):
        os.makedirs('result')

    import matplotlib.gridspec as gridspec

    # 绘制 image_approx 列表中的图像
    n_images = len(image_approx)
    widths = [img.shape[1] for img in image_approx]
    heights = [img.shape[0] for img in image_approx]

    # 使用 gridspec 定义子图的布局
    gs = gridspec.GridSpec(1, n_images, width_ratios=widths)  # 使用每个图像的宽度进行大小比例
        
    # 设置整体图形大小
    plt.figure(figsize=(20, 10))
    plt.suptitle(f'result/{filename}_{string}', fontsize=16)  
    for index, approx_img in enumerate(image_approx):
        cv2.imwrite(f'result/_{filename}_{string}_{index}.jpg', approx_img)
        plt.subplot(gs[index])  # 依次填充每个子图
        plt.imshow(approx_img, cmap='gray' if approx_img.ndim == 2 else 'color')
        plt.title(f'{index + 1}')
        plt.axis('off')

    plt.savefig(f'result/{filename}_{string}.jpg')
    plt.show()


def dwt2d(image):   
    """
    对图像进行二维离散小波变换。
    """  
    # 定义滤波器系数 h0
    g0 = np.array([0.0322, -0.0126, -0.0992, 0.2979, 0.8037, 0.4976, -0.0296, -0.0758])
    g1 = np.array([(-1) ** n * g0[7 - n] for n in range(len(g0))])

    h0 = np.array(g0[::-1])  # 低通滤波器
    h1 =np.array(g1[::-1])  # 高通滤波器
    # 行滤波：使用低通和高通滤波器逐行处理
    cA_rows = convolve2d(image, h0[None, :], mode='same')[::2, :] # 低通滤波器
    cD_rows = convolve2d(image, h1[None, :], mode='same')[::2, :]  # 高通滤波器
    # 列滤波：使用低通和高通滤波器处理每一列
    cA = convolve2d(cA_rows, h0[:, None], mode='same')[:, ::2]  # 低通滤波器
    cH = convolve2d(cA_rows, h1[:, None], mode='same')[:, ::2] # 高通滤波器
    cV = convolve2d(cD_rows, h0[:, None], mode='same')[:, ::2]  # 低通滤波器
    cD = convolve2d(cD_rows, h1[:, None], mode='same')[:, ::2]  # 高通滤波器

    return cA, cH, cV, cD

def idwt2d(cA, cH, cV, cD):
    """
    对图像进行二维离散小波反变换。
    """
    rows, cols = cA.shape
    # 初始化用于存储中间结果的数组
    reconstructed_rows_HA = np.zeros((rows*2 , cols ), dtype=float)
    reconstructed_rows_DV= np.zeros((rows*2, cols), dtype=float)
    # 列重构
    for j in range(cols):
        reconstructed_rows_HA[:, j ] = convolve_and_upsample(cA[:, j], 0) + convolve_and_upsample(cH[:, j], 1)
        reconstructed_rows_DV[:, j ] = convolve_and_upsample(cD[:, j], 1) + convolve_and_upsample(cV[:, j], 0)
    # 行重构
    reconstructed_image = np.zeros((rows * 2, cols * 2), dtype=float)
    for i in range(rows * 2):
        reconstructed_image[i, :] = convolve_and_upsample(reconstructed_rows_HA[i, :], 0) + convolve_and_upsample(reconstructed_rows_DV[i, :], 1)
    return reconstructed_image

def convolve_and_upsample(signal, flag):
    """
    对信号进行上采样并应用卷积。
    """
    g0 = np.array([0.0322, -0.0126, -0.0992, 0.2979, 0.8037, 0.4976, -0.0296, -0.0758])
    g1 = np.array([(-1) ** n * g0[7 - n] for n in range(len(g0))])
    # 上采样
    signal = signal.reshape(1, -1)
    #upsampled = bilinear(signal, 2)[1]
    upsampled = (bilinear(signal, 2)[1] + bilinear(signal, 2)[0])
    # 卷积
    if flag == 0:
        filter = g0
        conv_result = np.convolve(upsampled, filter, mode='same')  # 使用 'same' 模式
    else:
        filter = g1
        conv_result = np.convolve(upsampled, filter, mode='same')  # 使用 'same' 模式
    return conv_result

def normalize_image(image):
    """将输入图像归一化到0到1的范围"""
    image_min = np.min(image)
    image_max = np.max(image)
    if image_max == image_min:
        # 防止除以零，返回一个全零的图像
        return np.zeros_like(image)  # 或者 return np.ones_like(image)
    return (image - image_min) / (image_max - image_min)
 
def concatenate_image(cA, cH, cV, cD):    
    AH = np.concatenate([normalize_image(cA), normalize_image(cH)], axis=1)
    VD = np.concatenate([normalize_image(cV), normalize_image(cD)], axis=1)
    dwt1 = np.concatenate([AH, VD], axis=0)
    return dwt1
