import os  
import numpy as np
import cv2
import matplotlib.gridspec as gridspec
from scipy.ndimage import rotate

def find_files_recursive(root_dir, extensions, max_level=float('inf')):  
    """  
    递归查找指定目录下的文件，直到达到指定的最大层级或找到所有匹配的文件。  
  
    :param root_dir: 根目录  
    :param extensions: 一个包含文件扩展名的列表，如 ['.txt', '.csv', '.nii']  
    :param max_level: 最大搜索层级，默认为无穷大  
    :return: 所有找到的文件列表  
    """  
    found_files = [] 

  
    def search_files(path, current_level):  
        nonlocal found_files  
        for item in os.listdir(path):  
            item_path = os.path.join(path, item)  
            #如果是文件且扩展名在列表中，则添加到列表中  
            if os.path.isfile(item_path) and os.path.splitext(item_path)[1].lower() in extensions:  
                found_files.append(item_path)  
            #如果是目录且未达到最大层级，则递归搜索  
            elif os.path.isdir(item_path) and current_level < max_level:  
                search_files(item_path, current_level + 1)  
  
    search_files(root_dir, 0)  
    return found_files  


def transform(image):

    #image= np.pad(im, 10, mode='constant', constant_values=0)
    
    M, N = image.shape
    angles = np.arange(0, 180, 1)  # 角度数组
    max_dist = int(np.sqrt(M**2 + N**2))
    projection = np.zeros((max_dist, len(angles)))

    # 计算坐标网格
    x = np.arange(M)
    y = np.arange(N)
    x, y = np.meshgrid(x, y)

    for i, angle in enumerate(angles):
        theta = np.radians(angle)
        # 计算 rho
        rho = (y - N / 2) * np.cos(theta) + (x - M / 2) * np.sin(theta) #矩阵中x是行，放到坐标系下要交换
        rho = np.clip(np.round(rho) + max_dist // 2, 0, max_dist - 1).astype(int)  # 确保 rho 在有效范围内
        # 累加投影值
        np.add.at(projection[:, i], rho, image[x, y])

    return projection



def inverse_radon_transform(projection):
    """
    实现 Radon 变换的逆变换

    :param projection: 输入的投影图像
    :return: 逆变换结果
    """
    #projection= np.pad(projection, 10, mode='constant', constant_values=0)
    max_dist, num_angles = projection.shape
    M = N = int(np.sqrt(max_dist**2 / 2))  # 计算图像大小
    image = np.zeros((M, N))

    # 计算坐标网格
    x, y = np.meshgrid(np.arange(M), np.arange(N)) 

    # 遍历每个角度进行逆变换
    for angle in range(num_angles):
        theta = np.radians(angle)  # 转换为弧度
        # 计算 rho 
        rho = (y - N / 2) * np.cos(theta) + (x - M / 2) * np.sin(theta) + max_dist // 2 #从矩阵变成坐标系，行为y轴，列为x轴

        rho = np.clip(np.round(rho), 0, max_dist - 1).astype(int)  # 确保 rho 在有效范围内
        
        # 累加投影值
        image += projection[rho, angle]  # 利用NumPy的广播机制进行加法

    return image.T #把坐标系变成矩阵


def dft(g,diraction):
    """
    实现离散傅里叶变换,二维函数的一维变换
    :param g: 输入的二维矩阵
    :param diraction: 变换方向，1表示正变换，0表示逆变换
    :return: 变换结果
    """
    M, N = g.shape
    G = np.zeros((M, N), dtype=complex)
    #计算G[Ω, θ],离散傅里叶变换
    # 生成频率矩阵
    m = np.arange(M)          # 0到M-1的数组
    n = np.arange(N)          # 0到N-1的数组

    # 计算重复的指数部分
    if diraction == 1:
        exponent = -1j * (2 * np.pi / M) * m[:, np.newaxis] * n
    else:
        exponent = 1j * (2 * np.pi / M) * m[:, np.newaxis] * n

    # 使用矩阵运算计算 DFT
    G = np.exp(exponent)* g
    return G

def filter_transform(projection, filter_type):  
    """
    执行反投影以重建图像
    :param projection: 输入投影数据，形状为 (num_rho, num_angles)
    :param filter_type: 过滤器类型，0表示斜坡滤波，1表示汉明窗滤波
    :return: 重建出的图像
    """
    max_dist, num_angles = projection.shape

    # 计算G[Ω, θ],离散傅里叶变换
    G = np.fft.fft(projection, axis=0)
    
    # 频域滤波
    wG = np.zeros((max_dist, num_angles), dtype=complex)  # 滤波后的矩阵, 注意这里是复数

    # 预计算频率
    w = np.fft.fftfreq(max_dist)

    if filter_type == 0:
        # 斜坡滤波
        H = abs(w)
        for e in range(num_angles):
            wG[:, e] = G[:, e] * H  # 直接应用滤波器
    else:
        # 汉明窗滤波
        coeff = 2 * np.pi / max_dist  # 预计算常量
        hamming_window = 0.54 + 0.46 * np.cos(coeff * np.arange(max_dist))
        for e in range(num_angles):
            wG[:, e] = G[:, e] * hamming_window * abs(w)

    # 回到空域
    projection_filtered = np.fft.ifft(wG, axis=0)
    
    # 计算逆变换
    image = inverse_radon_transform(np.real(projection_filtered))
    return image

