import numpy as np
import cv2
from scipy.signal import convolve2d 
def myMedian(image):
    """  
    对给定的图像应用中值滤波。  
  
    参数:  
    image (list of list of int): 输入的灰度图像，表示为二维列表，其中每个元素是像素的灰度值（0-255）。  
  
    返回:  
    list of list of int: 滤波后的图像，表示为二维列表。
      
    """  
    kernel_size = 3 #3×3的卷积核
  
    # 获取图像尺寸  
    rows,cols,channel = image.shape 
  
    # 初始化输出图像  
    filtered_image = np.zeros_like(image)   
  
    # 获取滤波器半径  
    half_size = kernel_size // 2  

    padded_image = np.pad(image, ((half_size, half_size), (half_size, half_size),(0,0)), mode='edge') #复制填充
  
    # 遍历图像的每个像素
    for i in range(rows):  
        for j in range(cols):  
            # 计算邻域内像素的平均值，找到卷积领域 
            region = padded_image[i:i+kernel_size, j:j+kernel_size]  
            # 计算中值  
            # 计算窗口内像素的中值（对每个通道分别计算）  
            filtered_image[i, j] = np.median(region, axis=(0, 1))  
   
    return filtered_image 



def myHisteq(image,n):
    '''
    image:the image input,
    n : the target new grey scale

    '''
    hist, bins = np.histogram(image.flatten(), 256, [0, 256]) # we get the occurrence times(a number,nut  a rate ) of every grey scale
    cdf = hist.cumsum() #cdf[i] 表示灰度级小于或等于 i 的像素总数。

    cdf_normalized = (cdf - cdf.min()) * (n - 1) / (cdf.max() - cdf.min())  # 归一化CDF到[0, n-1]  
    lut = np.round(cdf_normalized).astype(np.uint8)  # lut[i]代表原来灰度为i的变换为lut[i]
    lut = np.round(lut*255.0/(n-1)).astype(np.uint8)
    newimage = cv2.LUT(image, lut)

    return newimage

def myAverage(im):
    """  
    对给定的图像应用均值滤波。  
  
    参数:  
    image (list of list of int): 输入的灰度图像，表示为二维列表，其中每个元素是像素的灰度值（0-255）。  
  
    返回:  
    list of list of int: 滤波后的图像，表示为二维列表。
      
    """  
    kernel_size = 3 #3×3的卷积核
  
    # 获取图像尺寸  
    rows,cols,channel = im.shape 
  
    # 初始化输出图像  
    filtered_image = np.zeros_like(im)   
  
    # 获取滤波器半径  
    half_size = kernel_size // 2  

    padded_image = np.pad(im, ((half_size, half_size), (half_size, half_size),(0,0)), mode='edge') #复制填充
  
    # 遍历图像的每个像素
    for i in range(rows):  
        for j in range(cols):  
            # 计算邻域内像素的平均值，找到卷积领域 
            region = padded_image[i:i+kernel_size, j:j+kernel_size]  
            # 计算均值  
            filtered_image[i, j] = np.median(region)  
  
 
  
    return filtered_image 

def mySharpen(image,alpha):
    """  
    使用拉普拉斯算子对图像进行锐化增强。  
  
    参数:  
    image (numpy.ndarray): 输入的灰度图像（二维数组）。  
 
    返回:  
    numpy.ndarray: 锐化后的图像（二维数组）。  
    """  
    # 构建拉普拉斯核  
 
    laplacian_kernel = np.array([[0, -1, 0],  
                                [-1,  4,-1],  
                                [0, -1, 0]])  
    
    image = image[:, :, 0] # 灰度图，只对一个通道进行操作

  
    # 对图像进行拉普拉斯卷积  
    laplacian = convolve2d(image, laplacian_kernel, mode='same', boundary='fill', fillvalue=0)  #采用0填充
  
    # 将拉普拉斯结果添加到原图像以实现锐化  
    sharpened_image = image + laplacian*alpha  # 中心系数+ 
  
    # 确保结果图像的数据类型与输入图像相同  
    sharpened_image = np.clip(sharpened_image, 0, 255).astype(image.dtype)  
  
    return sharpened_image 

     
