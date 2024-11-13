import numpy as np

def erode(img, kernel):
    # 获取图像和结构元素的尺寸
    kernel = kernel*255 # 结构元素的值需要调整为0-255
    img_h, img_w = img.shape
    kernel_h, kernel_w = kernel.shape
    
    # 创建输出图像
    eroded_img = np.ones_like(img) * 0 # 以黑色背景初始化

    # 执行腐蚀操作
    for i in range(img_h-kernel_h+1):
        for j in range(img_w-kernel_w+1):
            # 提取当前区域
            region = img[i:i + kernel_h , j:j + kernel_w ] #包含起始但不包含终点
            # 如果结构元素与区域完全匹配，则将中心像素设置为1（白色）；否则为0（黑色），即腐蚀
            #print(region.shape, kernel.shape)

            if np.array_equal(region, kernel):
                eroded_img[i, j] = 255
                # print(i,j)

    return eroded_img

from scipy.signal import convolve2d

def erosion(img, kernel):

    eroded_img = np.zeros_like(img)    
    convolved = convolve2d(img, kernel, mode='same')

    kernel_sum = np.sum(kernel)*255

    eroded_img[convolved == kernel_sum] = 255

    return eroded_img
from scipy.signal import convolve2d

def dilation(img, kernel):
    # 获取结构元素的尺寸
    kernel_h, kernel_w = kernel.shape
    # 创建一个输出图像，初始化为0（黑色）
    dilated_img = np.zeros_like(img)
    
    # 进行卷积运算
    convolved = convolve2d(img, kernel, mode='same')

    # 判断卷积结果
    # 如果区域有任何一个像素为255，则设置为255
    dilated_img[convolved > 0] = 255

    return dilated_img


def reconstruction_opening(input,target, kernel):
     # 复制输入图像作为初始图像
    input = np.copy(input)
    # 重复膨胀操作直到无法再膨胀
    while True:
        # 进行膨胀
        dilated = dilation(input, kernel)   
        # 使用逻辑与操作将膨胀结果限制在目标图像内
        output = np.logical_and(dilated, target) #全部是0或者1
        # 检查是否达到稳定状态，若膨胀后的图像与当前图像相同，则停止
        if np.array_equal(input,output):
            break
        input = output
    return output*255 #统一二值化的定义


