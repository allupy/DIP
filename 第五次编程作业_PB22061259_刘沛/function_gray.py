import numpy as np

def create_flat_disk_se(radius):
    diameter = 2 * radius + 1  # 计算直径
    se = np.zeros((diameter, diameter), dtype=np.uint8)  # 创建全零数组

    # 计算圆心
    center = radius

    for i in range(diameter):
        for j in range(diameter):
            # 计算当前像素到圆心的距离
            if (i - center) ** 2 + (j - center) ** 2 <= radius ** 2:
                se[i, j] = 1  # 设置为1

    return se

def erode(image, kernel):
    # 获取图像及其尺寸
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # 计算与膨胀相同的尺寸
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    # 使用255填充图像
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=255)
    # 创建输出图像
    eroded_image = np.zeros_like(image)

    # 执行腐蚀操作  
    for i in range(image_height):
        for j in range(image_width):
            # 提取当前区域
            region = padded_image[i:i + kernel_height, j:j + kernel_width]
            region_value = region * kernel
            eroded_image[i, j] = np.min(region_value[region_value > 0])

    return eroded_image

def dilate(image, kernel):
    # 获取图像及其尺寸
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # 计算与腐蚀相同的尺寸
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    # 使用零填充图像
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)
    # 创建输出图像
    dilated_image = np.zeros_like(image)

    # 执行膨胀操作
    for i in range(image_height):
        for j in range(image_width):
            # 提取当前区域
            region = padded_image[i:i + kernel_height, j:j + kernel_width]
            dilated_image[i, j] = np.max(region*kernel)
            
    return dilated_image

def opening(image, kernel):
    # 先腐蚀再膨胀
    eroded_image = erode(image, kernel) 
    dilated_image = dilate(eroded_image, kernel)
    return dilated_image

def closing(image, kernel):
    # 先膨胀再腐蚀
    dilated_image = dilate(image, kernel)
    eroded_image = erode(dilated_image, kernel)
    return eroded_image

def create_flat_disk_se(radius):
    diameter = 2 * radius + 1  # 计算直径
    se = np.zeros((diameter, diameter), dtype=np.uint8)  # 创建全零数组

    # 计算圆心
    center = radius

    for i in range(diameter):
        for j in range(diameter):
            # 计算当前像素到圆心的距离
            if (i - center) ** 2 + (j - center) ** 2 <= radius ** 2:
                se[i, j] = 1  # 设置为1

    return se