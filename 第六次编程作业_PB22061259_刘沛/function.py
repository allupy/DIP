import cv2
import numpy as np

def HoughTransform(img, threshold, minLineLength, maxLineGap):
    """
    args :
        img : 输入的二值图像
        lines：输出的极坐标来表示直线
        minLineLength：最小直线长度，比这个短的线都会被忽略。
        maxLineGap：最大间隔，如果小于此值，这两条直线 就被看成是一条直线。                      
    return :
        Hough空间图像 
    """
    rho = 1  # 步长

    h,w = img.shape #图像的高和宽
    rho_range = int((h**2 + w**2)**0.5)  # 极坐标范围

    print(rho_range)
    hough = np.zeros((rho_range, 180), dtype=np.uint64)  # 极坐标图像
    
    # 计算坐标网格
    x = np.arange(h) 
    y = np.arange(w) 
    x, y = np.meshgrid(x, y)
    
    # 霍夫变换


    for angle in range(180):
        theta = angle * np.pi / 180  # 角度转换为弧度
        rho =(x * np.cos(theta) + y* np.sin(theta) ).astype(np.int64)  # 计算极坐标
        # 对 rho 应用限制
        valid_rho_indices = np.where((rho >= 0) & (rho <= rho_range))
        
        # 仅对符合条件的 rho 进行累加
        np.add.at(hough, (rho[valid_rho_indices], angle), img[x[valid_rho_indices], y[valid_rho_indices]]) 
    



    return hough
    
def drawLines(edges, hough, threshold):
    """
    args :
        img : 输入的二值图像  
        hough : 阈值化过的霍夫空间图像
        threshold：交点数量的阈值，只有获得足够交点的极坐标点才被看成是直线
    return :
        输出的直线图像
    """
    # 设置最大间隔
    max_gap = 100
     # 阈值化
    hough[hough < threshold*255] = 0  # 小于阈值的点设置为 0

    h, w = edges.shape  # 图像的高和宽

    result = np.zeros((h, w), dtype=np.uint8)  # 创建空白图像

    # 生成坐标网格
    i_indices, j_indices = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')  
    
    rhos ,thetas =  np.where(hough > 0)

    for angle, rho in zip(thetas, rhos):
        # 设置条件并更新 result
        # 计算 rho
        theta = angle * np.pi / 180  # 角度转换为弧度
        rho_img = (i_indices * np.cos(theta) + j_indices * np.sin(theta)).astype(np.int32)

        result[rho_img == rho] = edges[rho_img == rho]  # 找到交点，将其设置为 255

        # points = np.argwhere(rho_img == rho) #在这条直线上的全部像素点的坐标
        # for i in range(len(points) - 1):
        #     start = points[i]
        #     if result[start[0], start[1]] == 0: #则跳过
        #         continue
        #     end = points[max_gap]

        #     # 如果距离小于 or 等于 max_gap，插入连接
        #     if result[end[0], end[1]] == 1:  #两点间所有点可以连接
        #         result[start[0]:end[0], start[1]:end[1]] = 255  # 将直线上的点设置为 255
        #         i += max_gap  # 跳过间隔内的点

    return result

def drawLines_with_gap(edges, hough, threshold, max_gap):
    """
    img : 输入的二值图像  
    hough : 阈值化过的霍夫空间图像
    threshold：交点数量的阈值，只有获得足够交点的极坐标点才被看成是直线
    max_gap：直线之间的最大间隔
    return :输出的直线图像
    """


     # 阈值化
    hough[hough < threshold*255] = 0  # 小于阈值的点设置为 0

    h, w = edges.shape  # 图像的高和宽

    result = np.zeros((h, w), dtype=np.uint8)  # 创建空白图像

    # 生成坐标网格
    i_indices, j_indices = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')  
    
    rhos ,thetas =  np.where(hough > 0)

    for angle, rho in zip(thetas, rhos):
        # 设置条件并更新 result
        # 计算 rho
        theta = angle * np.pi / 180  # 角度转换为弧度
        rho_img = (i_indices * np.cos(theta) + j_indices * np.sin(theta)).astype(np.int32)

        result[rho_img == rho] = edges[rho_img == rho]  # 找到交点，将其设置为 255

        points = np.argwhere(rho_img == rho) #在这条直线上的全部像素点的坐标
        for i in range(len(points) - 1):
            start = points[i] #第一个黑点

            if result[start[0], start[1]] == 255: #则跳过 
                continue
            end = min(max_gap, len(points)-i-1) #最后一个黑点

            for j in range(end): #从第二个黑点开始，到最后一个黑点
                if result[points[i+j][0], points[i+j][1]] == 255: #如果中间有白点，则跳过
                    flag =1
                    break
                else:
                    flag = 0

            if flag == 1 : #可以连接
                for j in range(end): 
                    result[points[i+j][0], points[i+j][1]] = 255 #将中间的白点设置为255
                i = i+max_gap
            if flag == 0: #不能连接 
                 i = i+max_gap

    return result
    
def basic_threshold(img,delta):
    """
    全局阈值化
    img : 输入的图像
    T : 初始阈值
    delta : 阈值的精确度
    return : 二值化后的图像
    """

    #初始化T
    T = np.mean(img)
  
    #分割
    while True:
        img_low = np.mean(img[img<=T])
        img_high = np.mean(img[img>T])
        T_new = (img_low + img_high)/2
        if abs(T_new - T) < delta:
            break
        T = T_new
    result = np.zeros_like(img)
    result[img>T] = 255
    print("基本阈值：",T)

    return result

def Otus(img):
    sigmoid = [] #类间方差列表
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    hist = hist/(img.shape[0]*img.shape[1])
    mg = np.sum(i*hist[i] for i in range(256)) #累积均值
    print(mg)
    for k in range (256):
        P1 = np.sum(hist[i] for i in range(k))
        P2 = 1-P1
        mk = np.sum(i*hist[i] for i in range(k)) #累积均值
        if (P1*P2==0):
            sigmoid.append(0) 
        else:
            sigmoid.append((mg*P1 - mk)**2/(P1*P2))

    T = np.argmax(sigmoid) #找到最大值对应的灰度级
    print(T)
    result = np.zeros(img.shape, np.uint8)
    result[img >= T] = 255
    return result

def local_threshold(img, block_size:tuple):
    """
    局部阈值分割
    block_size: 块数量(宽分割数量，高分割数量)
    """
    height, width = img.shape
    block_height = height // block_size[1]
    block_width = width // block_size[0]
    result = np.zeros((height, width), dtype=np.uint8)
    for i in range(block_size[1]):
        for j in range(block_size[0]):
            block = img[i*block_height:(i+1)*block_height, j*block_width:(j+1)*block_width]
            block_result = Otus(block)
            result[i*block_height:(i+1)*block_height, j*block_width:(j+1)*block_width] = block_result
    return result