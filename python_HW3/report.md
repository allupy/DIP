
# Report of DIP 2024, Homework 3
**PB22061259 刘沛**
## Introduction

这次实验的任务是雷登变换投影重建图象。
具体的难点是在于，数学公式的实现有些复杂，涉及到傅里叶变换、频域滤波、多重积分等的代码实现。

## Method
### Radon Transform
离散情况下的radon transform。
数学上可以写为:
$$g(\rho,\theta)=\sum_{x=0}^{M-1}\sum_{y=0}^{N-1}f(x,y)\delta(x\cos\theta+y\sin\theta-\rho)dxdy$$
注意到这里的$\delta$函数，所以只有当且仅当$$x\cos\theta+y\sin\theta = \rho$$
这样的$\rho$才能对g产生贡献。
而在python代码里，一切取值都是离散的。所以我们可以给出一个列表
>       project[max_dist,180] ,全部初始化为0
其中max_dist是图像的最大距离也就是对角线长度，180是角度的范围，这里的角度步长为1。所以我们有
$$projection[\rho, \theta]=\sum_{x=0}^{M-1}\sum_{y=0}^{N-1}im[x,y]，\rho=x\cos\theta+y\sin\theta+maxdist/2$$
这里我们对$\theta$进行遍历，对于每一个$\theta$，我们对整个矩阵image的每一个像素点都可以计算出对应的$\rho$值（这里利用了numpy当中的矩阵计算和广播机制进行时间复杂度优化），这样我们就得到了一个形状和image一样的映射，map[x,y] = $\rho$,最后将所有映射到$\rho$的像素值加起来，就得到了$\theta$角度下投影矩阵。


实际上，图像矩阵的原中心是左上角，我们首先得先将坐标轴移到中心才能适配上面的数学公式。而且图像矩阵中的im[x,y]，x是行数，y是列数，而我们，转换到笛卡尔坐标系当中的时候，坐标轴的方向是反过来的，行对应的是纵坐标。
具体实现的时候$\rho$由于是列表索引，不能取负数，所以我们需要将其从[-max_dist/2,max_dist/2]映射到[0,max_dist]区间。这里加上最大长度的二分之一。
$$projection[\rho, \theta]=\sum_{x=0}^{M-1}\sum_{y=0}^{N-1}im[x,y]，\rho=(x-M/2)\sin\theta+(y-N/2)\cos\theta+maxdist/2$$

这样就得到了离散情况下的randon transform，接着就可以根据projection来计算sinogram。

### inverse randon transform
#### 从正弦图反投影得到图像
根据前面计算得到的projection，也就是投影矩阵，我们可以根据投影矩阵反投影得到原来的图像f(x,y)。
$$f_\theta(x,y)=g(x\cos\theta+y\sin\theta,\theta)dxdy\\f(x,y)=\sum_{\theta=0}^\pi f_\theta(x,y) $$
在代码中，可以写成$$f[x,y]  = \sum_{\theta=0}^\pi g[\rho,\theta]\\
\rho=(x-M/2)\cos\theta+(y-N/2)\sin\theta+maxdist/2$$
实际上，由于我们拿到手中的投影图没有包含原图像的形状信息，只有ρ和θ的取值范围，所以只能默认图像的形状是正方形，或许在实际应用中，进行投影之前会先预先设定好图像的形状（我觉得正方形很好）。


#### 滤波反投影
实际上就是在反投影之前，先对投影图在频域上进行一个简单的滤波。
$$G(\omega,\theta)= F(\omega\cos\theta,\omega\sin\theta)$$
而
$$f(x,y)=\int_0^{2\pi}\int_0^{+\infty}G\left(\omega,\theta\right)\mathrm{e}^{j2\pi\omega(x\cos\theta+y\sin\theta)}\omega d\omega d\theta \\=\int_0^\pi\left[\int_{-\infty}^{+\infty}\lvert\omega\rvert G\left(\omega,\theta\right)\mathrm{e}^{j2\pi\omega\rho}d\omega\right]_{\rho=x\cos\theta+y\sin\theta}d\theta $$
其实也可以写为：
$$f[x,y]  = \sum_{\theta=0}^\pi g^\prime[\rho,\theta],g^\prime是经过频域滤波的投影图$$

接下来完成频域滤波的工作。

**DFT 离散傅里叶变换**
离散的情况下：
$G[w,\theta]=\sum_{\rho=0}^{\rho_{\max}}g[\rho,\theta]e^{-j\frac{2\pi}{p_{\max}}w\cdot \rho}$，在代码中实现的时候需要注意，需要人为的给出$\omega$的取值范围，这里是$[0,\rho_{\max}]$，没有进行负频率的考虑。
**滤波器**
斜坡加矩形窗滤波器：
$$h(\omega)=\begin{cases}\omega/(M-1)，&\quad0\leq\omega\leq(M-1)\\0,&\quad\text{其他}\end{cases}$$

汉明窗斜坡滤波器,c取0.54：
$$h(\omega)=\begin{cases}[c+(c-1)\cos\frac{2\pi\omega}{M}]\omega/(M-1),&\quad0\leq\omega\leq(M-1)\\0,&\quad\text{其他}\end{cases}$$

则G_filtered[w,θ] = G[w,θ] × h(w)。
接下来对G_filtered进行**逆变换**就可以得到原来的f[x,y]。

## Result
下面的图像是包含正弦图、反投影变换，以及两种滤波反投影的结果。

可以看到，在直接进行反投影变换的时候，所有图像的边缘都出现了非常严重的模糊，观感非常差。
而在滤波反投影之后，图像边缘的模糊程度大大减少，但是出现了其他的问题，背景变得不再平滑而是出现了一些有规律的条纹，而对于这种条纹，加矩形窗的斜坡滤波器的情况明显比汉明窗严重许多，这或许就是振铃效应。
![Local Image](./result/Fig0533(a)(circle).tif_radon_transform.jpg)
矩形窗滤波图像中出现奇怪的十字白色粗条纹，在汉明窗中消失。
![Local Image](./result/Fig0534(a)(ellipse_and_circle).tif_radon_transform.jpg)
情况也相似。
![Local Image](./result/Fig0539(a)(vertical_rectangle).tif_radon_transform.jpg)
这里两种滤波图像都出现了严重的条纹，或许这两种滤波器都**对于直线条比较敏感**，而对于椭圆和圆形的边缘，汉明窗的滤波效果更好。
![Local Image](./result/Fig0539(c)(shepp-logan_phantom).tif_radon_transform.jpg)
这里的汉明窗滤波效果不错，显然对于椭圆和圆形的边缘，汉明窗的滤波效果可以达到比较令人满意的效果。