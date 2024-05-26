import numpy as np
import matplotlib.pyplot as plt
import math

if __name__ == '__main__':
    pic_path = 'D:\cv_workspace\picture\lenna.png'
    img = plt.imread(pic_path)
    print("image", img)
    # [-4:] -> 从后向前数4位  .png图片在这里的存储格式是0到1的浮点数，所以要扩展到255再计算（优化项）
    if pic_path[-4:] == '.png':
        img = img * 255  # 还是浮点数类型
    img = img.mean(axis=-1)  # 取均值的方法进行灰度化

    # 1、高斯平滑
    # sigma = 1.52  # 高斯平滑时的高斯核参数，标准差，可调
    # 高斯平滑时的高斯核参数，标准差，可调   sigma = 0.5：高斯平滑时的高斯核参数，标准差，用于控制平滑程度，值越大平滑效果越明显。
    sigma = 0.5
    # 高斯核尺寸（推荐）
    dim = 5
    # 存储高斯核，这是数组(5X5全零矩阵)不是列表了
    Gaussian_filter = np.zeros([dim, dim])
    '''
    生成一个序列，这个序列在高斯核计算中用于表示离散的坐标值。 
        range(dim)：这部分代码表示生成一个从 0 到 dim-1 的整数序列，即 [0, 1, 2, ..., dim-1]。
        i-dim//2：对于这个整数序列中的每个元素 i，都执行减法操作 i-dim//2。这里的 dim//2 表示 dim 除以 2 的整数部分，即 dim 的一半。
        因此，整个表达式 [i-dim//2 for i in range(dim)] 的含义是，对于从 0 到 dim-1 的每个整数 i，都计算 i-dim//2 的值，然后将这些值组成一个新的列表。
        * 代表5*5临域内各点与中心点的坐标差值，通过坐标差值反应对中心的影响程度（高斯核中每个位置的权重值），x轴和y轴都是这个结果 。
    '''
    tmp = [i - dim // 2 for i in range(dim)]  # 生成一个序列
    print('生成一个序列，这个序列在高斯核计算中用于表示离散的坐标值:\n', tmp)
    ''' 
    range(dim)                 [0,  1, 2, 3, 4]
       计算： i - dim // 2   例： 2 - 5//2  = 2-2 = 0
    tmp                        [-2, -1, 0, 1, 2] 
    '''

    '''
    高斯平滑水平和垂直方向呈现高斯分布，更突出了中心点在像素平滑后的权重
    ========================================================                             
    |                                    x²  +  y²         |
    |                    1         (-) -------------       |   
    | 二维： G(x,y) = ----------- e         2 σ ²           |
    |                  2 Π σ ²                             |
    ========================================================
    如下： n1 = 1/(2*math.pi*sigma**2) 和 n2 = -1/(2*sigma**2) 用于计算高斯核的两个常数部分：
    n1 = 1/(2*math.pi*sigma**2)：这行代码计算了高斯核的一个【常数部分】，其中 math.pi 是圆周率，sigma 是高斯核的标准差。
         这个常数部分是高斯函数中的一个系数，用于确保高斯核的总和为1。
    n2 = -1/(2*sigma**2)：这行代码计算了高斯核的另一个常数部分，同样也是高斯函数中的一个系数【幂的系数】，也用于确保高斯核的总和为1。
         在高斯核的计算中，这两个常数部分会与高斯函数的指数部分相乘，最终得到高斯核中每个位置的权重值。
         这两个常数部分的计算是高斯平滑算法中的重要一步，它们确保了生成的高斯核满足高斯分布的性质，从而实现了图像的平滑处理。
    '''
    n1 = 1 / (2 * math.pi * sigma ** 2)
    n2 = -1 / (2 * sigma ** 2)
    print('n1、n2分别为', n1, n2)
    '''n1、n2分别为 0.6366197723675814 -2.0'''

    # 遍历高斯核的每个元素(5X5=25个)，计算高斯核的值。
    for i in range(dim):
        for j in range(dim):
            '''
            【计算高斯核中每个位置的权重值】
                tmp[i] 和 tmp[j] =》  这两个变量分别表示高斯核中的行索引和列索引，它们来自于之前生成的序列 tmp = [i-dim//2 for i in range(dim)]，
                                     用于表示离散的坐标值。  -> x y 
                tmp[i] ** 2 + tmp[j] ** 2 =》  这部分计算了离散坐标的【平方和】，用于构建高斯函数的指数部分。 ->  x² + y²
                n2 * (tmp[i] ** 2 + tmp[j] ** 2) =》   其中 n2 是之前计算的高斯核的常数部分， 这部分是高斯函数的【指数部分】
                math.exp(n2 * (tmp[i] ** 2 + tmp[j] ** 2)) =》  这部分使用了 Python 中的 math.exp 函数，计算了【指数部分的指数函数值】
                                                                math.exp(x) 方法返回 e 的 x 次幂（次方）Ex，其中 e = 2.718281... 是自然对数的基数。
                n1 * math.exp(n2 * (tmp[i] ** 2 + tmp[j] ** 2)) =》  最终，将常数部分 n1 与指数函数值相乘，得到了【高斯核中每个位置的权重值】 -> G(x,y)
                这行代码实际上是在根据高斯函数的定义，计算了高斯核中每个位置的权重值，这些权重值将用于后续的图像平滑操作，以实现高斯平滑的效果。
            '''
            Gaussian_filter[i, j] = n1 * math.exp(n2 * (tmp[i] ** 2 + tmp[j] ** 2))
            print('Gaussian_filter为', Gaussian_filter)
            '''
            Gaussian_filter为 [ [7.16421173e-08 2.89024930e-05 2.13562142e-04 2.89024930e-05 7.16421173e-08]
                                [2.89024930e-05 1.16600979e-02 8.61571172e-02 1.16600979e-02 2.89024930e-05]
               5 X 5            [2.13562142e-04 8.61571172e-02 6.36619772e-01 8.61571172e-02 2.13562142e-04]
                                [2.89024930e-05 1.16600979e-02 8.61571172e-02 1.16600979e-02 2.89024930e-05]
                                [7.16421173e-08 2.89024930e-05 2.13562142e-04 2.89024930e-05 7.16421173e-08] ]
            '''

    '''
    【对高斯核进行归一化】，使得高斯核的所有元素之和为1。(优化项)
        归一化是指将一组数据按比例缩放，使其落入特定的范围或者满足特定的要求。在这里，对高斯核进行归一化意味着将高斯核中的所有元素按比例缩放，使它们的总和等于1。
        在图像处理中，对高斯核进行归一化是一种常见的优化方法。通过归一化，可以确保高斯核在进行卷积操作时不会改变图像的亮度和对比度，同时也能够保持图像的整体特征。
        这样做有助于避免图像在进行平滑处理时出现过度模糊或者失真的情况。
        因此，对高斯核进行归一化是一种常见的图像处理优化手段，它有助于确保图像在进行平滑处理时能够保持良好的视觉效果和图像特征。
    '''
    Gaussian_filter = Gaussian_filter / Gaussian_filter.sum()
    print('归一化后 Gaussian_filter为', Gaussian_filter)
    '''
    Gaussian_filter为 [  [6.96247819e-08 2.80886418e-05 2.07548550e-04 2.80886418e-05 6.96247819e-08]
                         [2.80886418e-05 1.13317669e-02 8.37310610e-02 1.13317669e-02 2.80886418e-05]
       5  X  5           [2.07548550e-04 8.37310610e-02 6.18693507e-01 8.37310610e-02 2.07548550e-04]
                         [2.80886418e-05 1.13317669e-02 8.37310610e-02 1.13317669e-02 2.80886418e-05]
                         [6.96247819e-08 2.80886418e-05 2.07548550e-04 2.80886418e-05 6.96247819e-08]  ]
    '''

    dx, dy = img.shape  # 获取图像的尺寸
    img_new = np.zeros(img.shape)  # 存储平滑之后的图像，zeros函数得到的是浮点型数据
    '''
    【步长和填充的计算】
        1. 步长-stride
           用(f,f)的卷积核来卷积一张(h,w)大小的图片，每次移动s个像素 :
                h - f              w - f
           (  --------- + 1  ,   ---------  + 1  )
                  s                  s
           例如： 用(3,3)的卷积核卷积一张(5,5)的图片，每次移动1个像素 :  卷积后原图变成 3 x 3
                 ( 5 - 3 + 1 , 5 - 3 + 1 )  =  ( 3 , 3 )
        2. 填充- pading
           填充p圈, 卷积后的图像大小为：
                h - f              w - f                             h - f + 2p             w - f + 2p
           (  --------- + 1  ,   ---------  + 1  )     =》      ( -------------- + 1  ,   --------------  + 1  )
                  s                  s                                   s                       s
           如果让宽高不变，假设步长为1： 需要填充p圈
                  f - 1
           p = -----------   (或者 除2后向下取整)
                    2
    '''
    tmp = dim // 2  # 计算边界填充时的填充大小
    print('计算边界填充时的填充大小tmp为', tmp)  # 2

    '''
    【对原始图像进行边缘填充】
        np.pad 函数的作用是对数组进行填充，以扩展数组的尺寸。在这里，img 是原始图像，((tmp, tmp), (tmp, tmp)) 是填充的宽度，'constant' 表示填充的方式为常数填充。
        ((tmp, tmp), (tmp, tmp))：这个参数表示在图像的上下左右分别填充 tmp 个像素。这样做的目的是为了在进行卷积操作时，
                                  避免图像边缘的像素无法完全参与卷积，从而保证卷积操作的有效性。
        tmp 的计算规则：在之前的代码中，tmp = dim//2，即 tmp 是高斯核尺寸的一半。这里的 tmp 用于表示边缘填充的宽度，以确保在进行卷积操作时，
                                  图像的边缘像素也能够被充分考虑。
        因此，这段代码的作用是对原始图像进行边缘填充，以便后续的卷积操作能够充分考虑图像边缘的像素，从而保证图像处理的准确性和完整性。
    '''
    img_pad = np.pad(img, ((tmp, tmp), (tmp, tmp)), 'constant')  # 边缘填补
    print('对原始图像进行边缘填充img_pad为', img_pad)
    '''
    [[  0.         0.         0.       ...   0.         0.         0.      ]
     [  0.         0.         0.       ...   0.         0.         0.      ]
     [  0.         0.       162.66667  ... 129.66667    0.         0.      ]
     ...
     [  0.         0.        53.666668 ... 113.333336   0.         0.      ]
     [  0.         0.         0.       ...   0.         0.         0.      ]
     [  0.         0.         0.       ...   0.         0.         0.      ]]
    '''

    #  遍历图像的每个像素，进行高斯平滑操作
    for i in range(dx):
        for j in range(dy):
            '''
            【对图像进行高斯平滑操作】
                img_new[i, j]：表示对新图像的第 i 行、第 j 列的像素进行赋值操作。
                img_pad[i:i + dim, j:j + dim]：这部分代码是从填充后的图像中提取出与高斯核相同大小的区域，以便进行卷积操作。
                                               这个区域的大小为 dim x dim，即与高斯核相同的尺寸。
                                               str[x:y] 从(起始)x位置截取到y(终止)位置，左闭右开区间  str[x:] 从x截取到最后       
                Gaussian_filter：这是之前计算得到的高斯核，用于进行卷积操作。
                img_pad[i:i + dim, j:j + dim] * Gaussian_filter：这部分代码是对提取出的图像区域与高斯核进行逐元素相乘，得到每个位置的加权值。
                        表示对填充后的图像进行切片操作，以提取出与高斯核相同大小的区域。具体来说，img_pad[i:i + dim, j:j + dim] 表示从填充后的图像 img_pad 中
                        提取出一个子区域，该子区域的行范围是从 i 到 i+dim-1，列范围是从 j 到 j+dim-1，即提取出了一个与高斯核相同大小的区域。
                np.sum：这是 NumPy 库中的函数，用于计算数组中所有元素的和。在这里，对相乘得到的加权值进行求和，得到卷积操作的结果。
                最终，将卷积操作的结果赋值给新图像的对应位置，完成了对图像的高斯平滑操作。
                因此，这段代码的作用是对图像进行高斯平滑处理，通过对图像与高斯核进行卷积操作，实现了图像的平滑效果。
            '''
            img_new[i, j] = np.sum(img_pad[i:i + dim, j:j + dim] * Gaussian_filter)
    print('对图像进行高斯平滑操作img_new为', img_new)
    '''
    对图像进行高斯平滑操作卷积后的img_new为 
        [[129.80168522 145.42782059 146.44679108 ... 147.82649141 136.18950389 105.75955711]
         [145.26985493 162.75812109 163.89851995 ... 165.44263602 152.41889533 118.36268145]
         [145.30819669 162.80107861 163.94177847 ... 165.48630207 152.45912397 118.39392147]
         ...
         [ 48.43107854  54.75298069  58.70400538 ... 108.96181835 106.21440301 95.10599934]
         [ 47.96174316  54.6520051   61.63559545 ... 109.18681542 110.45932378 100.28254646]
         [ 42.82614345  48.84789639  55.43053349 ...  97.57231717  99.18952147 90.21082284]]
    '''
    plt.figure(1)
    # 绘制平滑之后的图像，以灰度图像的形式显示
    plt.imshow(img_new.astype(np.uint8), cmap='gray')  # 此时的img_new是255的浮点型数据，强制类型转换才可以，gray灰阶
    # 关闭坐标轴显示
    plt.axis('off')

    # 2、求梯度。以下两个是滤波用的sobel矩阵（检测图像中的水平、垂直和对角边缘）
    # 这部分定义了两个 Sobel 算子的矩阵，分别用于在 x 方向和 y 方向进行边缘检测
    sobel_kernel_x = np.array([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]])
    sobel_kernel_y = np.array([[1, 2, 1],
                               [0, 0, 0],
                               [-1, -2, -1]])
    # 创建用于存储梯度图像的数组
    img_tidu_x = np.zeros(img_new.shape)  # 存储梯度图像   x 方向梯度图像
    img_tidu_y = np.zeros([dx, dy])  # 存储梯度图像   y 方向梯度图像
    img_tidu = np.zeros(img_new.shape)  # 总梯度图像 img_tidu
    '''
    在这段代码中，对图像进行边缘填充时，使用了 np.pad 函数，并传入了参数 ((1, 1), (1, 1))。这个参数表示在图像的上下左右分别填充 1 个像素。
    这里填充 1 个像素的原因是因为 Sobel 算子的矩阵结构决定了需要在图像周围填充一圈像素，以便在进行卷积操作时，能够充分考虑到图像边缘的像素。
    Sobel 算子的矩阵结构是一个 3x3 的矩阵，因此需要在图像周围填充 1 个像素，以确保卷积操作的有效性。
    '''
    img_pad = np.pad(img_new, ((1, 1), (1, 1)), 'constant')  # 边缘填补，根据上面矩阵结构所以写1

    # 这段代码实现了对图像进行Sobel算子边缘检测，计算了图像在x方向和y方向上的梯度，然后根据这两个方向上的梯度值计算了每个像素点的总梯度值。
    for i in range(dx):
        for j in range(dy):
            '''
            进行梯度计算，分别使用 Sobel 算子对图像进行 x 方向和 y 方向的卷积操作，得到了 x 方向梯度图像 img_tidu_x、
            y 方向梯度图像 img_tidu_y，并计算了总梯度图像 img_tidu。
            '''
            # 这行代码计算了图像在x方向上的梯度。首先，从填充后的图像中提取出一个3x3的区域，然后将这个区域与Sobel算子的x方向核进行逐元素相乘，
            # 然后对所有元素求和，得到了x方向上的梯度值。(x方向上的边缘检测- Gx)
            img_tidu_x[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_x)  # x方向
            # 这行代码计算了图像在y方向上的梯度。同样地，从填充后的图像中提取出一个3x3的区域，然后将这个区域与Sobel算子的y方向核进行逐元素相乘，
            # 然后对所有元素求和，得到了y方向上的梯度值。(y方向上的边缘检测- Gy)
            img_tidu_y[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_y)  # y方向
            # 利用x方向和y方向上的梯度值，按照勾股定理计算了每个像素点的总梯度值，即梯度的大小。 总的边缘检测 G = √ (Gx² + Gy²)
            img_tidu[i, j] = np.sqrt(img_tidu_x[i, j] ** 2 + img_tidu_y[i, j] ** 2)  # 总梯度图像
    print('x方向上的梯度值img_tidu_x为', img_tidu_x)
    '''
    x方向上的梯度值img_tidu_x为 [   [ 453.61376228   51.91887675   -1.01036757 ...  -44.35624628  -131.21382315 -424.79790311]
                                 [ 633.74514139   72.53601769   -1.41158754 ...  -61.97024409  -183.31922402 -593.48641851]
                                 [ 651.16130747   74.52752577   -1.45221545 ...  -63.67733402  -188.35657555 -609.7845344 ] 
                                 ... ... ...      ... ... ...   ... ... ...      ... ... ...    ... ... ...   ... ... ...  
    '''
    print('y方向上的梯度值img_tidu_y为', img_tidu_y)
    '''
    y方向上的梯度值img_tidu_y为 [   [-4.53297831e+02 -6.34684617e+02 -6.52950759e+02 ... -6.51638231e+02  -5.88643108e+02 -3.89144258e+02]
                                 [-4.83862810e+01 -6.77480149e+01 -6.96977941e+01 ... -6.95576911e+01  -6.28334152e+01 -4.15383488e+01]
                                 [-1.19904745e-01 -1.65844981e-01 -1.66945983e-01 ... -1.31555935e-01  -1.11390431e-01 -7.09893998e-02]
                                 ... ... ...      ... ... ...   ... ... ...      ... ... ...    ... ... ...   ... ... ...  
    '''
    print('总梯度图像img_tidu为', img_tidu)
    '''
    总梯度图像img_tidu为       [   [641.28337642 636.80462695 652.95154057 ... 653.14612491 603.09018905 576.09592274]
                                 [635.58959747  99.25355099  69.71208702 ...  93.15891557 193.78848252 594.93828536]
                                 [651.16131851  74.5277103    1.46177997 ...  63.67746992 188.35660849 609.78453853]
                                ... ... ...      ... ... ...   ... ... ...      ... ... ...    ... ... ...   ... ... ...  

    '''
    # 这里对梯度图像进行了处理，避免了除零错误，并计算了梯度的角度。 0.00000001值设计很小避免对原数据产生大的影响
    img_tidu_x[img_tidu_x == 0] = 0.00000001
    '''
    计算图像中每个像素点的梯度方向角度。具体来说，它计算了每个像素点的 y 方向梯度与 x 方向梯度的比值，以得到梯度方向的角度。
    angle 中的每个元素将等于 img_tidu_y 中对应位置的元素除以 img_tidu_x 中对应位置的元素,这种逐元素的除法运算在 NumPy 中是合法的，
          它可以用于对矩阵中的每个元素进行相应的数学运算，而不需要额外的循环或者操作
    在图像处理中，梯度方向角度可以用来表示图像中每个像素点的边缘方向，通常用于边缘检测和特征提取。
    '''
    angle = img_tidu_y / img_tidu_x
    print('angle为', angle)
    '''
    angle为 [[-9.99303523e-01 -1.22245445e+01  6.46250712e+02 ...  1.46910139e+01
               4.48613640e+00  9.16069160e-01]
             [-7.63497466e-02 -9.33991375e-01  4.93754670e+01 ...  1.12243694e+00
               3.42754098e-01  6.99903949e-02]
             [-1.84139849e-04 -2.22528495e-03  1.14959514e-01 ...  2.06597743e-03
               5.91380634e-04  1.16417186e-04]
             ... ... ...      ... ... ...   ... ... ...      ... ... ...    ... ... ...  
    '''

    '''
    将计算得到的总梯度图像进行显示，以便观察图像中的边缘信息。
    这段代码实现了对图像进行梯度计算，并使用 Sobel 算子检测了图像中的边缘信息，最终显示了总梯度图像。
    (图中存在一些不需要细小的或者冗余的边缘 -> 非极大值抑制)
    '''
    plt.figure(2)
    plt.imshow(img_tidu.astype(np.uint8), cmap='gray')
    plt.axis('off')

    # 3、非极大值抑制
    # 创建一个与梯度图像大小相同的全零矩阵，用于存储非极大值抑制后的结果。
    img_yizhi = np.zeros(img_tidu.shape)
    # 遍历图像中除去边缘(边框)的每个像素， 没有设置填充 ∴ range(1, dx - 1)
    for i in range(1, dx - 1):
        for j in range(1, dy - 1):
            flag = True  # 在8邻域内是否要抹去做个标记
            # 提取出当前像素的 3x3 邻域矩阵，用于后续的梯度幅值比较。
            temp = img_tidu[i - 1:i + 2, j - 1:j + 2]
            '''
            梯度幅值的8邻域矩阵: 
                    [  [641.28337642 636.80462695 652.95154057], 
                       [635.58959747  99.25355099  69.71208702], 
                       [651.16131851  74.5277103    1.46177997]  ]
            下面4个条件分支： 每个分支下都计算出两个temp值用于和中心像素比较 c>temp1 & c>temp2  => 保留c 否则 抑制c
            '''
            # tanθ <= -1：表示梯度角度θ对应的直线斜率小于等于 - 1，通常对应着近似垂直方向的边缘。
            # ∵ 梯度角度θ为0°~180°  tanθ = y / x =》 ∵ 梯度角度θ为0°~180°  tanθ = y / x =》 y / tanθ =》 x再加上起始点
            if angle[i, j] <= -1:  # 使用线性插值法判断抑制与否
                '''
                temp[0, 1] 和 temp[0, 0] 分别表示当前像素点的上方和左上方的梯度值，angle[i, j] 表示当前像素点的梯度方向角度。
                temp[0, 1] - temp[0, 0] 表示当前像素点上方的梯度值与左上方的梯度值之差
                (temp[0, 1] - temp[0, 0]) / angle[i, j] 表示上方梯度值与左上方梯度值之差除以当前像素点的梯度方向角度的tan值。
                最后加上 temp[0, 1] 根据插值法的原理，需要将计算得到的值加上已知点的值，以得到当前像素点在梯度方向上的期望值。
                '''
                num_1 = (temp[0, 1] - temp[0, 0]) / angle[i, j] + temp[0, 1]
                num_2 = (temp[2, 1] - temp[2, 2]) / angle[i, j] + temp[2, 1]
                '''
                c>temp1 & c>temp2  => 保留c 否则 抑制c
                具体的判断逻辑是根据梯度方向进行插值，然后比较当前像素的梯度值与插值后的值，若不满足条件则将 flag 置为 False。
                '''
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False   # => 抑制
            # tanθ >= 1：表示梯度角度θ对应的直线斜率大于等于1，通常对应着近似水平的边缘。
            elif angle[i, j] >= 1:
                '''
                单线性插值（相似三角形） ：(x,y)为待求点坐标，（x0,y0）（x1,y1）分别为两点坐标
                      y - y0        x - x0                      x1 - x              x - x0 
                     -------   =  --------       ==》     y  =  -------  *  y0  +  --------- y1
                     y1 - y0       x1 - x0                      x1 - x0             x1 - x0 
                     已知    （x0 , y0）     ->   (0, 1)
                            （x1 , y1）     ->    (0,2)
                     tan = 1 / x     = >   x = 1 / tan  即 x - x0 = 1 / tan         剩余  1 - 1/ tan     即  x1 - x = 1 - 1 / tan 
                       y - y0        x - x0                     x1 - x              x - x0 
                     -------   =  --------       ==》     y  =  -------  *  y0  +  --------- y1
                      y1 - y0       x1 - x0                     x1 - x0              x1 - x0 
                    因为 x1 - x0 = 1
                    所以:                         ==》     y  =  (x1 - x) * y0  +  (x - x0 ) * y1
                                                             =  (1 / tan) * temp[0,1]   +   1 / tan * temp[0,2]
                                                             =  (temp[0,2]-temp[0,1]) / ang[i,j] + temp[0,1]
                '''
                num_1 = (temp[0, 2] - temp[0, 1]) / angle[i, j] + temp[0, 1]
                num_2 = (temp[2, 0] - temp[2, 1]) / angle[i, j] + temp[2, 1]
                '''
                   c>temp1 & c>temp2  => 保留c 否则 抑制c
                   具体的判断逻辑是根据梯度方向进行插值，然后比较当前像素的梯度值与插值后的值，若不满足条件则将 flag 置为 False。
                '''
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            # tanθ > 0：表示梯度角度θ对应的直线斜率大于0，梯度方向角度在第一和第三象限， 通常对应着从左下到右上方向的边缘。
            elif angle[i, j] > 0:
                num_1 = (temp[0, 2] - temp[1, 2]) * angle[i, j] + temp[1, 2]
                num_2 = (temp[2, 0] - temp[1, 0]) * angle[i, j] + temp[1, 0]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            # tanθ < 0：表示梯度角度θ对应的直线斜率小于0，梯度方向角度在第二和第四象限，通常对应着从左上到右下方向的边缘。
            elif angle[i, j] < 0:  # -0.9339913749084802
                # 根据梯度方向角度进行插值计算，以确定当前像素点在梯度方向上的期望值。因此，这里的 (temp[1, 0] - temp[0, 0]) * angle[i, j]
                # 是基于线性插值法的原理，而不是在计算角度的正切值。
                num_1 = (temp[1, 0] - temp[0, 0]) * angle[i, j] + temp[1, 0]
                num_2 = (temp[1, 2] - temp[2, 2]) * angle[i, j] + temp[1, 2]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            if flag:
                img_yizhi[i, j] = img_tidu[i, j]
    '''  
        根据梯度方向 angle[i, j] 的不同取值，分别进行线性插值法的判断，以确定是否抑制当前像素。
        将计算得到的极大值抑制图像进行显示。
        (图中将冗余的线去掉，同一个边缘之前可能存在多条线，现在只剩一条最优线)
    '''
    plt.figure(3)
    plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
    plt.axis('off')

    lower_boundary = img_tidu.mean() * 0.5
    high_boundary = lower_boundary * 3  # 这里我设置高阈值是低阈值的三倍
    print('lower_boundary 、 high_boundary分别为', lower_boundary, high_boundary)
    ''' lower_boundary 、 high_boundary分别为 22.03517254942563 66.1055176482769 '''
    zhan = []
    for i in range(1, img_yizhi.shape[0] - 1):  # 外圈边缘不考虑了
        for j in range(1, img_yizhi.shape[1] - 1):
            ''' 
                 大于高阈值的为强边缘，小于低阈值的不是边缘，介于中间的是弱边缘：

                    不是边缘（抑制）        若边缘（是否连接强边缘）         强边缘（保留）
                 ------------------●--------------------------●---------------------------
                          低阈值(lower_boundary)          高阈值(high_boundary)    
                            22.03517254942563             66.1055176482769
            '''
            if img_yizhi[i, j] >= high_boundary:  # 取，一定是边的点   >= 66.1055176482769
                img_yizhi[i, j] = 255  #   目标图对应点标记为强边缘
                zhan.append([i, j])   #   一定是边的点
            elif img_yizhi[i, j] <= lower_boundary:  # 不是边缘->舍(抑制)    <= 22.03517254942563
                img_yizhi[i, j] = 0

    # 4、双阈值检测，连接边缘。遍历所有一定是边的点,查看8邻域是否存在有可能是边的点，进栈
    '''
    8邻域8个点的判断
    基于边缘检测结果的边缘跟踪算法，通常用于从边缘检测结果中提取出连续的边缘。
            （具体来说，它使用了一个栈（stack）数据结构来存储待处理的像素点，然后不断地从栈中取出像素点进行处理，并将其周围的像素点加入栈中，直到栈为空为止。
    算法不断地从栈中取出像素点，对于每个像素点 (temp_1, temp_2)，算法检查其周围的像素点，如果满足一定条件，则将其标记为边缘，并将其周围的像素点加入栈中。
    '''
    while not len(zhan) == 0:
        temp_1, temp_2 = zhan.pop()  # 出栈     temp_1 行 510，temp_2  列  382  ->  zhan[510,382] 为一定是边的点 
        '''
        img_yizhi 是一个二维的图像数组，temp_1 和 temp_2 分别表示待处理像素点的行和列坐标。
        这行代码使用了 Python 中的切片（slice）操作，具体来说，img_yizhi[temp_1 - 1:temp_1 + 2, temp_2 - 1:temp_2 + 2] 表示从 img_yizhi 中提取出
        以 (temp_1, temp_2) 为中心的 3x3 的区域（  ：切片操作 左闭右开区间 ），
              其中 zhan[temp_1,temp_2] 为  ‘一定是边的点’
                  img_yizhi[temp_1,temp_2] 为 3x3区域的中心点(目标图对应点已经标记为255)
        这样做的目的是为了获取待处理像素点周围的像素值，如果周围的像素值在lower_boundary到high_boundary之间(弱边缘)且连接了强边缘(中心点255)，则将此弱边缘点标记为强边缘(255)。
        '''
        a = img_yizhi[temp_1 - 1:temp_1 + 2, temp_2 - 1:temp_2 + 2]
        '''
          [ [  0. 255.   0.], 
            [  0. 255.   0.], 
            [  0.   0.   0.] ]
        '''
        '''
        【边缘标记和边缘跟踪】：根据边缘检测结果和阈值条件，将符合条件的像素点标记为边缘，并将其周围的像素点加入到栈中，以便进行下一轮的边缘跟踪操作。
        if (a[0, 0] < high_boundary) and (a[0, 0] > lower_boundary):
           判断小区域 a 中左上角像素点的灰度值是否在指定的高阈值和低阈值之间。如果满足条件即a[0, 0]为弱边缘，则执行下面的两行代码，否则跳过这个条件块。
                【边缘标记】img_yizhi[temp_1 - 1, temp_2 - 1] = 255 # 这个像素点标记为边缘
                     如果条件满足，即左上角像素点的灰度值在指定的阈值范围内，那么将图像 img_yizhi 中对应的像素点标记为边缘，通常用一个较大的灰度值（比如255）来表示边缘。
                【边缘跟踪】zhan.append([temp_1 - 1, temp_2 - 1]) # 进栈
                     接着，将当前像素点的坐标 [temp_1 - 1, temp_2 - 1] 加入到栈 zhan 中，以便后续的边缘跟踪操作。这样，下一轮处理时，算法将从栈中取出这个像素点进行处理。
        '''
        if (a[0, 0] < high_boundary) and (a[0, 0] > lower_boundary):
            img_yizhi[temp_1 - 1, temp_2 - 1] = 255  
            zhan.append([temp_1 - 1, temp_2 - 1])  # 进栈
        if (a[0, 1] < high_boundary) and (a[0, 1] > lower_boundary):
            img_yizhi[temp_1 - 1, temp_2] = 255
            zhan.append([temp_1 - 1, temp_2])
        if (a[0, 2] < high_boundary) and (a[0, 2] > lower_boundary):
            img_yizhi[temp_1 - 1, temp_2 + 1] = 255
            zhan.append([temp_1 - 1, temp_2 + 1])
        if (a[1, 0] < high_boundary) and (a[1, 0] > lower_boundary):
            img_yizhi[temp_1, temp_2 - 1] = 255
            zhan.append([temp_1, temp_2 - 1])
        if (a[1, 2] < high_boundary) and (a[1, 2] > lower_boundary):
            img_yizhi[temp_1, temp_2 + 1] = 255
            zhan.append([temp_1, temp_2 + 1])
        if (a[2, 0] < high_boundary) and (a[2, 0] > lower_boundary):
            img_yizhi[temp_1 + 1, temp_2 - 1] = 255
            zhan.append([temp_1 + 1, temp_2 - 1])
        if (a[2, 1] < high_boundary) and (a[2, 1] > lower_boundary):
            img_yizhi[temp_1 + 1, temp_2] = 255
            zhan.append([temp_1 + 1, temp_2])
        if (a[2, 2] < high_boundary) and (a[2, 2] > lower_boundary):
            img_yizhi[temp_1 + 1, temp_2 + 1] = 255
            zhan.append([temp_1 + 1, temp_2 + 1])

    ''' 弱边缘已经加入到img_yizhi中，剩余的不大于低阈值(22)和不小于高阈值(66)的点 且不为0 和 不为255（上面若边缘已经记为255） 为0  '''
    for i in range(img_yizhi.shape[0]):
        for j in range(img_yizhi.shape[1]):
            if img_yizhi[i, j] != 0 and img_yizhi[i, j] != 255:
                img_yizhi[i, j] = 0
    "去掉弱的边缘，加强强的边缘"

    # 绘图
    plt.figure(4)
    plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
    plt.axis('off')  # 关闭坐标刻度值
    plt.show()
