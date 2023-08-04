from skimage.feature import hog
import matplotlib.pyplot as plt
import cv2
import numpy as np


# 可视化HOG特征图——————————————————————————————————————————————————
def visual_hog():
    #  读入图像
    img = cv2.imread('./img/Walking.jpg')
    image1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(1, 2, 1)
    plt.imshow(image1)

    #  转化为灰度图
    image2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #  HOG特征
    ft, hogimage = hog(image2,  # 输入图像
                       orientations=9,  # 将180°划分为9个bin，20°一个
                       pixels_per_cell=(8, 8),  # 每个cell有8*8=64个像素点
                       cells_per_block=(8, 8),  # 每个block中有8*8=64个cell
                       block_norm='L2-Hys',  # 块向量归一化 str {‘L1’, ‘L1-sqrt’, ‘L2’, ‘L2-Hys’}
                       transform_sqrt=True,  # gamma归一化
                       feature_vector=True,  # 转化为一维向量输出
                       visualize=True)  # 输出HOG图像
    plt.subplot(1, 2, 2)
    plt.imshow(hogimage)
    plt.show()


# 可视化循环采样
def visual_sampling():
    img_path = r'D:\Tracking\Datasets\OTB100\Basketball\img\0001.jpg'
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    gt_state = np.array([198, 214, 34, 81])

    # 循环位移样本
    tl = gt_state[:2] - gt_state[2:] // 2
    br = gt_state[:2] + gt_state[2:] + gt_state[2:] // 2
    im_crop = img[tl[1]:br[1], tl[0]:br[0]]

    plt.subplot(1, 2, 1)
    plt.axis('off')
    plt.imshow(im_crop)

    im_crop1 = np.roll(im_crop, -30, axis=0)
    plt.subplot(1, 2, 2)
    plt.axis('off')
    plt.imshow(im_crop1)
    plt.show()


if __name__ == "__main__":
    # visual_hog()
    visual_sampling()
