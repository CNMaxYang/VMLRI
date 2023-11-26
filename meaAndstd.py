import numpy as np
import cv2
import os
import random


# def cutimg(img):
#     img = img[[not np.all(img[i] < 10) for i in range(img.shape[0])], :, :]
#     img = img[:, [not np.all(img[:, j] < 10) for j in range(img.shape[1])], :]
#     return img
#
# path = '../pretraindata1'
# save_path = '../pretraindata'
# for i in os.listdir(path):
#     img = cv2.imread(os.path.join(path, i))
#     img = cutimg(img)
#     img = cv2.resize(img, (512, 512))
#     cv2.imwrite(os.path.join(save_path, i), img)





means, stdevs = [], []
img_list = []
path = '../pretraindata'

imgs_path_list = os.listdir(path)
len_ = len(imgs_path_list)
i = 0
for item in imgs_path_list:
    img = cv2.imread(os.path.join(path, item))
    img = img[:, :, :, np.newaxis]
    img_list.append(img)
    i += 1
    print(i, '/', len_)


imgs = np.concatenate(img_list, axis=3)
imgs = imgs.astype(np.float32) / 255.
for i in range(3):
    pixels = imgs[:, :, i, :].ravel()
    means.append(np.mean(pixels))
    stdevs.append(np.std(pixels))

print('normMean = {}'.format(means))
print('normStd = {}'.format(stdevs))

# APTOS+MESSIDOR2
# normMean = [0.07919538, 0.25130245, 0.48584846]
# normStd = [0.07214867, 0.14223132, 0.2546199]



# means, stdevs = [], []
# img_list = []
# path_n = '../dataset/EyePACSrDR/0v'
# path_p = '../dataset/EyePACSrDR/1v'
#
# imgs_path_list = os.listdir(path_p)
# selected_imgs = random.sample(imgs_path_list, k=2400)
# len_ = len(selected_imgs)
# i = 0
# for item in selected_imgs:
#     img = cv2.imread(os.path.join(path_p, item))
#     img = cv2.resize(img, (299, 299))
#     img = img[:, :, np.newaxis]
#     img_list.append(img)
#     i += 1
#     print(i, '/', len_)
#
# imgs_path_list = os.listdir(path_n)
# selected_imgs = random.sample(imgs_path_list, k=10000)
# len_ = len(selected_imgs)
# i = 0
# for item in selected_imgs:
#     img = cv2.imread(os.path.join(path_n, item))
#     img = cv2.resize(img, (299, 299))
#     img = img[:, :, np.newaxis]
#     img_list.append(img)
#     i += 1
#     print(i, '/', len_)
#
# imgs = np.concatenate(img_list, axis=2)
# imgs = imgs.astype(np.float32) / 255.
# pixels = imgs[:, :, i, :].ravel()
# means.append(np.mean(pixels))
# stdevs.append(np.std(pixels))
#
# print('normMean = {}'.format(means))
# print('normStd = {}'.format(stdevs))

