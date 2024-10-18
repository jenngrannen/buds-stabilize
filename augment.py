import os
import xml.etree.cElementTree as ET
from xml.dom import minidom
import random
import colorsys
import numpy as np
import cv2
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage

sometimes = lambda aug: iaa.Sometimes(0.5, aug)

KPT_AUGS = [
    iaa.LinearContrast((0.95, 1.05), per_channel=0.25),
    iaa.Add((-10, 10), per_channel=False),
    iaa.GammaContrast((0.95, 1.05)),
    iaa.GaussianBlur(sigma=(0.0, 0.6)),
    iaa.MultiplySaturation((0.95, 1.05)),
    iaa.AdditiveGaussianNoise(scale=(0, 0.0125*255)),
    iaa.flip.Flipud(0.5),
    sometimes(iaa.Affine(
                scale={"x": (1.0, 1.2), "y": (1.0, 1.2)}, # scale images to 80-120% of their size, individually per axis
                translate_percent={"x": (-0.08, 0.08), "y": (-0.08, 0.08)}, # translate by -20 to +20 percent (per axis)
                rotate=(-15, 15), # rotate by -45 to +45 degrees
                shear=(-8, 8), # shear by -16 to +16 degrees
                order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                cval=(0, 20), # if mode is constant, use a cval between 0 and 255
                mode=['constant',  'edge']
                #mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            ))
    ]

seq_kpts = iaa.Sequential(KPT_AUGS, random_order=True)

def augment(img, keypoints, img_dir, output_keypoints_dir, new_idx, show=False):
    seq = seq_kpts
    kps = [Keypoint(x, y) for x, y in keypoints]
    kps = KeypointsOnImage(kps, shape=img.shape)
    img_aug, kps_aug = seq(image=img, keypoints=kps)
    vis_img_aug = img_aug.copy()
    kps_aug = kps_aug.to_xy_array().astype(int)
    #kps_aug = list(kps_aug)
    #kps_aug =  np.array(kps_aug)
    #kps_aug = np.array(kps_aug.to_list() + [cls])
    for i, (u,v) in enumerate(kps_aug[:-1]):
        (r, g, b) = colorsys.hsv_to_rgb(float(i)/keypoints.shape[0], 1.0, 1.0)
        R, G, B = int(255 * r), int(255 * g), int(255 * b)
        cv2.circle(vis_img_aug,(u,v),4,(R,G,B), -1)
    if show:
        cv2.imshow("img", vis_img_aug)
        cv2.waitKey(0)

    cv2.imwrite(os.path.join(img_dir, "%05d.jpg"%new_idx), img_aug)
    np.save(os.path.join(keypoints_dir, "%05d.npy"%new_idx), kps_aug)

if __name__ == '__main__':
    keypoints_dir = 'real_data_train/keypoints'
    img_dir = 'real_data_train/images'

    orig_len = len(os.listdir(img_dir))
    idx = orig_len
    num_augs_per_img = 8
    for i in range(orig_len):
        print(i, orig_len)
        img = cv2.imread(os.path.join(img_dir, '%05d.jpg'%i))
        kpts = np.load(os.path.join(keypoints_dir, '%05d.npy'%i), allow_pickle=True)
        for _ in range(num_augs_per_img):
            augment(img, kpts, img_dir, keypoints_dir, idx+i, show=False)
            idx += 1
        idx -= 1
