# Import necessary libraries
import tensorflow as tf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import cv2


# Set up the paths of csv files
path_default = "./datasets/data_07021234/t1_clw_fair_07020145"
path_dataset1 = "./datasets/data_07021234/t1_ccw_good_07021219"
path_dataset2 = "./datasets/data_07021234/t1_ccw_good_07021210"
path_dataset3 = "./datasets/data_07021234/t1_clw_good_07020152"
path_udacity = "./datasets/data_udacity/data"
fullpath_imgs = path_default + "/IMG"
path_models = "./models"


# Load images
def read_img(path_imgs, dir_imgs="/IMG"):
    path_org = path_udacity + dir_imgs
    
    if "07021219" in path_imgs:
        path_org = path_dataset1 + dir_imgs
    elif "07021210" in path_imgs:
        path_org = path_dataset2 + dir_imgs
    elif "07020152" in path_imgs:
        path_org = path_dataset3 + dir_imgs

    fullpath_imgs = "{0}/{1}".format(path_org, path_imgs.split("\\")[-1])
#    print("--------------------")
#    print(fullpath_imgs)
#    print("--------------------")

    img_org = cv2.imread(fullpath_imgs)
    img_fin = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)

    # Color space conversion from BGR to RGB
    return img_fin



## Manipulate images for data augmentation ##
# Horizontally flip an image
def flipHorImg(img):
    
    return cv2.flip(img, 1)


# Apply Gaussian blurring to an image
def blurGauImg(img, ksize=5):
    img = cv2.GaussianBlur(img, (ksize, ksize), 0)
    img = np.clip(img, 0, 255)
    
    return img.astype(np.uint8)


# Modigy the brightness of an image
def modBrigImg(img, coef_lower=0.2, coef_upper=0.75):
    img = img.astype(np.float32)
    coef = np.random.uniform(coef_lower, coef_upper)
    img = img*coef
    np.clip(img, 0, 255)
    
    return img.astype(np.uint8)


# Translate an image to a random direction
def tranMovImg(img, st_ang, rng_x_lower, rng_x_upper, rng_y_lower, rng_y_upper, d_st_ang_per_px):
    nrows, ncols = (img.shape[0], img.shape[1])
    trans_x = np.random.randint(rng_x_lower,rng_x_upper)
    trans_y = np.random.randint(rng_y_lower,rng_y_upper)
    st_ang = st_ang + trans_x * d_st_ang_per_px
    trans_matrix = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    img = cv2.warpAffine(img, trans_matrix, (ncols, nrows))
    
    return img, st_ang


# Add random shadows to an image
def addShadImg(img, dark_lower=0.6, dark_upper=0.85):
    nrows, ncols = (img.shape[0], img.shape[1])
    top_y_l = np.random.random_sample()*nrows
    bot_y_l = np.random.random_sample()*nrows
    bot_y_r = np.random.random_sample()*(nrows-bot_y_l)
    top_y_r = np.random.random_sample()*(nrows-top_y_l)
    
    if np.random.random_sample() <= 0.5:
        bot_y_r = bot_y_l - np.random.random_sample()*bot_y_l
        top_y_r = top_y_l - np.random.random_sample()*top_y_l

    poly = np.asarray([[ [top_y_l,0], [bot_y_l,ncols], [bot_y_r,ncols], [top_y_r,0] ]], dtype=np.int32)
    mask_weight = np.random.uniform(dark_lower, dark_upper)
    org_weight = 1 - mask_weight

    mask = np.copy(img).astype(np.int32)
    cv2.fillPoly(mask, poly, (0,0,0))

    return cv2.addWeighted(img.astype(np.int32), org_weight, mask, mask_weight, 0).astype(np.uint8)


# Data augmentation
def AugDataImg(img, st_ang, prob=1.0):
    img_aug = np.copy(img)

    if np.random.random_sample() <= prob:
        img_aug = flipHorImg(img_aug)
        st_ang = -st_ang

    if np.random.random_sample() <= prob:
        img_aug = modBrigImg(img_aug)

    if np.random.random_sample() <= prob:
        img_aug = addShadImg(img_aug, dark_lower=0.45)

    if np.random.random_sample() <= prob:
        img_aug, st_ang = tranMovImg(img_aug, st_ang, -60, 61, -20, 21, 0.35/100.0)

    return img_aug, st_ang


# Generate image dataset

def genImgData(csv, target_dim, imgtypes, st_col, st_ang_cal, st_ang_thresh=0.05, shuffle=True, batch_size=100, aug_likelihood=0.5, data_aug_pct=0.8, neutral_drop_pct=0.25):
    batch = np.zeros((batch_size, target_dim[0], target_dim[1], target_dim[2]), dtype=np.float32)
    out_stang = np.zeros(batch_size)
    len_csv = len(csv)

    while True:
        k = 0
        while k < batch_size:
            idx = np.random.randint(0, len_csv)
            for type_img, st_calib in zip(imgtypes, st_ang_cal):
                if k >= batch_size:
                    break

                row = csv.iloc[idx]
                st_ang = row[st_col]

                if abs(st_ang) < st_ang_thresh and np.random.random_sample() <= neutral_drop_pct:
                    continue

                st_ang = st_ang + st_calib
                path_imgtype = row[type_img]
                img = read_img(path_imgtype)

                img, st_ang = AugDataImg(img, st_ang, prob=aug_likelihood) if np.random.random_sample() <= data_aug_pct else (img, st_ang)
                batch[k] = img
#                print(st_ang)
                out_stang[k] = st_ang
                k = k + 1

        yield batch, np.clip(out_stang, -1, 1)

