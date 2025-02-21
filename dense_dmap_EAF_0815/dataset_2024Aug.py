from configparser import Interpolation
import pandas as pd
import torch.utils.data as data
import torch
import numpy as np
import os
from skimage.transform import rotate
from skimage import transform
from losses import expand_as_one_hot
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import torch
import cv2
import h5py
from scipy import signal

def syn_air(r):
    patch = np.zeros([2*r, 2*r])
    for x in range(2*r):
        for y in range(2*r):
            if (x - r)**2 + (y - r)**2 < r**2:
                patch[x][y] = 1
    return patch


def generate_dmap(h,w,rh,rw):
    dmap = np.zeros([h,w])
    for i in range(h):
        for j in range(w):
            dmap[i][j] = np.sqrt(((i - h//2)*rh)**2 + ((j - w//2)*rw)**2)
    return dmap

class Dataset(data.Dataset):
    def __init__(self, path_csv, opt, aug = True, normalize = True, patch=False):
        super(Dataset, self).__init__()
        self.df = pd.read_csv(path_csv)
        self.num_img = self.df.shape[0]
        self.aug = aug
        self.args = opt
        self.patch = patch
        self.normalize = normalize
        self.air_temp = {}
        self.air_temp[16] = syn_air(16)
        self.air_temp[32] = syn_air(32)
        #self.distance_map = generate_dmap(512)
        
        #self.air_temp[64] = syn_air(64)

    def augment(self, r, image, label):
        # label[label == 5] = 0
        # label[label == 4] = 3
        # rotation
        H, W = image.shape
        if r[0] < 0.4:
            # angle = np.random.randint(-30, 30)
            # M = cv2.getRotationMatrix2D((H//2, W//2), angle, 1.0)
            # image = cv2.warpAffine(image, M, (W, H), 1, 0)
            # label = cv2.warpAffine(label, M, (W, H), 0, 0)
            k = np.random.randint(0, 3)
            image = np.rot90(image, k)
            label = np.rot90(label, k)
        if r[1] < 0.25:
            H, W = image.shape
            k1, k3 = r[7]/10, r[8]/10
            k2, k4 = 1 - r[5]/10, 1 - r[6]/10
            image = cv2.resize(image[int(k1 * H): int(k2 * H), int(k3 * W): int(k4 * W)], (W, H),
                               interpolation=cv2.INTER_CUBIC)
            label = cv2.resize(label[int(k1 * H): int(k2 * H), int(k3 * W): int(k4 * W)], (W, H),
                               interpolation=cv2.INTER_NEAREST)
        elif r[1] > 0.75:
            tmp_img = np.zeros([H, W])
            tmp_label = np.zeros([H, W])
            Hn, Wn = int(H * np.random.uniform(0.7, 1)), int(H * np.random.uniform(0.7, 1))
            x, y = np.random.randint(H - Hn), np.random.randint(W - Wn)
            tmp_img[x:x+Hn, y:y+Wn] = cv2.resize(image, (Wn, Hn), interpolation = cv2.INTER_NEAREST)
            tmp_label[x:x+Hn, y:y+Wn] = cv2.resize(label, (Wn, Hn), interpolation = cv2.INTER_NEAREST)
            image = tmp_img
            label = tmp_label
        # horizonal flipping
        if np.random.rand() < 0.25:
            image = cv2.flip(image, flipCode=1)
            label = cv2.flip(label, flipCode=1)
        # vertical flipping
        # if np.random.rand() < 0.25:
        #     image = cv2.flip(image, flipCode=0)
        #     label = cv2.flip(label, flipCode=0)
        # gaussian noise
        if np.random.rand() < 0.3:
            image = image + np.random.normal(0, np.random.uniform(0.05, 0.2), image.shape)

        label = np.round(label)
        # # # # intensity scale
        # if r[3] < 0.1:
        #     if len(np.where(label == 1)[0]) != 0 and len(np.where(label == 2)[0]) != 0:
        #         miu_csf = np.mean(image[label == 1])
        #         miu_tissue = np.mean(image[label == 2])
        #         # miu_csf_n1 = abs(
        #         #     np.random.uniform(-0.02, 0.02) + self.mean_tissue_csf[self.indice_patient[idx_new1]][1])
        #         # miu_tissue_n1 = abs(
        #         #     np.random.uniform(-0.02, 0.02) + self.mean_tissue_csf[self.indice_patient[idx_new1]][0])
        #         # miu_csf_n2 = abs(
        #         #     np.random.uniform(-0.02, 0.02) + self.mean_tissue_csf[self.indice_patient[idx_new2]][1])
        #         # miu_tissue_n2 = abs(
        #         #     np.random.uniform(-0.02, 0.02) + self.mean_tissue_csf[self.indice_patient[idx_new2]][0])
        #         # tmp = np.random.beta(0.5, 0.5)
        #         # miu_csf_n = tmp * miu_csf_n1 + (1 - tmp) * miu_csf_n2  # + np.random.uniform(-0.02, 0.02)
        #         # miu_tissue_n = tmp * miu_tissue_n1 + (1 - tmp) * miu_tissue_n2
        #         miu_csf_n = np.clip(miu_csf + np.random.normal(0, 0.12), miu_tissue+0.1, 0.9)
        #         #miu_csf_n = np.min([1, miu_csf_n])
        #         miu_tissue_n = np.clip(miu_tissue + np.random.normal(0, 0.12), 0.1, miu_csf_n)

        #         image[image < miu_tissue] = image[image < miu_tissue] * miu_tissue_n / (miu_tissue)
        #         image[(image < miu_csf) & (image > miu_tissue)] = miu_tissue_n + \
        #                                                           (image[(image < miu_csf) & (
        #                                                                       image > miu_tissue)] - miu_tissue) \
        #                                                           * (miu_csf_n - miu_tissue_n) / (miu_csf - miu_tissue)
        #         image[image > miu_csf] = miu_csf_n + (image[image > miu_csf] - miu_csf) * (1 - miu_csf_n) / (
        #                     1 - miu_csf)
        # synthesize air
        # if np.random.rand() < 0.5:
        #     r = list(self.air_temp.keys())[np.random.randint(2)]
        #     x_air, y_air = np.random.randint(r,512-r), np.random.randint(r,512-r)
        #     air_mask = self.air_temp[r] * label[x_air - r:x_air + r, y_air - r: y_air + r]  
        #     back_patch1 = np.concatenate([image[:r, :r], image[:r, 512 - r:512]], 1)
        #     back_patch2 = np.concatenate([image[512 - r:512, :r], image[512 - r:512, 512 - r:512]], 1)
        #     back_patch = np.concatenate([back_patch1, back_patch2], 0)
        #     image[x_air - r:x_air + r, y_air - r: y_air + r][air_mask != 0] = back_patch[air_mask != 0]
        #     label[x_air - r:x_air + r, y_air - r: y_air + r][air_mask != 0] = 3


        # deform
        # if r[4] < 0.5:
        #     points_of_interest = np.array([[128, 128],
        #                           [128, 384],
        #                           [384, 128],
        #                           [384, 384]])
        #     range_warp = 44
        #     projection = np.array([[128 + np.random.randint(-range_warp, range_warp), 128 + np.random.randint(-range_warp, range_warp)],
        #                  [128 + np.random.randint(-range_warp, range_warp), 384+ np.random.randint(-range_warp, range_warp)],
        #                  [384 + np.random.randint(-range_warp, range_warp), 128+ np.random.randint(-range_warp, range_warp)],
        #                  [384 + np.random.randint(-range_warp, range_warp), 384+ np.random.randint(-range_warp, range_warp)]])
        #     tform = transform.estimate_transform('projective', points_of_interest, projection)
        #     image = transform.warp(image, tform.inverse, mode='edge')
        #     label = transform.warp(label, tform.inverse, mode='edge')
        return image, label

    def augment2(self, r, image, dmap, label, opt):
        # label[label == 5] = 0
        # label[label == 4] = 3
        # rotation
        H, W = image.shape
        if r[0] < 0.4:
            # angle = np.random.randint(-30, 30)
            # M = cv2.getRotationMatrix2D((H//2, W//2), angle, 1.0)
            # image = cv2.warpAffine(image, M, (W, H), 1, 0)
            # label = cv2.warpAffine(label, M, (W, H), 0, 0)
            k = np.random.randint(0, 3)
            image = np.rot90(image, k)
            label = np.rot90(label, k)
        if r[1] < 0.25:
            # enlarge
            H, W = image.shape
            k1, k3 = r[7]/10, r[8]/10
            k2, k4 = 1 - r[5]/10, 1 - r[6]/10
            image = cv2.resize(image[int(k1 * H): int(k2 * H), int(k3 * W): int(k4 * W)], (W, H),
                               interpolation=cv2.INTER_CUBIC)
            dmap = cv2.resize(dmap[int(k1 * H): int(k2 * H), int(k3 * W): int(k4 * W)], (W, H),
                               interpolation=cv2.INTER_CUBIC)
            label = cv2.resize(label[int(k1 * H): int(k2 * H), int(k3 * W): int(k4 * W)], (W, H),
                               interpolation=cv2.INTER_NEAREST)
        elif r[1] > 0.75:
            # shrink
            tmp_img = np.zeros([H, W])
            tmp_dmap = np.ones([H, W]) * np.max(dmap) # notice this
            tmp_label = np.zeros([H, W])
            Hn, Wn = int(H * np.random.uniform(0.7, 1)), int(H * np.random.uniform(0.7, 1))
            x, y = np.random.randint(H - Hn), np.random.randint(W - Wn)
            tmp_img[x:x+Hn, y:y+Wn] = cv2.resize(image, (Wn, Hn), interpolation = cv2.INTER_NEAREST)
            tmp_dmap[x:x+Hn, y:y+Wn] = cv2.resize(dmap, (Wn, Hn), interpolation = cv2.INTER_NEAREST)
            tmp_label[x:x+Hn, y:y+Wn] = cv2.resize(label, (Wn, Hn), interpolation = cv2.INTER_NEAREST)
            image = tmp_img
            dmap = tmp_dmap
            label = tmp_label
        # horizonal flipping
        if np.random.rand() < 0.25:
            image = cv2.flip(image, flipCode=1)
            dmap = cv2.flip(dmap, flipCode=1)
            label = cv2.flip(label, flipCode=1)
        # vertical flipping
        if np.random.rand() < 0.2:
            image = cv2.flip(image, flipCode=0)
            label = cv2.flip(label, flipCode=0)
        # gaussian noise
        if np.random.rand() < 0.3:
            image = image + np.random.normal(0, np.random.uniform(0.05, 0.2), image.shape)
        if opt.syn_air == 1:
            if np.random.rand() < 0.1:
                r = list(self.air_temp.keys())[np.random.randint(2)]
                x_air, y_air = np.random.randint(r,512-r), np.random.randint(r,512-r)
                air_mask = self.air_temp[r] * label[x_air - r:x_air + r, y_air - r: y_air + r]  
                back_patch1 = np.concatenate([image[:r, :r], image[:r, 512 - r:512]], 1)
                back_patch2 = np.concatenate([image[512 - r:512, :r], image[512 - r:512, 512 - r:512]], 1)
                back_patch = np.concatenate([back_patch1, back_patch2], 0)
                image[x_air - r:x_air + r, y_air - r: y_air + r][air_mask != 0] = back_patch[air_mask != 0]
                label[x_air - r:x_air + r, y_air - r: y_air + r][air_mask != 0] = opt.num_classes -1
        label = np.round(label)
        # # # # intensity scale
        
        return image, dmap, label

    def __getitem__(self, item):
        path_image = self.df['image'][item]
        path_dmap = self.df['dmap'][item]
        path_label = self.df['label'][item]
        eaf = self.df['eaf'][item]
        
        img = np.load(path_image)
        dmap = np.load(path_dmap)
        label = np.load(path_label)
        
        if self.patch:
            cx, cy = np.clip(np.random.normal(0, 0.16, 2)*224 + 224, 128, 320)
            cx, cy = int(cx), int(cy)
            img = img[cx - 128:cx + 128, cy - 128: cy+ 128]
            label = label[cx - 128:cx + 128, cy - 128: cy+ 128]
        if self.aug:
            r = np.random.uniform(0, 1, 10)
            if self.args.distance_map == 1:
                img, dmap, label = self.augment2(r, img, dmap, label, self.args)
            else:
                img, label = self.augment(r, img, label)
        if self.args.distance_map == 1:
            img = np.concatenate([img[np.newaxis, :, :], img[np.newaxis, :, :], dmap[np.newaxis, :, :]], 0)
        else:
            img = np.concatenate([img[np.newaxis, :, :], img[np.newaxis, :, :], img[np.newaxis, :, :]], 0)
        label = np.round(label)
        # if self.args.reg_TV == 1:
        #     csf = np.zeros(label.shape)
        #     csf[label == 1] = 1
        #     csf = signal.convolve2d(csf, 1/64*np.ones([8,8]), mode='same')
        #     csf = signal.convolve2d(csf, 1/64*np.ones([8,8]), mode='same')
        #     csf[csf < 1] = 0
        # else:
        #     csf = np.zeros(label.shape)

        if self.normalize:
            mean = np.array([0.485, 0.456, 0.406]).reshape([3, 1, 1])
            std = np.array([0.229, 0.224, 0.225]).reshape([3, 1, 1])
            img = (img - mean)/std

        return torch.from_numpy(img.copy()).float(), \
               torch.from_numpy(label[None, :, :].copy()).float(), path_image.split('.')[-2], np.array([eaf])


    def __len__(self):
        return self.num_img




class Dataset_test(data.Dataset):
    def __init__(self, path_csv, opt, aug = True, normalize = True):
        super(Dataset_test, self).__init__()
        self.filename_list = pd.read_csv(path_csv)['0']
        self.filename_list = self.filename_list.tolist()
        self.filename_list.sort()
        self.num_img = len(self.filename_list)
        self.args = opt

        self.normalize = normalize
 
    def __getitem__(self, item):
        filename = self.filename_list[item]
        path_image = self.args.path_data + '/images/' + filename
        path_dmap = self.args.path_data + '/dmap/' + filename
        path_label = self.args.path_data + '/labels/' + filename[:-4] + '_label.png'
        
        img = cv2.imread(path_image)[:, :, 0]/255
        dmap = cv2.imread(path_dmap)[:, :, 0]/255
        
        label = cv2.imread(path_label)[:, :, 0]
        label[label == 255] = 0
        label = label/177
        label = label * 2
        
        if self.args.distance_map == 1:
            img = np.concatenate([img[np.newaxis, :, :], img[np.newaxis, :, :], dmap[np.newaxis, :, :]], 0)
        else:
            img = np.concatenate([img[np.newaxis, :, :], img[np.newaxis, :, :], img[np.newaxis, :, :]], 0)
        label = np.round(label)

        if self.normalize:
            mean = np.array([0.485, 0.456, 0.406]).reshape([3, 1, 1])
            std = np.array([0.229, 0.224, 0.225]).reshape([3, 1, 1])
            img = (img - mean)/std

        return torch.from_numpy(img.copy()).float(), \
               torch.from_numpy(label[None, :, :].copy()).float(), filename.split('.')[-2]


    def __len__(self):
        return self.num_img







class Dataset_Fill(data.Dataset):
    def __init__(self, path_h5, path_indice_table, path_tissue_csf_table, indice, opt, aug = True, normalize = True):
        super(Dataset_Fill, self).__init__()
        h5f = h5py.File(path_h5, 'r')
        indice_table = pd.read_csv(path_indice_table).values
        self.indice_patient = indice
        self.indice_image = []
        for idx in indice:
            indice_scan = [int(x) for x in indice_table[idx] if np.isnan(x) ==  False]
            self.indice_image += indice_scan

        #self.data = h5f['data']
        self.data = h5f['input']
        self.label = h5f['label']
        self.ID = h5f['ID']
        self.num_img = len(self.indice_image)
        self.aug = aug
        self.normalize = normalize
        self.adjacent = opt.adjacent

    def augment(self, r, image, label):
        # label[label == 5] = 0
        # label[label == 4] = 3
        # rotation
        H, W = image.shape
        if r[0] < 0.5:
            angle = np.random.randint(-180, 180)
            M = cv2.getRotationMatrix2D((H//2, W//2), angle, 1.0)
            image = cv2.warpAffine(image, M, (W, H), 1, 0)
            label = cv2.warpAffine(label, M, (W, H), 0, 0)
        if r[1] < 0.3:
            H, W = image.shape
            k1, k3 = r[7]/10, r[8]/10
            k2, k4 = 1 - r[5]/10, 1 - r[6]/10
            image = cv2.resize(image[int(k1 * H): int(k2 * H), int(k3 * W): int(k4 * W)], (512, 512),
                               interpolation=cv2.INTER_NEAREST)
            label = cv2.resize(label[int(k1 * H): int(k2 * H), int(k3 * W): int(k4 * W)], (512, 512),
                               interpolation=cv2.INTER_NEAREST)
        elif np.random.rand() < 0.3:
            img = np.zeros([H, W])
            label = np.zeros([H, W])
            Hn, Wn = int(H * np.random.uniform(0.7, 1)), int(H * np.random.uniform(0.7, 1))
            img_r = cv2.resize(img, (Wn, Hn), interpolation = cv2.INTER_NEAREST)
            label_r = cv2.resize(img, (Wn, Hn), interpolation = cv2.INTER_NEAREST)
            x, y = np.random.randint(H - Hn), np.random.randint(W - Wn)
            img[x:x+Hn, y:y+Wn] = img_r
            label[x:x+Hn, y:y+Wn] = label_r

        # horizonal flipping
        if r[2] < 0.3:
            image = cv2.flip(image, flipCode=1)
            label = cv2.flip(label, flipCode=1)
        if np.random.rand()<1:
            for i in range(3):
                h = np.random.randint(20, 40)
                w = np.random.randint(20, 40)
                x = np.random.randint(200, 312)
                y = np.random.randint(200, 312)
                image[x : x + h, y:y+ w] = 0.1
        # # # # intensity scale
        # if r[3] < 0.6 and r[4] < 0.6:
        #     miu_csf = np.mean(image[label == 1])
        #     miu_tissue = np.mean(image[label == 2])
        #     if not np.isnan(miu_tissue) and not np.isnan(miu_csf):
        #         # miu_csf_n1 = abs(
        #         #     np.random.uniform(-0.02, 0.02) + self.mean_tissue_csf[self.indice_patient[idx_new1]][1])
        #         # miu_tissue_n1 = abs(
        #         #     np.random.uniform(-0.02, 0.02) + self.mean_tissue_csf[self.indice_patient[idx_new1]][0])
        #         # miu_csf_n2 = abs(
        #         #     np.random.uniform(-0.02, 0.02) + self.mean_tissue_csf[self.indice_patient[idx_new2]][1])
        #         # miu_tissue_n2 = abs(
        #         #     np.random.uniform(-0.02, 0.02) + self.mean_tissue_csf[self.indice_patient[idx_new2]][0])
        #         # tmp = np.random.beta(0.5, 0.5)
        #         # miu_csf_n = tmp * miu_csf_n1 + (1 - tmp) * miu_csf_n2  # + np.random.uniform(-0.02, 0.02)
        #         # miu_tissue_n = tmp * miu_tissue_n1 + (1 - tmp) * miu_tissue_n2
        #         miu_csf_n = miu_csf + np.random.uniform(-0.2,0.2) 
        #         miu_csf_n = np.min([1, miu_csf_n])
        #         miu_tissue_n = miu_tissue + np.random.uniform(-0.2,0.2) 

        #         image[image < miu_tissue] = image[image < miu_tissue] * miu_tissue_n / (miu_tissue)
        #         image[(image < miu_csf) & (image > miu_tissue)] = miu_tissue_n + \
        #                                                           (image[(image < miu_csf) & (
        #                                                                       image > miu_tissue)] - miu_tissue) \
        #                                                           * (miu_csf_n - miu_tissue_n) / (miu_csf - miu_tissue)
        #         image[image > miu_csf] = miu_csf_n + (image[image > miu_csf] - miu_csf) * (1 - miu_csf_n) / (
        #                     1 - miu_csf)

        # deform
        # if r[4] < 0.5:
        #     points_of_interest = np.array([[128, 128],
        #                           [128, 384],
        #                           [384, 128],
        #                           [384, 384]])
        #     range_warp = 44
        #     projection = np.array([[128 + np.random.randint(-range_warp, range_warp), 128 + np.random.randint(-range_warp, range_warp)],
        #                  [128 + np.random.randint(-range_warp, range_warp), 384+ np.random.randint(-range_warp, range_warp)],
        #                  [384 + np.random.randint(-range_warp, range_warp), 128+ np.random.randint(-range_warp, range_warp)],
        #                  [384 + np.random.randint(-range_warp, range_warp), 384+ np.random.randint(-range_warp, range_warp)]])
        #     tform = transform.estimate_transform('projective', points_of_interest, projection)
        #     image = transform.warp(image, tform.inverse, mode='edge')
        #     label = transform.warp(label, tform.inverse, mode='edge')
        return image, label

    def __getitem__(self, item):
    
        idx = self.indice_image[item]
        img = self.data[idx, 0, :, :]
        label = self.label[idx, 0, :, :]
        ID = str(self.ID[idx])[2:-1]
        label[label == 5] = 0
        label[label == 4] = 3

        if self.aug:
            r = np.random.uniform(0, 1, 10)
            img, label = self.augment(r, img, label)
        img = np.tile(img[np.newaxis, :, :], (3, 1,1))
        label = label[np.newaxis, :, :]
        label = np.round(label)

        if self.normalize:
            mean = np.array([0.485, 0.456, 0.406]).reshape([3, 1, 1])
            std = np.array([0.229, 0.224, 0.225]).reshape([3, 1, 1])
            img = (img - mean)/std

        return torch.from_numpy(img.copy()).float(), \
               torch.from_numpy(label.copy()).float(), ID


    def __len__(self):
        return self.num_img




class Dataset_SR(data.Dataset):
    def __init__(self, path_h5, opt, aug = True, normalize = True):
        super(Dataset_SR, self).__init__()
        h5f = h5py.File(path_h5, 'r')

        self.data = h5f['data']

        self.num_img = self.data.shape[0]
        self.aug = aug
        self.normalize = normalize

    def augment(self, r, image):
        # rotation
        H, W = image.shape
        if r[0] < 1:
            angle = (r[0] - 0.2) * 10
            M = cv2.getRotationMatrix2D((H//2, W//2), angle, 1.0)
            image = cv2.warpAffine(image, M, (W, H), 1, 0)
            
        # if r[1] < 0.3:
        #     H, W = image.shape
        #     k1, k3 = r[7]/10, r[8]/10
        #     k2, k4 = 1 - r[5]/10, 1 - r[6]/10
        #     image = cv2.resize(image[int(k1 * H): int(k2 * H), int(k3 * W): int(k4 * W)], (W, H),
        #                        interpolation=cv2.INTER_NEAREST)
            
        # horizonal flipping
        if r[2] < 0.3:
            image = cv2.flip(image, flipCode=np.random.randint(2))
        
        # deform
        # if r[4] < 0.5:
        #     points_of_interest = np.array([[128, 128],
        #                           [128, 384],
        #                           [384, 128],
        #                           [384, 384]])
        #     range_warp = 44
        #     projection = np.array([[128 + np.random.randint(-range_warp, range_warp), 128 + np.random.randint(-range_warp, range_warp)],
        #                  [128 + np.random.randint(-range_warp, range_warp), 384+ np.random.randint(-range_warp, range_warp)],
        #                  [384 + np.random.randint(-range_warp, range_warp), 128+ np.random.randint(-range_warp, range_warp)],
        #                  [384 + np.random.randint(-range_warp, range_warp), 384+ np.random.randint(-range_warp, range_warp)]])
        #     tform = transform.estimate_transform('projective', points_of_interest, projection)
        #     image = transform.warp(image, tform.inverse, mode='edge')
        #     label = transform.warp(label, tform.inverse, mode='edge')
        return image

    def __getitem__(self, item):
        cx, cy = np.random.randint(256 - 64, 256 + 64, 2)
        img = self.data[item, 0, cx -128-64:cx + 128+64, cy-128-64:cy+128+64]
        if self.aug:
            r = np.random.uniform(0, 1, 10)
            img = self.augment(r, img)
        GT = img[None, :, :]
        input = img[::4, ::4]
        input = cv2.resize(input, (256+128, 256+128), interpolation=cv2.INTER_CUBIC)
        input = np.clip(input, 0, 1)
        input = np.tile(input, (3, 1, 1))
        if self.normalize:
            mean = np.array([0.485, 0.456, 0.406]).reshape([3, 1, 1])
            std = np.array([0.229, 0.224, 0.225]).reshape([3, 1, 1])
            input = (input - mean)/std

        return torch.from_numpy(input.copy()).float(), torch.from_numpy(GT.copy()).float()


    def __len__(self):
        return self.num_img


class Dataset_adja(data.Dataset):
    def __init__(self, path_h5, path_indice_table, path_tissue_csf_table, indice, opt, aug=True, normalize=True):
        super(Dataset_adja, self).__init__()
        h5f = h5py.File(path_h5, 'r')
        indice_table = pd.read_csv(path_indice_table).values
        self.indice_patient = indice
        self.indice_image = []
        for idx in indice:
            indice_scan = [int(x) for x in indice_table[idx] if np.isnan(x) == False]
            self.indice_image += indice_scan

        # self.data = h5f['data']
        self.data = h5f['input']
        self.label = h5f['label']
        self.depth = h5f['depth']
        self.mean_tissue_csf = pd.read_csv(path_tissue_csf_table).values[:, 1:]
        self.num_img = len(self.indice_image)
        self.aug = aug
        self.normalize = normalize
        self.adjacent = opt.adjacent

    def augment(self, r, image, label):
        # label[label == 5] = 0
        # label[label == 4] = 3
        # rotation
        if r[0] < 0.4:
            angle = (r[0] - 0.2) * 10
            image = rotate(image, angle=angle, mode='wrap')
            label = rotate(label, angle=angle, mode='wrap')
        if r[1] < 0.3:
            H, W = image.shape
            k1, k3 = r[7]/10, r[8]/10
            k2, k4 = 1 - r[5]/10, 1 - r[6]/10
            image = cv2.resize(image[int(k1 * H): int(k2 * H), int(k3 * W): int(k4 * W)], (512, 512),
                               interpolation=cv2.INTER_NEAREST)
            label = cv2.resize(label[int(k1 * H): int(k2 * H), int(k3 * W): int(k4 * W)], (512, 512),
                               interpolation=cv2.INTER_NEAREST)
        # horizonal flipping
        if r[2] < 0.3:
            image = cv2.flip(image, flipCode=1)
            label = cv2.flip(label, flipCode=1)
        # # # # intensity scale
        if r[3] < 0.6 and r[4] < 0.6:
            idx_new1 = int(r[3] // 0.03)
            idx_new2 = int(r[4] // 0.03)
            miu_csf = np.mean(image[label == 1])
            miu_tissue = np.mean(image[label == 2])
            if not np.isnan(miu_tissue) and not np.isnan(miu_csf):
                # miu_csf_n1 = abs(
                #     np.random.uniform(-0.02, 0.02) + self.mean_tissue_csf[self.indice_patient[idx_new1]][1])
                # miu_tissue_n1 = abs(
                #     np.random.uniform(-0.02, 0.02) + self.mean_tissue_csf[self.indice_patient[idx_new1]][0])
                # miu_csf_n2 = abs(
                #     np.random.uniform(-0.02, 0.02) + self.mean_tissue_csf[self.indice_patient[idx_new2]][1])
                # miu_tissue_n2 = abs(
                #     np.random.uniform(-0.02, 0.02) + self.mean_tissue_csf[self.indice_patient[idx_new2]][0])
                # tmp = np.random.beta(0.5, 0.5)
                # miu_csf_n = tmp * miu_csf_n1 + (1 - tmp) * miu_csf_n2  # + np.random.uniform(-0.02, 0.02)
                # miu_tissue_n = tmp * miu_tissue_n1 + (1 - tmp) * miu_tissue_n2
                miu_csf_n = miu_csf + (r[5]-0.66)/3
                miu_csf_n = np.min([1, miu_csf_n])
                miu_tissue_n = miu_tissue + (r[6]-0.34)/3

                image[image < miu_tissue] = image[image < miu_tissue] * miu_tissue_n / (miu_tissue)
                image[(image < miu_csf) & (image > miu_tissue)] = miu_tissue_n + \
                                                                  (image[(image < miu_csf) & (
                                                                              image > miu_tissue)] - miu_tissue) \
                                                                  * (miu_csf_n - miu_tissue_n) / (miu_csf - miu_tissue)
                image[image > miu_csf] = miu_csf_n + (image[image > miu_csf] - miu_csf) * (1 - miu_csf_n) / (
                            1 - miu_csf)

        # deform
        # if r[4] < 0.5:
        #     points_of_interest = np.array([[128, 128],
        #                           [128, 384],
        #                           [384, 128],
        #                           [384, 384]])
        #     range_warp = 44
        #     projection = np.array([[128 + np.random.randint(-range_warp, range_warp), 128 + np.random.randint(-range_warp, range_warp)],
        #                  [128 + np.random.randint(-range_warp, range_warp), 384+ np.random.randint(-range_warp, range_warp)],
        #                  [384 + np.random.randint(-range_warp, range_warp), 128+ np.random.randint(-range_warp, range_warp)],
        #                  [384 + np.random.randint(-range_warp, range_warp), 384+ np.random.randint(-range_warp, range_warp)]])
        #     tform = transform.estimate_transform('projective', points_of_interest, projection)
        #     image = transform.warp(image, tform.inverse, mode='edge')
        #     label = transform.warp(label, tform.inverse, mode='edge')
        return image, label

    def __getitem__(self, item):
        item = self.indice_image[item]
        count = 0
        item_start = item
        item_end = item
        can_go_right, can_go_left = True, True
        count_num = 5
        while count < count_num:
            r = np.random.randint(2)
            if can_go_left:
                if r == 1:
                    if self.depth[item_start - 1] < self.depth[item_start]:
                        item_start -= 1
                        count += 1
                    else:
                        can_go_left = False
            elif can_go_right:
                if r == 0:
                    if self.depth[item_end] < self.depth[item_end + 1]:
                        item_end += 1
                        count += 1
                    else:
                        can_go_right = False
            else:
                raise ValueError('Something wrong in item assignment')
        items = np.sort(np.random.permutation(count_num)[:5]) + item_start
        img1 = self.data[items[0]][0]
        label1 = self.label[items[0]][0]
        img2 = self.data[items[1]][0]
        label2 = self.label[items[1]][0]
        img3 = self.data[items[2]][0]
        label3 = self.label[items[2]][0]
        img4 = self.data[items[3]][0]
        label4 = self.label[items[3]][0]
        img5 = self.data[items[4]][0]
        label5 = self.label[items[4]][0]

        if self.aug:
            r = np.random.uniform(0, 1, 10)
            img1, label1 = self.augment(r, img1, label1)
            img2, label2 = self.augment(r, img2, label2)
            img3, label3 = self.augment(r, img3, label3)
            img4, label4 = self.augment(r, img4, label4)
            img5, label5 = self.augment(r, img5, label5)
        img1 = np.tile(img1, (3, 1, 1))
        img2 = np.tile(img2, (3, 1, 1))
        img3 = np.tile(img3, (3, 1, 1))
        img4 = np.tile(img4, (3, 1, 1))
        img5 = np.tile(img5, (3, 1, 1))

        if self.normalize:
            mean = np.array([0.485, 0.456, 0.406]).reshape([3, 1, 1])
            std = np.array([0.229, 0.224, 0.225]).reshape([3, 1, 1])
            img1 = (img1 - mean) / std
            img2 = (img2 - mean) / std
            img3 = (img3 - mean) / std
            img4 = (img4 - mean) / std
            img5 = (img5 - mean) / std
        img_all = np.concatenate([img1[None, :, :, :], img2[None, :, :, :], img3[None, :, :, :], img4[None, :, :, :], img5[None, :, :, :]], axis=0)
        label_all = np.concatenate([label1[None, :, :], label2[None, :, :], label3[None, :, :], label4[None, :, :], label5[None, :, :]], 0)
        return torch.from_numpy(img_all).float(), \
               torch.from_numpy(label_all).float()

    def __len__(self):
        return self.num_img


def get_adjacent(self, item, dataset=1):
    if dataset == 1:
        if self.depth1[item - 1] > self.depth1[item]:
            item += 1
        elif self.depth1[item] > self.depth1[(item + 1)%self.num_img1]:
            item -= 1
        return (self.data1[item - 1], self.data1[item], self.data1[item + 1]),\
               (self.label1[item - 1], self.label1[item], self.label1[item+1]), np.array(self.depth1[item])
    # else:
    #     if self.depth2[item - 1] > self.depth2[item] :
    #         item += 1
    #     elif self.depth2[item] > self.depth2[(item + 1)%self.num_img1]:
    #         item -= 1
    #     return (self.data2[item - 1], self.data2[item], self.data2[item + 1]), \
    #            (self.label2[item - 1], self.label2[item], self.label2[item + 1]), np.array(self.depth2[item])

def augment(image, label):
    #label[label == 5] = 0
    #label[label == 4] = 3
    # rotation
    H, W = image.shape
    if np.random.rand() < 0.4:
        angle = np.random.randint(-15, 15)
        image = rotate(image, angle=angle, mode='wrap')
        label = rotate(label, angle=angle, mode='wrap')
    if np.random.rand() < 0.2:
        H, W = image.shape
        [k1, k3] = np.random.uniform(0, 0.1, 2)
        [k2, k4] = np.random.uniform(0.9, 1, 2)
        image = cv2.resize(image[int(k1 * H): int(k2 * H), int(k3 * W): int(k4 * W)], (H, W),
                             interpolation=cv2.INTER_CUBIC)
        label = cv2.resize(label[int(k1 * H): int(k2 * H), int(k3 * W): int(k4 * W)], (H, W),
                               interpolation=cv2.INTER_NEAREST)
    # horizonal flipping
    # if r[2] < 0.3:
    #     image = np.flip(image, axis=1)
    #     label = np.flip(label, axis=1)
    # intensity scale
    # if r[3] < 0.4:
    #     miu2 = np.mean(image[label == 1])
    #     miu1 = np.mean(image[label == 2])
    #     if not np.isnan(miu1) and not np.isnan(miu2):
    #         miu1_n = abs(np.random.uniform(-0.1, 0.2) + miu1)
    #         miu2_n = abs(np.random.uniform(-0.2, 0.1) + miu2)
    #         for i in range(512):
    #             for j in range(512):
    #                 val = image[i][j]
    #                 if val < miu1:
    #                     image[i][j] = val * miu1_n / (miu1 + 0.0001)
    #                 elif val < miu2:
    #                     image[i][j] = miu1_n + (miu2_n - miu1_n) / (abs(miu2 - miu1) + 0.0001) * (val - miu1)
    #                 else:
    #                     image[i][j] = miu2_n + (1 - miu2_n) / (abs(1 - miu2) + 0.0001) * (val - miu2)
    # deform
    # if r[4] < 0.5:
    #     points_of_interest = np.array([[128, 128],
    #                           [128, 384],
    #                           [384, 128],
    #                           [384, 384]])
    #     range_warp = 44
    #     projection = np.array([[128 + np.random.randint(-range_warp, range_warp), 128 + np.random.randint(-range_warp, range_warp)],
    #                  [128 + np.random.randint(-range_warp, range_warp), 384+ np.random.randint(-range_warp, range_warp)],
    #                  [384 + np.random.randint(-range_warp, range_warp), 128+ np.random.randint(-range_warp, range_warp)],
    #                  [384 + np.random.randint(-range_warp, range_warp), 384+ np.random.randint(-range_warp, range_warp)]])
    #     tform = transform.estimate_transform('projective', points_of_interest, projection)
    #     image = transform.warp(image, tform.inverse, mode='edge')
    #     label = transform.warp(label, tform.inverse, mode='edge')
    return image, np.round(label)


class Dataset_joint(data.Dataset):
    def __init__(self, path1, path2, opt, aug = True):
        super(Dataset_joint, self).__init__()
        h5f1 = h5py.File(path1, 'r')
        h5f2 = h5py.File(path2, 'r')
        #self.data = h5f['data']
        self.data1 = h5f1['input']
        self.label1 = h5f1['label']
        #self.depth1 = h5f1['depth']

        self.data2 = h5f2['data']
        self.label2 = h5f2['label']
        #self.depth2 = h5f2['depth']

        self.num_img1 = self.data1.shape[0]
        self.num_img2 = self.data2.shape[0]
        self.aug = aug
        self.adjacent = opt.adjacent

    def __getitem__(self, item):
        r = np.random.randint(3)
        if self.adjacent == True:
            if r >0:
                item = item % self.num_img1
                images, labels, depth = get_adjacent(self, item, 1)
            else:
                images, labels, depth = get_adjacent(self, item, 2)
            r = np.random.uniform(0, 1,4)

            image1, _ = augment(r, images[0][0], labels[0][0])
            image2, label = augment(r, images[1][0], labels[1][0])
            image3, _ = augment(r, images[2][0], labels[2][0])
            images = np.concatenate([image1[np.newaxis, :, :], image2[np.newaxis, :, :], image3[np.newaxis, :, :]], 0)
        else:
            if r > 0:
                dataset = np.array(0)
                item = item % self.num_img1
                image, label = self.data1[item][0], self.label1[item][0]
            else:
                dataset = np.array(1)
                item = item % self.num_img2
                image, label = self.data2[item][0], self.label2[item][0]
            label[label == 5] = 0
            label[label == 4] = 3
            r = np.random.uniform(0, 1, 4)
            image, label = augment(r, image, label)
            images = np.tile(image, (3, 1, 1))

        mean = np.array([0.485, 0.456, 0.406]).reshape([3, 1, 1])
        std = np.array([0.229, 0.224, 0.225]).reshape([3, 1, 1])
        images = (images - mean)/std

        label = label[np.newaxis, :, :]

        return torch.from_numpy(images.copy()).float().cuda(), \
               torch.from_numpy(label.copy()).float().cuda(),  torch.from_numpy(dataset).float().cuda()


    def __len__(self):
        return max(self.num_img1, self.num_img2)





class DatasetH5(data.Dataset):
    def __init__(self, path, aug = True):
        super(DatasetH5, self).__init__()
        h5f1 = h5py.File(path, 'r')
        self.data = h5f1['data']
        self.label = h5f1['label']
        self.num_img = self.data.shape[0]
        self.aug = aug

    def __getitem__(self, item):
        image = self.data[item]
        label = self.label[item]
        image, label = augment(image, label)
        images = np.tile(image, (3, 1, 1))

        mean = np.array([0.485, 0.456, 0.406]).reshape([3, 1, 1])
        std = np.array([0.229, 0.224, 0.225]).reshape([3, 1, 1])
        images = (images - mean)/std

        label = label[np.newaxis, :, :]
        return torch.from_numpy(images.copy()).float(), torch.from_numpy(label.copy()).float()


    def __len__(self):
        return self.num_img





class Dataset_UDA(data.Dataset):
    def __init__(self, opt):
        super(Dataset_UDA, self).__init__()
        self.isTrain = opt.isTrain
        path_data_B = opt.path_B
        if opt.isTrain:
            self.aug = True
            path_data_A = opt.path_train_A
        else:
            self.aug = False
            path_data_A = opt.path_test_A
        h5_A = h5py.File(path_data_A, 'r')
        h5_B = h5py.File(path_data_B, 'r')
        self.real_A_dataset = h5_A['input']
        #self.real_A_labelset = h5_A['label']
        self.real_B_dataset = h5_B['input']
        self.num_A = self.real_A_dataset.shape[0]
        self.num_B = self.real_B_dataset.shape[0]
        self.num_imgs = max(self.num_A, self.num_B)
        if not opt.isTrain:
            if opt.domain == 'B':
                self.num_imgs = self.num_B

    def __getitem__(self, index):
        real_A = self.real_A_dataset[index % self.num_A, 0, 16:-16, 16:-16]
        #label_A = self.real_A_labelset[index % self.num_A, 0, M:512-M, M:512-M]
        real_B = self.real_B_dataset[index % self.num_B, 0, 16:-16, 16:-16]
        #label_A[label_A == 5] = 0
        #label_A[label_A == 4] = 3
        if self.aug:
            r = np.random.randint(2)
            if r == 1:
                r1 = np.random.randint(-15, 15)
                real_A = rotate(real_A, angle=2 * r1, mode='wrap')
                #label_A = rotate(label_A, angle=2 * r1, mode='wrap')
                real_B = rotate(real_B, angle=2 * r1, mode='wrap')

            # cropping and resizing
            r = np.random.randint(6)
            H, W = real_A.shape
            Hn, Wn = 448, 448
            if r == 1:
                real_A = cv2.resize(real_A[int(0.1 * H): int(0.9 * H), int(0.1 * W): int(0.9 * W)], (Hn, Wn))
                #label_A = cv2.resize(label_A[int(0.1 * H): int(0.9 * H), int(0.1 * W): int(0.9 * W)], (H, W))
                real_B = cv2.resize(real_B[int(0.1 * H): int(0.9 * H), int(0.1 * W): int(0.9 * W)], (Hn, Wn))
            elif r == 2:
                real_A = cv2.resize(real_A[int(0.15 * H): int(0.85 * H), int(0.15 * W): int(0.85 * W)], (Hn, Wn))
                #label_A = cv2.resize(label_A[int(0.15 * H): int(0.85 * H), int(0.15 * W): int(0.85 * W)], (H, W))
                real_B = cv2.resize(real_B[int(0.15 * H): int(0.85 * H), int(0.15 * W): int(0.85 * W)], (Hn, Wn))
            else:
                real_A = cv2.resize(real_A, (Hn, Wn))
                real_B = cv2.resize(real_B, (Hn, Wn))
            # flipping
            r = np.random.randint(4)
            if r == 0:
                real_A = np.flip(real_A, axis=1)
                #label_A = np.flip(label_A, axis=1)
                real_B = np.flip(real_B, axis=1)

            # noise
            # r = np.random.randint(6)
            # if r == 1:
            #     real_A = np.random.normal(0, 0.015, [H, W]) + real_A
            #     real_B = np.random.normal(0, 0.015, [H, W]) + real_B
            # elif r == 2:
            #     real_A = np.random.normal(0, 0.015, [H, W]) + real_A
            #     real_B = np.random.normal(0, 0.015, [H, W]) + real_B
        real_A = real_A[np.newaxis, :, :]
        real_B = real_B[np.newaxis, :, :]
        return torch.from_numpy(real_A.copy()).float(), \
               torch.from_numpy(real_B.copy()).float()

    def __len__(self):
        return self.num_imgs