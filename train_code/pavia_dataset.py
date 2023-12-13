from torch.utils.data import Dataset
import numpy as np
import random
import cv2
import h5py

class TrainDataset(Dataset):
    def __init__(self, data_root, crop_size, arg=True, stride=8):
        self.crop_size = crop_size
        self.hypers = []
        self.bgrs = []
        self.dy = []
        self.arg = arg
        h, w = 610, 340  # img shape
        self.stride = stride
        self.patch_per_line = (w-crop_size)//stride+1
        self.patch_per_colum = (h-crop_size)//stride+1
        self.patch_per_img = self.patch_per_line*self.patch_per_colum
        self.val = self.patch_per_img//10 + 1

        tag = 1
        for i in range(self.patch_per_img):
            if tag % 10 == 0:
                tag = tag + 1
            self.dy.append(tag)
            tag = tag + 1

        self.hyper_data_path = f'{data_root}/Train_Spec/'
        self.bgr_data_path = f'{data_root}/Train_RGB/'

        with open(f'{data_root}/split_txt/train_list.txt', 'r') as fin:
            self.hyper_list = [line.replace('\n', '.mat') for line in fin]
            self.bgr_list = [line for line in self.hyper_list]
        self.hyper_list.sort()
        self.bgr_list.sort()
        print(f'len(hyper) of pavia dataset:{len(self.hyper_list)}')
        print(f'len(bgr) of pavia dataset:{len(self.bgr_list)}')
        hyper_path = self.hyper_data_path + self.hyper_list[0]
        with h5py.File(hyper_path, 'r') as mat1:
            hyper =np.float32(np.array(mat1['pavia']))
        hyper = (hyper-hyper.min())/(hyper.max()-hyper.min())
        self.hyper = np.transpose(hyper, [0, 2, 1])
        bgr_path = self.bgr_data_path + self.bgr_list[0]
        with h5py.File(bgr_path, 'r') as mat2:
            bgr = np.float32(np.array(mat2['pavia']))
        bgr = (bgr-bgr.min())/(bgr.max()-bgr.min())
        self.bgr = np.transpose(bgr, [0, 2, 1])  # [3,482,512]
        mat1.close()
        mat2.close()

        self.img_num = len(self.hyper_list)
        self.length = self.patch_per_img * self.img_num - self.val

    def arguement(self, img, rotTimes, vFlip, hFlip):
        # Random rotation
        for j in range(rotTimes):
            img = np.rot90(img.copy(), axes=(1, 2))
        # Random vertical Flip
        for j in range(vFlip):
            img = img[:, :, ::-1].copy()
        # Random horizontal Flip
        for j in range(hFlip):
            img = img[:, ::-1, :].copy()
        return img

    def __getitem__(self, idx):

        stride = self.stride
        crop_size = self.crop_size
        img_idx, patch_idx = idx//self.patch_per_img, idx%self.patch_per_img
        h_idx, w_idx = patch_idx//self.patch_per_line, patch_idx%self.patch_per_line
        bgr = self.bgr[:, h_idx * stride:h_idx * stride + crop_size, w_idx * stride:w_idx * stride + crop_size]
        hyper = self.hyper[:, h_idx * stride:h_idx * stride + crop_size, w_idx * stride:w_idx * stride + crop_size]
        rotTimes = random.randint(0, 3)
        vFlip = random.randint(0, 1)
        hFlip = random.randint(0, 1)
        if self.arg:
            bgr = self.arguement(bgr, rotTimes, vFlip, hFlip)
            hyper = self.arguement(hyper, rotTimes, vFlip, hFlip)
        return np.ascontiguousarray(bgr), np.ascontiguousarray(hyper)

    def __len__(self):
        return self.patch_per_img - self.val

class ValidDataset(Dataset):
    def __init__(self, data_root, crop_size, arg=True, stride=8):
        self.crop_size = crop_size
        self.hypers = []
        self.bgrs = []
        self.dy = []
        self.arg = arg
        h, w = 610, 340  # img shape
        self.stride = stride
        self.patch_per_line = (w-crop_size)//stride+1
        self.patch_per_colum = (h-crop_size)//stride+1
        self.patch_per_img = self.patch_per_line*self.patch_per_colum
        self.val = self.patch_per_img//10 + 1
        self.front = self.patch_per_img - self.val

        tag = 1
        for i in range(self.patch_per_img):
            if tag % 10 == 0:
                tag = tag + 1
            self.dy.append(tag)
            tag = tag + 1

        self.hyper_data_path = f'{data_root}/Train_Spec/'
        self.bgr_data_path = f'{data_root}/Train_RGB/'

        with open(f'{data_root}/split_txt/train_list.txt', 'r') as fin:
            self.hyper_list = [line.replace('\n', '.mat') for line in fin]
            self.bgr_list = [line for line in self.hyper_list]
        self.hyper_list.sort()
        self.bgr_list.sort()
        print(f'len(hyper) of pavia dataset:{len(self.hyper_list)}')
        print(f'len(bgr) of pavia dataset:{len(self.bgr_list)}')

        i = 0
        hyper_path = self.hyper_data_path + self.hyper_list[i]
        with h5py.File(hyper_path, 'r') as mat1:
            hyper = np.float32(np.array(mat1['pavia']))
        hyper = (hyper - hyper.min()) / (hyper.max() - hyper.min())
        self.hyper = np.transpose(hyper, [0, 2, 1])
        bgr_path = self.bgr_data_path + self.bgr_list[i]
        with h5py.File(bgr_path, 'r') as mat2:
            bgr = np.float32(np.array(mat2['pavia']))
        bgr = (bgr - bgr.min()) / (bgr.max() - bgr.min())
        self.bgr = np.transpose(bgr, [0, 2, 1])  # [3,482,512]
        mat1.close()
        mat2.close()

        self.img_num = len(self.hyper_list)
        self.length = self.val

    def __getitem__(self, idx):

        idx = idx + self.front

        stride = self.stride
        crop_size = self.crop_size
        img_idx, patch_idx = idx//self.patch_per_img, idx%self.patch_per_img
        h_idx, w_idx = patch_idx//self.patch_per_line, patch_idx%self.patch_per_line


        # print(f'Pavia scene {i} is loaded.')

        bgr = self.bgr
        hyper = self.hyper
        return np.ascontiguousarray(bgr), np.ascontiguousarray(hyper)

    def __len__(self):
        return 1