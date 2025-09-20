import os
import numpy as np
import SimpleITK as sitk
import random
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
data_path = ''
data_path_valid = ''
data_path_test = ''


class Dataset_lungmr(Dataset):
    def __init__(self, args, mode='train'):
        self.mode = mode
        self.root = args['data']['image']
        self.nomiss = args['data']['nomiss']

        self.file_X = []
        self.indicator = []

        with open(args['data'][mode], 'r') as f:
            strs = f.readlines()
            for line in strs:
                name = line.split('\n')[0]
                self.file_X.append(name)
                if self.nomiss and self.mode=='train' or self.mode=='valid':
                    self.indicator.append('1111')
                else:
                    self.indicator.append('1111')
    def __len__(self):
        return len(self.file_X)
    
    def norm(self, arr):
        """ norm (0, 99%) to (0, 1)
        arr: [s,d,w,h]
        """
        amax = np.percentile(arr, 99) * 2
        arr = np.clip(arr, 0, amax) / amax * 2 - 1
        return arr
    def norm_vibe(self, arr):
        """ norm (0, 200) to (0, 1)
        arr: [s,d,w,h]
        """
        amax = 200
        arr = np.clip(arr, 0, amax) / amax * 2 - 1
        return arr
    def preprocess(self, x, k=[0,0,0], axis=[0,1,2]):
        nd = 60
        d, w, h = x.shape
        rd = (d-nd)//2 if d>nd else 0
        x = x[rd:rd+nd]

        if self.mode=='train':
            x = np.transpose(x, axes=axis)
            if k[2]==1:
                x = x[:, :, ::-1]
            if k[1]==1:
                x = x[:, ::-1, :]
            if k[0]==1:
                x = x[::-1, :, :]
            x = x.copy()

        x = np.expand_dims(x, axis=0)
        x = np.expand_dims(x, axis=0) # [1,1,d,w,h]
        return x
    
    def __getitem__(self, index):
        x_idx = self.file_X[index]
        i_idx = self.indicator[index]


        if self.mode=='train':
            #import image as nifti format

            img_vibe = self.norm(sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(x_idx, '{}_VIBE.nii.gz'.format(os.path.basename(x_idx))))))
            img_pbf = self.norm(sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(x_idx, '{}_PBF.nii.gz'.format(os.path.basename(x_idx))))))
            img_postvibe = self.norm(sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(x_idx, '{}_postVIBE.nii.gz'.format(os.path.basename(x_idx))))))
            img_rmax = self.norm(sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(x_idx, '{}IRFmax.nii.gz'.format(os.path.basename(x_idx))))))
            img_prm = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(x_idx, '{}_PRM.nii.gz'.format(os.path.basename(x_idx)))))
            img_prm = (img_prm>0) * (3-img_prm)   #here 2=Empy 1=fsad 0=normal

        elif self.mode=='valid':

            img_vibe = self.norm(sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(x_idx, '{}_VIBE.nii.gz'.format(os.path.basename(x_idx))))))
            img_pbf = self.norm(sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(x_idx, '{}_PBF.nii.gz'.format(os.path.basename(x_idx))))))
            img_postvibe = self.norm(sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(x_idx, '{}_postVIBE.nii.gz'.format(os.path.basename(x_idx))))))
            img_rmax = self.norm(sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(x_idx, '{}IRFmax.nii.gz'.format(os.path.basename(x_idx))))))
            img_prm = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(x_idx, '{}_PRM.nii.gz'.format(os.path.basename(x_idx)))))
        elif self.mode=='test':
            img_vibe = self.norm(sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(x_idx, '{}_VIBE.nii.gz'.format(os.path.basename(x_idx))))))
            img_pbf = self.norm(sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(x_idx, '{}_PBF.nii.gz'.format(os.path.basename(x_idx))))))
            img_postvibe = self.norm(sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(x_idx, '{}_postVIBE.nii.gz'.format(os.path.basename(x_idx))))))
            img_rmax = self.norm(sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(x_idx, '{}IRFmax.nii.gz'.format(os.path.basename(x_idx))))))
            img_prm = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(x_idx, '{}_PRM.nii.gz'.format(os.path.basename(x_idx)))))

        k = [0, 0, random.randint(0,1)]
        axis = [0,1,2]

        img_vibe = self.preprocess(img_vibe, k, axis)
        img_rmax = self.preprocess(img_rmax, k, axis)
        img_pbf = self.preprocess(img_pbf, k, axis)
        img_postvibe = self.preprocess(img_postvibe, k, axis)
        img_prm = self.preprocess(img_prm, k, axis)


        ind = []
        for i in range(4):
            if i_idx[i] == '1':
                ind.append(i)
        return {
            'vibe': torch.from_numpy(img_vibe),
            'rmax': torch.from_numpy(img_rmax),
            'pbf': torch.from_numpy(img_pbf),
            'postvibe': torch.from_numpy(img_postvibe),
            'prm':torch.from_numpy(img_prm),
            'flag': ind,
            'path': [x_idx],
        }