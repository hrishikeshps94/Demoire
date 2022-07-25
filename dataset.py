import random
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
import PIL.Image as Image
import numpy as np
import os


class Custom_Dataset(Dataset):
    def __init__(self,root_dir,im_shape=None, is_train=False):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.is_train = is_train
        self.hq_im_file_list = []
        self.lq_im_file_list = []
        self.hq_train_files = {}
        self.lq_train_files = {}
        for dir_path, _, file_names in os.walk(root_dir):
            for f_paths in sorted(file_names):
                if dir_path.endswith('GT'):
                    self.hq_im_file_list.append(os.path.join(dir_path,f_paths))
                elif dir_path.endswith('input'):
                    self.lq_im_file_list.append(os.path.join(dir_path,f_paths))
        for im_names in self.hq_im_file_list:
            self.hq_train_files[im_names] = Image.open(im_names).convert('RGB')
        for im_names in self.lq_im_file_list:
            self.lq_train_files[im_names] = Image.open(im_names).convert('RGB')
        if is_train:
            self.train_transform = T.Compose([T.RandomCrop((im_shape[0],im_shape[1])),T.RandomHorizontalFlip(p=0.2),T.RandomVerticalFlip(p=0.3)])
        self.tensor_transform = T.Compose([T.ToTensor(),T.Normalize(mean=[0.64, 0.6, 0.58],std=[0.14,0.15, 0.152])])
        self.gt_transform = T.ToTensor()
        self.blur = T.GaussianBlur((5,5))
    def __len__(self):
        return len(self.hq_im_file_list)

    def __getitem__(self, idx):
        rand_num  = random.randint(0,10)
        image_hq_fname = self.hq_train_files[self.hq_im_file_list[idx]]
        image_lq_fname = self.lq_train_files[self.lq_im_file_list[idx]]
        hq_image = self.gt_transform(image_hq_fname).unsqueeze(dim=0)
        lq_image = self.tensor_transform(image_lq_fname).unsqueeze(dim=0)
        concat_img = torch.cat([hq_image,lq_image],dim=0)
        if self.is_train:
            image = self.train_transform(concat_img)
            if rand_num==5:
                image = self.blur(image)
            
        else:
            image = concat_img
        hq_img,lq_img = image.tensor_split(2)
        return lq_img.squeeze(0),hq_img.squeeze(0)
