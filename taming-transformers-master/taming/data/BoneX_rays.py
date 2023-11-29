import os, tarfile, glob, shutil
from cv2 import dft
import yaml
import numpy as np
from tqdm import tqdm
from PIL import Image
import albumentations
from omegaconf import OmegaConf
from torch.utils.data import Dataset
import random
from taming.data.base import ImagePaths
from taming.util import download, retrieve
import taming.data.utils as bdu
import pandas as pd
from torchvision import transforms

class BonexrayBase(Dataset):
    def __init__(self, config, train):
        self.config = config or OmegaConf.create()

        if train:
            self.image_list = open(self.config['train'].train_path).readlines()
            self.base_path =  self.config['train'].base_path
            r = random.uniform(0.9,1)
            r_size = round(256*r)
            self.transform = transforms.Compose([
                                                 transforms.RandomRotation(15),
                                                 transforms.Resize((256,256)),
                                                 transforms.CenterCrop(r_size),
                                                 transforms.Resize((256,256)),
                                                 transforms.RandomHorizontalFlip(p=0.5),
                                                #  transforms.RandomVerticalFlip(p=0.5),
                                                #  transforms.ToTensor(),
                                                #  transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]),
                                                #  transforms.ToPILImage(),
                                                ])
            
        else:
            self.image_list = open(self.config['validation'].test_path).readlines()
            self.base_path = self.config['validation'].base_path
            self.transform = transforms.Compose([
                                                 transforms.Resize((256,256)),
                                                ])
            
        
        
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, i):
        image_path = os.path.join(self.base_path, self.image_list[i].strip())
        image = Image.open(image_path)
        image_resize = self.transform(image) # 剪裁、旋转
        # image_resize=image.resize((256, 256), Image.ANTIALIAS)
        
        if not image_resize.mode == "RGB":
                image_resize = image_resize.convert("RGB")
        pixel_values = np.array(image_resize).astype(np.uint8)
        pixel_values = (pixel_values/127.5 - 1.0).astype(np.float32)
        #黑白颠倒
        pixel_values = -pixel_values

        return {'image': pixel_values}

def create_datasets(config):
    train = BonexrayBase(config, train=True)
    test = BonexrayBase(config, train=False)
    return train, test

class TotalBonexrayBase(Dataset):
    def __init__(self, config, train):
        self.config = config or OmegaConf.create()
        if train:
               self.image_list = open(self.config['train'].train_path).readlines()
               self.bone_base_path =  self.config['train'].bone_base_path
               self.bse_base_path = self.config['train'].bse_base_path
               r = random.uniform(0.9,1)
               r_size = round(256*r)
               number_list = ['0', '1']
               p = int(random.choice(number_list))
               self.transform = transforms.Compose([
                                                #  transforms.RandomRotation(15),
                                                 transforms.Resize((256,256)),
                                                 transforms.CenterCrop(r_size),
                                                 transforms.Resize((256,256)),
                                                 transforms.RandomHorizontalFlip(p),
                                                #  transforms.RandomVerticalFlip(p=0.5),
                                                #  transforms.ToTensor(),
                                                #  transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]),
                                                #  transforms.ToPILImage(),
                                                ])
        else:
               self.image_list = open(self.config['validation'].test_path).readlines()
               self.bone_base_path =  self.config['validation'].bone_base_path
               self.bse_base_path = self.config['validation'].bse_base_path
               self.transform = transforms.Compose([
                                                 transforms.Resize((256,256)),
                                                ])
            
        # self.transform = None
        
        
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, i):
        bone_base_image_path = os.path.join(self.bone_base_path, self.image_list[i].strip())
        bse_base_image_path = os.path.join(self.bse_base_path, self.image_list[i].strip())
        bone_base_image = Image.open(bone_base_image_path)
        bse_base_image = Image.open(bse_base_image_path)
        bone_base_image_resize = self.transform(bone_base_image)
        bse_base_image_resize = self.transform(bse_base_image)
        # bone_base_image_resize=bone_base_image.resize((256, 256), Image.ANTIALIAS)
        # bse_base_image_resize=bse_base_image.resize((256, 256), Image.ANTIALIAS)

        # image_resize = Image.open(image_path) # if not resize
        if not bone_base_image_resize.mode == "RGB":
                bone_base_image_resize = bone_base_image_resize.convert("RGB")
        bone_base_pixel_values = np.array(bone_base_image_resize).astype(np.uint8)
        bone_base_pixel_values = (bone_base_pixel_values/127.5 - 1.0).astype(np.float32)
        # 黑白颠倒
        bone_base_pixel_values = -bone_base_pixel_values
        if not bse_base_image_resize.mode == "RGB":
                bse_base_image_resize = bse_base_image_resize.convert("RGB")
        bse_base_pixel_values = np.array(bse_base_image_resize).astype(np.uint8)
        bse_base_pixel_values = (bse_base_pixel_values/127.5 - 1.0).astype(np.float32)
        # 黑白颠倒
        bse_base_pixel_values = -bse_base_pixel_values

        return {'bone_base_image': bone_base_pixel_values,
                'bse_base_image': bse_base_pixel_values}

def create_total_datasets(config):
    train = TotalBonexrayBase(config, train=True)
    validation = TotalBonexrayBase(config, train=False)
    return train, validation

class ChestxrayBase(Dataset):
    def __init__(self, config, train):
        self.config = config or OmegaConf.create()

        if train:
            self.image_list = open(self.config['train'].train_path).readlines()
            self.base_path =  self.config['train'].base_path
        else:
            self.image_list = open(self.config['validation'].test_path).readlines()
            self.base_path = self.config['validation'].base_path
            
        self.transform = None
        
        
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, i):
        image_path = os.path.join(self.base_path, self.image_list[i].strip())
        image = Image.open(image_path)
        image_resize=image.resize((256, 256), Image.ANTIALIAS)
        # image_resize = Image.open(image_path) # if not resize
        if not image_resize.mode == "RGB":
                image_resize = image_resize.convert("RGB")
        pixel_values = np.array(image_resize).astype(np.uint8)
        pixel_values = (pixel_values/127.5 - 1.0).astype(np.float32)
        #pixel_values = self.transform(pixel_values)

        return {'image': pixel_values}

def create_chest_datasets(config):
    train = ChestxrayBase(config, train=True)
    test = ChestxrayBase(config, train=False)
    return train, test