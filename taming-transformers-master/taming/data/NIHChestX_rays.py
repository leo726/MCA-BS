import os, tarfile, glob, shutil
from cv2 import dft
import yaml
import numpy as np
from tqdm import tqdm
from PIL import Image
import albumentations
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from taming.data.base import ImagePaths
from taming.util import download, retrieve
import taming.data.utils as bdu
import pandas as pd

class ChestxrayBase(Dataset):
    def __init__(self, config, train):
        self.config = config or OmegaConf.create()

        if train:
            self.image_list = open(self.config['train'].train_path).readlines()
            self.base_path =  self.config['train'].base_path
        else:
            self.image_list = open(self.config['validation'].test_path).readlines()
            self.base_path = self.config['validation'].base_path

        self.label_dict = {}
        df = pd.read_csv(self.config['train'].csv_path)
        for i, j in zip(df['Image Index'], df['Finding Labels']):
            self.label_dict[i] = j
            
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

        # label = self.label_dict[self.image_list[i].strip()]

        # return {'image': pixel_values, 'label':label}
        return {'image': pixel_values}

def create_chest_datasets(config):
    train = ChestxrayBase(config, train=True)
    test = ChestxrayBase(config, train=False)
    return train, test

class ChestxrayDepth(Dataset):
    def __init__(self, config, train):
        self.config = config or OmegaConf.create()

        if train:
            self.image_list = open(self.config['train'].train_path).readlines()
            self.base_path =  self.config['train'].base_path
        else:
            self.image_list = open(self.config['validation'].test_path).readlines()
            self.base_path = self.config['validation'].base_path

        self.label_dict = {}
        df = pd.read_csv(self.config['train'].csv_path)
        for i, j in zip(df['Image Index'], df['Finding Labels']):
            self.label_dict[i] = j
            
        self.transform = None
        
        
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, i):
        image_path = os.path.join(self.base_path, self.image_list[i].strip())
        image = Image.open(image_path)
        image_resize=image.resize((256, 256), Image.ANTIALIAS)
        if not image_resize.mode == "RGBA":
                image_resize = image_resize.convert("RGBA")
        x = np.array(image_resize).astype(np.uint8)
        y = x.copy()
        y.dtype = np.float32
        y = y.reshape(x.shape[:2])
        depth = np.ascontiguousarray(y)
        depth = (depth - depth.min())/max(1e-8, depth.max()-depth.min())
        depth = 2.0*depth-1.0

        label = self.label_dict[self.image_list[i].strip()]

        return {'depth': depth, 'label':label}
    
def create_depth_datasets(config):
    train = ChestxrayDepth(config, train=True)
    test = ChestxrayDepth(config, train=False)
    return train, test