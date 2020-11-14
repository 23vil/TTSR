import os
from imageio import imread
from PIL import Image
import numpy as np
import glob
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")



class RandomRotate(object):
    def __call__(self, sample):
        k1 = np.random.randint(0, 4)
        sample['LR'] = np.rot90(sample['LR'], k1).copy()  # numpy.rot90(m, k=1, axes=(0, 1))---- Rotate an array by 90 degree k times -- In this case k = random number between 0 and 3
        sample['HR'] = np.rot90(sample['HR'], k1).copy()
        sample['LR_sr'] = np.rot90(sample['LR_sr'], k1).copy()
        k2 = np.random.randint(0, 4)
        sample['Ref'] = np.rot90(sample['Ref'], k2).copy()
        sample['Ref_sr'] = np.rot90(sample['Ref_sr'], k2).copy()
        return sample


class RandomFlip(object):
    def __call__(self, sample):
        if (np.random.randint(0, 2) == 1): #np.random.randint(0,2) can be 0 or 1.... Some Objects flipped others not
            sample['LR'] = np.fliplr(sample['LR']).copy()  #np.fliplr -- Flip array in the left/right direction.
            sample['HR'] = np.fliplr(sample['HR']).copy()
            sample['LR_sr'] = np.fliplr(sample['LR_sr']).copy()
        if (np.random.randint(0, 2) == 1):
            sample['Ref'] = np.fliplr(sample['Ref']).copy()
            sample['Ref_sr'] = np.fliplr(sample['Ref_sr']).copy()
        if (np.random.randint(0, 2) == 1):
            sample['LR'] = np.flipud(sample['LR']).copy()
            sample['HR'] = np.flipud(sample['HR']).copy()
            sample['LR_sr'] = np.flipud(sample['LR_sr']).copy()
        if (np.random.randint(0, 2) == 1):
            sample['Ref'] = np.flipud(sample['Ref']).copy()
            sample['Ref_sr'] = np.flipud(sample['Ref_sr']).copy()
        return sample


class ToTensor(object):
    def __call__(self, sample):
        LR, LR_sr, HR, Ref, Ref_sr = sample['LR'], sample['LR_sr'], sample['HR'], sample['Ref'], sample['Ref_sr']
        LR = LR.transpose((2,0,1)) #
        LR_sr = LR_sr.transpose((2,0,1))
        HR = HR.transpose((2,0,1))
        Ref = Ref.transpose((2,0,1))
        Ref_sr = Ref_sr.transpose((2,0,1))
        return {'LR': torch.from_numpy(LR).float(),
                'LR_sr': torch.from_numpy(LR_sr).float(),
                'HR': torch.from_numpy(HR).float(),
                'Ref': torch.from_numpy(Ref).float(),
                'Ref_sr': torch.from_numpy(Ref_sr).float()}
                
class RandomCrop(object):
    def __call__(self, sample):
        h,w = sample.shape[:2]
        hStart = torch.randint(0, h-160,(1,))
        wStart = torch.randint(0, w-160,(1,))
        crop = sample[hStart:hStart+160,wStart:wStart+160]
        return crop                


class TrainSet(Dataset):
    def __init__(self, args, transform=transforms.Compose([RandomFlip(), RandomRotate(), ToTensor()]) ):
        self.input_list = sorted([os.path.join(args.dataset_dir, 'train/input/', name) for name in #added / to end of path to make it windows compatible
            os.listdir( os.path.join(args.dataset_dir, 'train/input/') )]) # lists names of files in directory
        self.ref_list = sorted([os.path.join(args.dataset_dir, 'train/ref/', name) for name in #added / to end of path to make it windows compatible
            os.listdir( os.path.join(args.dataset_dir, 'train/ref/') )])  #added / to end of path to make it windows compatible
        self.transform = transform

    def __len__(self):
        return len(self.input_list)
        

    def __getitem__(self, idx):
        croper = RandomCrop() #initialize RandomCrop
        
        ### HR       
        HRpre = imread(self.input_list[idx])        
        HRpre = croper(HRpre) #Take a Random 160 x 160 Crop
        HR = np.array([HRpre, HRpre, HRpre]).transpose(2,1,0) #make RGB image from greyscale imput----- Solve differently later!
        #HR = imread(self.input_list[idx])
        #HR = np.array(Image.fromarray(HR).resize((160, 160), Image.BICUBIC))
        h,w = HR.shape[:2]

        #HR = HR[:h//4*4, :w//4*4, :]
        ### LR and LR_sr
        LR = np.array(Image.fromarray(HR).resize((w//4, h//4), Image.BICUBIC)) #HR image to LR image via bicubic... to 1/4 of height and width   --- Image.fromarray makes PILLOW Image from original
        LR_sr = np.array(Image.fromarray(LR).resize((w, h), Image.BICUBIC)) # no resize ??

        ### Ref and Ref_sr
        Ref_sub_pre = imread(self.ref_list[idx])
        Ref_sub_pre = croper(Ref_sub_pre) #Take a Random 160 x 160 Crop
        Ref_sub = np.array([Ref_sub_pre, Ref_sub_pre, Ref_sub_pre]).transpose(2,1,0) #make RGB image from greyscale imput----- Solve differently later!
        
        #Ref_sub = imread(self.ref_list[idx])
        #Ref_sub = np.array(Image.fromarray(Ref_sub).resize((160, 160), Image.BICUBIC)) # Bicubic degradation .. delete!
        
        h2, w2 = Ref_sub.shape[:2]
        Ref_sr_sub = np.array(Image.fromarray(Ref_sub).resize((w2//4, h2//4), Image.BICUBIC))
        Ref_sr_sub = np.array(Image.fromarray(Ref_sr_sub).resize((w2, h2), Image.BICUBIC))
    
        ### complete ref and ref_sr to the same size, to use batch_size > 1
        Ref = np.zeros((160, 160, 3))
        Ref_sr = np.zeros((160, 160, 3))
        Ref[:h2, :w2] = Ref_sub
        Ref_sr[:h2, :w2] = Ref_sr_sub

        ### change type
        LR = LR.astype(np.float32)
        LR_sr = LR_sr.astype(np.float32)
        HR = HR.astype(np.float32)
        Ref = Ref.astype(np.float32)
        Ref_sr = Ref_sr.astype(np.float32)

        ### rgb range to [-1, 1]
        LR = LR / 127.5 - 1.
        LR_sr = LR_sr / 127.5 - 1.
        HR = HR / 127.5 - 1.
        Ref = Ref / 127.5 - 1.
        Ref_sr = Ref_sr / 127.5 - 1.

        sample = {'LR': LR,  
                  'LR_sr': LR_sr,
                  'HR': HR,
                  'Ref': Ref, 
                  'Ref_sr': Ref_sr}

        if self.transform:
            sample = self.transform(sample)
        return sample


class TestSet(Dataset):
    def __init__(self, args, ref_level='1', transform=transforms.Compose([ToTensor()])):
        self.input_list = sorted(glob.glob(os.path.join(args.dataset_dir, 'test/CUFED5/', '*_0.tif'))) #path had to be altered manually --- Why *_0.png files for test?
        self.ref_list = sorted(glob.glob(os.path.join(args.dataset_dir, 'test/CUFED5/', #path had to be altered manually
            '*_' + ref_level + '.tif')))
        self.transform = transform
    #def __init__(self, args, ref_level='1', transform=transforms.Compose([ToTensor()])):
     #   self.input_list = sorted(glob.glob(os.path.join(args.dataset_dir, 'test/IMM/', '*.png'))) #path had to be altered manually --- Why *_0.png files for test?
      #  self.ref_list = sorted(glob.glob(os.path.join(args.dataset_dir, 'test/IMM/', #path had to be altered manually
       #     '*_' + ref_level + '.png')))
        #self.transform = transform 
     
     
    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        croper = RandomCrop() #initialize RandomCrop
        ### HR
        HRpre = imread(self.input_list[idx])
        HRpre = croper(HRpre) #Take a Random 160 x 160 Crop
        HR = np.array([HRpre, HRpre, HRpre]).transpose(2,1,0) #make RGB image from greyscale imput----- Solve differently later!
        #HR = imread(self.input_list[idx])
        #HR = HR.transforms.RandomCrop((160,160,3), padding=None, pad_if_needed=False, fill=0, padding_mode='constant')
        #HR = np.array(Image.fromarray(HR).resize((160, 160), Image.BICUBIC))
        
        h, w = HR.shape[:2]
        h, w = h//4*4, w//4*4
        HR = HR[:h, :w, :] ### crop to the multiple of 4

        ### LR and LR_sr
        LR = np.array(Image.fromarray(HR).resize((w//4, h//4), Image.BICUBIC))
        LR_sr = np.array(Image.fromarray(LR).resize((w, h), Image.BICUBIC))

        ### Ref and Ref_sr
        Ref_pre = imread(self.ref_list[idx])
        Ref_pre = croper(Ref_pre) #Take a Random 160 x 160 Crop
        Ref = np.array([Ref_pre, Ref_pre, Ref_pre]).transpose(2,1,0) #make RGB image from greyscale imput----- Solve differently later!
        #Ref_sub = imread(self.ref_list[idx])
        #Ref = Ref.transforms.RandomCrop((160,160,3), padding=None, pad_if_needed=False, fill=0, padding_mode='constant')
        #Ref = np.array(Image.fromarray(Ref).resize((160, 160), Image.BICUBIC))
        
        h2, w2 = Ref.shape[:2]
        h2, w2 = h2//4*4, w2//4*4
        Ref = Ref[:h2, :w2, :]
        Ref_sr = np.array(Image.fromarray(Ref).resize((w2//4, h2//4), Image.BICUBIC))
        Ref_sr = np.array(Image.fromarray(Ref_sr).resize((w2, h2), Image.BICUBIC))

        ### change type
        LR = LR.astype(np.float32)
        LR_sr = LR_sr.astype(np.float32)
        HR = HR.astype(np.float32)
        Ref = Ref.astype(np.float32)
        Ref_sr = Ref_sr.astype(np.float32)

        ### rgb range to [-1, 1]
        LR = LR / 127.5 - 1.
        LR_sr = LR_sr / 127.5 - 1.
        HR = HR / 127.5 - 1.
        Ref = Ref / 127.5 - 1.
        Ref_sr = Ref_sr / 127.5 - 1.

        sample = {'LR': LR,  
                  'LR_sr': LR_sr,
                  'HR': HR,
                  'Ref': Ref, 
                  'Ref_sr': Ref_sr}

        if self.transform:
            sample = self.transform(sample)
        return sample