import argparse
import math
import os
import random
import re
from subprocess import Popen, PIPE
import sys
from urllib.request import urlopen
# Supress warnings
import warnings
from scipy import spatial
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

import imageio
from tqdm import tqdm
import kornia.augmentation as K
import numpy as np
from omegaconf import OmegaConf
from PIL import ImageFile, Image, PngImagePlugin
ImageFile.LOAD_TRUNCATED_IMAGES = True

from tqdm import tqdm

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.cuda import get_device_properties
torch.backends.cudnn.benchmark = False
from torch_optimizer import DiffGrad, AdamP, RAdam
from torchvision import transforms
from torchvision.transforms import functional as TF

from CLIP import clip
from utils import *


# Check for GPU and reduce the default image size if low VRAM
default_image_size = 512  # >8GB VRAM
if not torch.cuda.is_available():
    default_image_size = 256  # no GPU found
elif get_device_properties(0).total_memory <= 2 ** 33:  # 2 ** 33 = 8,589,934,592 bytes = 8 GB
    default_image_size = 318  # <8GB VRAM

# Create the parser
parser = argparse.ArgumentParser(description='Similarity Score Analysis')

# Add the arguments
parser.add_argument("-p",    "--prompts", type=str, help="Text prompts", dest='prompts')
parser.add_argument("-dir", type=str, help="Directory for images", required=True)
parser.add_argument("-cd",   "--cuda_device", type=str, help="Cuda device to use", default="cuda:0", dest='cuda_device')
parser.add_argument("-m",    "--clip_model", type=str, help="CLIP model (e.g. ViT-B/32, ViT-B/16)", default='ViT-B/32', dest='clip_model')
parser.add_argument("-s",    "--size", nargs=2, type=int, help="Image size (width height) (default: %(default)s)", default=[default_image_size,default_image_size], dest='size')
parser.add_argument("-cuts", "--num_cuts", type=int, help="Number of cuts", default=1, dest='cutn')
parser.add_argument("-cutp", "--cut_power", type=float, help="Cut power", default=1., dest='cut_pow')
parser.add_argument("-cutm", "--cut_method", type=str, help="Cut method", choices=['original','updated','updatedpooling','latest'], default='latest', dest='cut_method')
parser.add_argument("-aug",  "--augments", nargs='+', action='append', type=str, choices=['Ji','Sh','Gn','Pe','Ro','Af','Et','Ts','Cr','Er','Re'], help="Enabled augments", default=[], dest='augments')


class Prompt(nn.Module):
    def __init__(self, embed, weight=1., stop=float('-inf')):
        super().__init__()
        self.register_buffer('embed', embed)
        self.register_buffer('weight', torch.as_tensor(weight))
        self.register_buffer('stop', torch.as_tensor(stop))

    def forward(self, input):
        input_normed = F.normalize(input.unsqueeze(1), dim=2)
        embed_normed = F.normalize(self.embed.unsqueeze(0), dim=2)
        dists = input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)
        dists = dists * self.weight.sign()
        return self.weight.abs() * replace_grad(dists, torch.maximum(dists, self.stop)).mean()



class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, args, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow
        
        # Pick your own augments & their order
        augment_list = []
        for item in args.augments[0]:
            if item == 'Ji':
                augment_list.append(K.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.7))
            elif item == 'Sh':
                augment_list.append(K.RandomSharpness(sharpness=0.3, p=0.5))
            elif item == 'Gn':
                augment_list.append(K.RandomGaussianNoise(mean=0.0, std=1., p=0.5))
            elif item == 'Pe':
                augment_list.append(K.RandomPerspective(distortion_scale=0.7, p=0.7))
            elif item == 'Ro':
                augment_list.append(K.RandomRotation(degrees=15, p=0.7))
            elif item == 'Af':
                augment_list.append(K.RandomAffine(degrees=15, translate=0.1, shear=5, p=0.7, padding_mode='zeros', keepdim=True)) # border, reflection, zeros
            elif item == 'Et':
                augment_list.append(K.RandomElasticTransform(p=0.7))
            elif item == 'Ts':
                augment_list.append(K.RandomThinPlateSpline(scale=0.8, same_on_batch=True, p=0.7))
            elif item == 'Cr':
                augment_list.append(K.RandomCrop(size=(self.cut_size,self.cut_size), pad_if_needed=True, padding_mode='reflect', p=0.5))
            elif item == 'Er':
                augment_list.append(K.RandomErasing(scale=(.1, .4), ratio=(.3, 1/.3), same_on_batch=True, p=0.7))
            elif item == 'Re':
                augment_list.append(K.RandomResizedCrop(size=(self.cut_size,self.cut_size), scale=(0.1,1),  ratio=(0.75,1.333), cropping_mode='resample', p=0.5))
        
        # Uncomment if you like seeing the list ;)
        # print(augment_list)
        
        self.augs = nn.Sequential(*augment_list)
        self.noise_fac = 0.1
        # self.noise_fac = False
        
        # Pooling
        self.av_pool = nn.AdaptiveAvgPool2d((self.cut_size, self.cut_size))
        self.max_pool = nn.AdaptiveMaxPool2d((self.cut_size, self.cut_size))

    def forward(self, input):
        cutouts = []
        
        for _ in range(self.cutn):            
            # Use Pooling
            cutout = (self.av_pool(input) + self.max_pool(input))/2
            cutouts.append(cutout)
            
        batch = self.augs(torch.cat(cutouts, dim=0))
        
        if self.noise_fac:
            facs = batch.new_empty([self.cutn, 1, 1, 1]).uniform_(0, self.noise_fac)
            batch = batch + facs * torch.randn_like(batch)
        return batch


# An updated version with Kornia augments and pooling (where my version started):
class MakeCutoutsPoolingUpdate(nn.Module):
    def __init__(self, cut_size, cutn, args, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

        self.augs = nn.Sequential(
            K.RandomAffine(degrees=15, translate=0.1, p=0.7, padding_mode='border'),
            K.RandomPerspective(0.7,p=0.7),
            K.ColorJitter(hue=0.1, saturation=0.1, p=0.7),
            K.RandomErasing((.1, .4), (.3, 1/.3), same_on_batch=True, p=0.7),            
        )
        
        self.noise_fac = 0.1
        self.av_pool = nn.AdaptiveAvgPool2d((self.cut_size, self.cut_size))
        self.max_pool = nn.AdaptiveMaxPool2d((self.cut_size, self.cut_size))

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        
        for _ in range(self.cutn):
            cutout = (self.av_pool(input) + self.max_pool(input))/2
            cutouts.append(cutout)
            
        batch = self.augs(torch.cat(cutouts, dim=0))
        
        if self.noise_fac:
            facs = batch.new_empty([self.cutn, 1, 1, 1]).uniform_(0, self.noise_fac)
            batch = batch + facs * torch.randn_like(batch)
        return batch


# An updated version with Kornia augments, but no pooling:
class MakeCutoutsUpdate(nn.Module):
    def __init__(self, cut_size, cutn, args, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow
        self.augs = nn.Sequential(
            K.RandomHorizontalFlip(p=0.5),
            K.ColorJitter(hue=0.01, saturation=0.01, p=0.7),
            # K.RandomSolarize(0.01, 0.01, p=0.7),
            K.RandomSharpness(0.3,p=0.4),
            K.RandomAffine(degrees=30, translate=0.1, p=0.8, padding_mode='border'),
            K.RandomPerspective(0.2,p=0.4),)
        self.noise_fac = 0.1


    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
        batch = self.augs(torch.cat(cutouts, dim=0))
        if self.noise_fac:
            facs = batch.new_empty([self.cutn, 1, 1, 1]).uniform_(0, self.noise_fac)
            batch = batch + facs * torch.randn_like(batch)
        return batch


# This is the original version (No pooling)
class MakeCutoutsOrig(nn.Module):
    def __init__(self, cut_size, cutn, args, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
        return clamp_with_grad(torch.cat(cutouts, dim=0), 0, 1)


def main():
    # Execute the parse_args() method
    args = parser.parse_args()
    if not args.prompts:
        args.prompts = "A cute, smiling, Nerdy Rodent"

    if not args.augments:
        args.augments = [['Af', 'Pe', 'Ji', 'Er']]

    # Do it
    device = torch.device(args.cuda_device)
    jit = True if float(torch.__version__[:3]) < 1.8 else False
    perceptor = clip.load(args.clip_model, jit=jit)[0].eval().requires_grad_(False).to(device)

    cut_size = perceptor.visual.input_resolution

    if args.cut_method == 'latest':
        make_cutouts = MakeCutouts(cut_size, args.cutn, args, cut_pow=args.cut_pow)
    elif args.cut_method == 'original':
        make_cutouts = MakeCutoutsOrig(cut_size, args.cutn, args, cut_pow=args.cut_pow)
    elif args.cut_method == 'updated':
        make_cutouts = MakeCutoutsUpdate(cut_size, args.cutn, args, cut_pow=args.cut_pow)
    else:
        make_cutouts = MakeCutoutsPoolingUpdate(cut_size, args.cutn, args, cut_pow=args.cut_pow) 

    sideX, sideY = args.size[0], args.size[1]

    male_prompt = "a photo of a male"
    female_prompt = "a photo of a female"

    # Split text prompts using the pipe character (weights are split later)
    # For stories, there will be many phrases
    male_story_phrases = [phrase.strip() for phrase in male_prompt.split("^")]
    
    # Make a list of all phrases
    male_all_phrases = []
    for phrase in male_story_phrases:
        male_all_phrases.append(phrase.split("|"))
    
    # First phrase
    male_prompt = male_all_phrases[0]

    female_story_phrases = [phrase.strip() for phrase in female_prompt.split("^")]
    
    # Make a list of all phrases
    female_all_phrases = []
    for phrase in female_story_phrases:
        female_all_phrases.append(phrase.split("|"))
    
    # First phrase
    male_prompt = male_all_phrases[0]
    female_prompt = female_all_phrases[0]

    # CLIP tokenize/encode   
    male_txt, male_weight, male_stop = split_prompt(male_prompt[0])
    male_prompt_embed = perceptor.encode_text(clip.tokenize(male_txt).to(device)).float()

    female_txt, female_weight, female_stop = split_prompt(female_prompt[0])
    female_prompt_embed = perceptor.encode_text(clip.tokenize(female_txt).to(device)).float()

    results = []
    # Image initialisation
    for img_file in tqdm(os.listdir(args.dir)):
        # print(f'File {args.dir}/{img_file}')

        pMs = []
        normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                        std=[0.26862954, 0.26130258, 0.27577711])

        # From imagenet - Which is better?
        #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                  std=[0.229, 0.224, 0.225])

        img = Image.open(f'{args.dir}/{img_file}')
        pil_image = img.convert('RGB')
        img = resize_image(pil_image, (sideX, sideY))
        batch = make_cutouts(TF.to_tensor(img).unsqueeze(0).to(device))
        img_embed = perceptor.encode_image(normalize(batch)).float()

        male_cosine = 1 - spatial.distance.cosine(male_prompt_embed.cpu().numpy().flatten(), img_embed.cpu().numpy().flatten())
        female_cosine = 1 - spatial.distance.cosine(female_prompt_embed.cpu().numpy().flatten(), img_embed.cpu().numpy().flatten())

        # 0 is female; 1 is male
        if(male_cosine > female_cosine):
            results.append((f'{img_file}', 1))
        else:
            results.append((f'{img_file}', 0))

    num_correct = 0
    for file, pred in results:
        if('female' in img_file and pred == 0):
            num_correct += 1
    
    print(num_correct / len(results))

    # _, bin_edges = np.histogram(fem_results['female'] + fem_results['male'])  
    # plt.hist(fem_results['female'], bins=bin_edges, label='Female', color='blue', alpha=0.4)
    # plt.hist(fem_results['male'], bins=bin_edges, label='Male', color='red', alpha=0.4)

    # plt.savefig('test.png')


if __name__ == '__main__':
    main()
