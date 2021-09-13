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
warnings.filterwarnings('ignore')

import imageio
from tqdm import tqdm
import kornia.augmentation as K
import numpy as np
from omegaconf import OmegaConf
from PIL import ImageFile, Image, PngImagePlugin
ImageFile.LOAD_TRUNCATED_IMAGES = True

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
vq_parser = argparse.ArgumentParser(description='Image generation using VQGAN+CLIP')

# Add the arguments
vq_parser.add_argument("-p",    "--prompts", type=str, help="Text prompts", default=None, dest='prompts')
vq_parser.add_argument("-ip",   "--image_prompts", type=str, help="Image prompts / target image", default=[], dest='image_prompts')
vq_parser.add_argument("-i",    "--iterations", type=int, help="Number of iterations", default=500, dest='max_iterations')
vq_parser.add_argument("-se",   "--save_every", type=int, help="Save image iterations", default=50, dest='display_freq')
vq_parser.add_argument("-s",    "--size", nargs=2, type=int, help="Image size (width height) (default: %(default)s)", default=[default_image_size,default_image_size], dest='size')
vq_parser.add_argument("-ii",   "--init_image", type=str, help="Initial image", default=None, dest='init_image')
vq_parser.add_argument("-in",   "--init_noise", type=str, help="Initial noise image (pixels or gradient)", default="pixels", dest='init_noise')
vq_parser.add_argument("-iw",   "--init_weight", type=float, help="Initial weight", default=0., dest='init_weight')
vq_parser.add_argument("-m",    "--clip_model", type=str, help="CLIP model (e.g. ViT-B/32, ViT-B/16)", default='ViT-B/32', dest='clip_model')
vq_parser.add_argument("-conf", "--vqgan_config", type=str, help="VQGAN config", default=f'checkpoints/vqgan_imagenet_f16_16384.yaml', dest='vqgan_config')
vq_parser.add_argument("-ckpt", "--vqgan_checkpoint", type=str, help="VQGAN checkpoint", default=f'checkpoints/vqgan_imagenet_f16_16384.ckpt', dest='vqgan_checkpoint')
vq_parser.add_argument("-nps",  "--noise_prompt_seeds", nargs="*", type=int, help="Noise prompt seeds", default=[], dest='noise_prompt_seeds')
vq_parser.add_argument("-npw",  "--noise_prompt_weights", nargs="*", type=float, help="Noise prompt weights", default=[], dest='noise_prompt_weights')
vq_parser.add_argument("-lr",   "--learning_rate", type=float, help="Learning rate", default=0.1, dest='step_size')
vq_parser.add_argument("-cutm", "--cut_method", type=str, help="Cut method", choices=['original','updated','updatedpooling','latest'], default='latest', dest='cut_method')
vq_parser.add_argument("-cuts", "--num_cuts", type=int, help="Number of cuts", default=32, dest='cutn')
vq_parser.add_argument("-cutp", "--cut_power", type=float, help="Cut power", default=1., dest='cut_pow')
vq_parser.add_argument("-sd",   "--seed", type=int, help="Seed", default=None, dest='seed')
vq_parser.add_argument("-opt",  "--optimiser", type=str, help="Optimiser", choices=['Adam','AdamW','Adagrad','Adamax','DiffGrad','AdamP','RAdam','RMSprop'], default='Adam', dest='optimiser')
vq_parser.add_argument("-o",    "--output", type=str, help="Output file", default="output.png", dest='output')
vq_parser.add_argument("-vid",  "--video", action='store_true', help="Create video frames?", dest='make_video')
vq_parser.add_argument("-zvid", "--zoom_video", action='store_true', help="Create zoom video?", dest='make_zoom_video')
vq_parser.add_argument("-zs",   "--zoom_start", type=int, help="Zoom start iteration", default=0, dest='zoom_start')
vq_parser.add_argument("-zse",  "--zoom_save_every", type=int, help="Save zoom image iterations", default=10, dest='zoom_frequency')
vq_parser.add_argument("-zsc",  "--zoom_scale", type=float, help="Zoom scale", default=0.99, dest='zoom_scale')
vq_parser.add_argument("-cpe",  "--change_prompt_every", type=int, help="Prompt change frequency", default=0, dest='prompt_frequency')
vq_parser.add_argument("-vl",   "--video_length", type=float, help="Video length in seconds (not interpolated)", default=10, dest='video_length')
vq_parser.add_argument("-ofps", "--output_video_fps", type=float, help="Create an interpolated video (Nvidia GPU only) with this fps (min 10. best set to 30 or 60)", default=0, dest='output_video_fps')
vq_parser.add_argument("-ifps", "--input_video_fps", type=float, help="When creating an interpolated video, use this as the input fps to interpolate from (>0 & <ofps)", default=15, dest='input_video_fps')
vq_parser.add_argument("-d",    "--deterministic", action='store_true', help="Enable cudnn.deterministic?", dest='cudnn_determinism')
vq_parser.add_argument("-aug",  "--augments", nargs='+', action='append', type=str, choices=['Ji','Sh','Gn','Pe','Ro','Af','Et','Ts','Cr','Er','Re'], help="Enabled augments", default=[], dest='augments')
vq_parser.add_argument("-cd",   "--cuda_device", type=str, help="Cuda device to use", default="cuda:0", dest='cuda_device')



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

def synth(z):
    return z

@torch.no_grad()
def checkin(i, losses, z, args):
    losses_str = ', '.join(f'{loss.item():g}' for loss in losses)
    tqdm.write(f'i: {i}, loss: {sum(losses).item():g}, losses: {losses_str}')
    out = z
    info = PngImagePlugin.PngInfo()
    info.add_text('comment', f'{args.prompts}')
    TF.to_pil_image(out[0].cpu()).save(args.output, pnginfo=info) 	


def ascend_txt(i, z, perceptor, normalize, make_cutouts, args, pMs):
    out = z
    iii = perceptor.encode_image(normalize(make_cutouts(out))).float()
    
    result = []

    if args.init_weight:
        # result.append(F.mse_loss(z, z_orig) * args.init_weight / 2)
        result.append(F.mse_loss(z, torch.zeros_like(z_orig)) * ((1/torch.tensor(i*2 + 1))*args.init_weight) / 2)

    for prompt in pMs:
        result.append(prompt(iii))
    
    if args.make_video:    
        img = np.array(out.mul(255).clamp(0, 255)[0].cpu().detach().numpy().astype(np.uint8))[:,:,:]
        img = np.transpose(img, (1, 2, 0))
        imageio.imwrite('./steps/' + str(i) + '.png', np.array(img))

    return result # return loss


def train(i, opt, z, perceptor, normalize, make_cutouts, args, pMs, z_min, z_max):
    opt.zero_grad(set_to_none=True)
    lossAll = ascend_txt(i, z, perceptor, normalize, make_cutouts, args, pMs)
    
    if i % args.display_freq == 0:
        checkin(i, lossAll, z, args)
       
    loss = sum(lossAll)
    loss.backward()
    opt.step()
    
    with torch.no_grad():
        z.copy_(z.maximum(z_min).minimum(z_max))


def main():
    # Execute the parse_args() method
    args = vq_parser.parse_args()
    if not args.prompts and not args.image_prompts:
        args. prompts = "A cute, smiling, Nerdy Rodent"

    if args.cudnn_determinism:
        torch.backends.cudnn.deterministic = True

    if not args.augments:
        args.augments = [['Af', 'Pe', 'Ji', 'Er']]

    # Split text prompts using the pipe character (weights are split later)
    if args.prompts:
        # For stories, there will be many phrases
        story_phrases = [phrase.strip() for phrase in args.prompts.split("^")]
        
        # Make a list of all phrases
        all_phrases = []
        for phrase in story_phrases:
            all_phrases.append(phrase.split("|"))
        
        # First phrase
        args.prompts = all_phrases[0]
        
    # Split target images using the pipe character (weights are split later)
    if args.image_prompts:
        args.image_prompts = args.image_prompts.split("|")
        args.image_prompts = [image.strip() for image in args.image_prompts]

    if args.make_video and args.make_zoom_video:
        print("Warning: Make video and make zoom video are mutually exclusive.")
        args.make_video = False
        
    # Make video steps directory
    if args.make_video or args.make_zoom_video:
        if not os.path.exists('steps'):
            os.mkdir('steps')

    # Fallback to CPU if CUDA is not found and make sure GPU video rendering is also disabled
    # NB. May not work for AMD cards?
    if not args.cuda_device == 'cpu' and not torch.cuda.is_available():
        args.cuda_device = 'cpu'
        args.video_fps = 0
        print("Warning: No GPU found! Using the CPU instead. The iterations will be slow.")
        print("Perhaps CUDA/ROCm or the right pytorch version is not properly installed?")
    

    # Do it
    device = torch.device(args.cuda_device)
    jit = True if float(torch.__version__[:3]) < 1.8 else False
    perceptor = clip.load(args.clip_model, jit=jit)[0].eval().requires_grad_(False).to(device)

    cut_size = perceptor.visual.input_resolution

    # Cutout class options:
    # 'latest','original','updated' or 'updatedpooling'
    if args.cut_method == 'latest':
        make_cutouts = MakeCutouts(cut_size, args.cutn, args, cut_pow=args.cut_pow)
    elif args.cut_method == 'original':
        make_cutouts = MakeCutoutsOrig(cut_size, args.cutn, args, cut_pow=args.cut_pow)
    elif args.cut_method == 'updated':
        make_cutouts = MakeCutoutsUpdate(cut_size, args.cutn, args, cut_pow=args.cut_pow)
    else:
        make_cutouts = MakeCutoutsPoolingUpdate(cut_size, args.cutn, args, cut_pow=args.cut_pow)    

    sideX, sideY = args.size[0], args.size[1]

    z_min = torch.zeros((sideX, sideY)).to(device)
    z_max = torch.ones((sideX, sideY)).to(device)


    # Image initialisation
    if args.init_image:
        if 'http' in args.init_image:
            img = Image.open(urlopen(args.init_image))
        else:
            img = Image.open(args.init_image)
            pil_image = img.convert('RGB')
            pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
            pil_tensor = TF.to_tensor(pil_image)
            z = pil_tensor.to(device).unsqueeze(0)
    elif args.init_noise == 'pixels':
        img = random_noise_image(args.size[0], args.size[1])    
        pil_image = img.convert('RGB')
        pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
        pil_tensor = TF.to_tensor(pil_image)
        z = pil_tensor.to(device).unsqueeze(0)
    elif args.init_noise == 'gradient':
        img = random_gradient_image(args.size[0], args.size[1])
        pil_image = img.convert('RGB')
        pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
        pil_tensor = TF.to_tensor(pil_image)
        z = pil_tensor.to(device).unsqueeze(0)
    else:
        raise RuntimeError("Choose another option!")
        #z = torch.rand_like(z)*2						# NR: check

    z_orig = z.clone()
    z.requires_grad_(True)

    pMs = []
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                    std=[0.26862954, 0.26130258, 0.27577711])

    # From imagenet - Which is better?
    #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])

    # CLIP tokenize/encode   
    for prompt in args.prompts:
        txt, weight, stop = split_prompt(prompt)
        embed = perceptor.encode_text(clip.tokenize(txt).to(device)).float()
        pMs.append(Prompt(embed, weight, stop).to(device))

    for prompt in args.image_prompts:
        path, weight, stop = split_prompt(prompt)
        img = Image.open(path)
        pil_image = img.convert('RGB')
        img = resize_image(pil_image, (sideX, sideY))
        batch = make_cutouts(TF.to_tensor(img).unsqueeze(0).to(device))
        embed = perceptor.encode_image(normalize(batch)).float()
        pMs.append(Prompt(embed, weight, stop).to(device))

    for seed, weight in zip(args.noise_prompt_seeds, args.noise_prompt_weights):
        gen = torch.Generator().manual_seed(seed)
        embed = torch.empty([1, perceptor.visual.output_dim]).normal_(generator=gen)
        pMs.append(Prompt(embed, weight).to(device))


    opt = get_opt(args.optimiser, args.step_size, z)

    # Output for the user
    print('Using device:', device)
    print('Optimising using:', args.optimiser)

    if args.prompts:
        print('Using text prompts:', args.prompts)  
    if args.image_prompts:
        print('Using image prompts:', args.image_prompts)
    if args.init_image:
        print('Using initial image:', args.init_image)
    if args.noise_prompt_weights:
        print('Noise prompt weights:', args.noise_prompt_weights)    


    if args.seed is None:
        seed = torch.seed()
    else:
        seed = args.seed  
    torch.manual_seed(seed)
    print('Using seed:', seed)

    i = 0 # Iteration counter
    j = 0 # Zoom video frame counter
    p = 1 # Phrase counter
    smoother = 0 # Smoother counter

    # Do it
    try:
        with tqdm() as pbar:
            while True:            
                # Change generated image
                if args.make_zoom_video:
                    if i % args.zoom_frequency == 0:
                        out = synth(z)
                        
                        # Save image
                        img = np.array(out.mul(255).clamp(0, 255)[0].cpu().detach().numpy().astype(np.uint8))[:,:,:]
                        img = np.transpose(img, (1, 2, 0))
                        imageio.imwrite('./steps/' + str(j) + '.png', np.array(img))

                        # Time to start zooming?                    
                        if args.zoom_start <= i:
                            # Convert z back into a Pil image                    
                            pil_image = TF.to_pil_image(out[0].cpu())
                            
                            # Zoom
                            pil_image_zoom = zoom_at(pil_image, args.size[0]/2, args.size[1]/2, args.zoom_scale)
                            
                            # Convert image back to a tensor again
                            pil_tensor = TF.to_tensor(pil_image_zoom)
                            
                            # Re-encode
                            z, *_ = model.encode(pil_tensor.to(device).unsqueeze(0) * 2 - 1)
                            z_orig = z.clone()
                            z.requires_grad_(True)

                            # Reset optimiser
                            opt = get_opt(args.optimiser, args.step_size)
                        
                        # Next
                        j += 1
                
                # Change text prompt
                if args.prompt_frequency > 0:
                    if i % args.prompt_frequency == 0 and i > 0:
                        # In case there aren't enough phrases, just loop
                        if p >= len(all_phrases):
                            p = 0
                        
                        pMs = []
                        args.prompts = all_phrases[p]

                        # Show user we're changing prompt                                
                        print(args.prompts)
                        
                        for prompt in args.prompts:
                            txt, weight, stop = split_prompt(prompt)
                            embed = perceptor.encode_text(clip.tokenize(txt).to(device)).float()
                            pMs.append(Prompt(embed, weight, stop).to(device))
                        
                        p += 1    

                # Training time
                train(i, opt, z, perceptor, normalize, make_cutouts, args, pMs, z_min, z_max)
                
                # Ready to stop yet?
                if i == args.max_iterations:
                    break

                i += 1
                pbar.update()
    except KeyboardInterrupt:
        pass


    # Video generation
    if args.make_video or args.make_zoom_video:
        init_frame = 1      # Initial video frame
        if args.make_zoom_video:
            last_frame = j
        else:
            last_frame = i  # This will raise an error if that number of frames does not exist.

        length = args.video_length # Desired time of the video in seconds

        min_fps = 10
        max_fps = 60

        total_frames = last_frame-init_frame

        frames = []
        tqdm.write('Generating video...')
        for i in range(init_frame,last_frame):
            temp = Image.open("./steps/"+ str(i) +'.png')
            keep = temp.copy()
            frames.append(keep)
            temp.close()
        
        if args.output_video_fps > 9:
            # Hardware encoding and video frame interpolation
            print("Creating interpolated frames...")
            ffmpeg_filter = f"minterpolate='mi_mode=mci:me=hexbs:me_mode=bidir:mc_mode=aobmc:vsbmc=1:mb_size=8:search_param=32:fps={args.output_video_fps}'"
            output_file = re.compile('\.png$').sub('.mp4', args.output)
            p = Popen(['ffmpeg',
                    '-y',
                    '-f', 'image2pipe',
                    '-vcodec', 'png',
                    '-r', str(args.input_video_fps),               
                    '-i',
                    '-',
                    '-b:v', '10M',
                    '-vcodec', 'h264_nvenc',
                    '-pix_fmt', 'yuv420p',
                    '-strict', '-2',
                    '-filter:v', f'{ffmpeg_filter}',
                    '-metadata', f'comment={args.prompts}',
                    output_file], stdin=PIPE)               
            for im in tqdm(frames):
                im.save(p.stdin, 'PNG')
            p.stdin.close()
            p.wait()
        else:
            # CPU
            fps = np.clip(total_frames/length,min_fps,max_fps)
            output_file = re.compile('\.png$').sub('.mp4', args.output)
            p = Popen(['ffmpeg',
                    '-y',
                    '-f', 'image2pipe',
                    '-vcodec', 'png',
                    '-r', str(fps),
                    '-i',
                    '-',
                    '-vcodec', 'libx264',
                    '-r', str(fps),
                    '-pix_fmt', 'yuv420p',
                    '-crf', '17',
                    '-preset', 'veryslow',
                    '-metadata', f'comment={args.prompts}',
                    output_file], stdin=PIPE)               
            for im in tqdm(frames):
                im.save(p.stdin, 'PNG')
            p.stdin.close()
            p.wait() 


if __name__ == '__main__':
    main()
