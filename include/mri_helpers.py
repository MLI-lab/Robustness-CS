import torch 
import torch.nn as nn
import torchvision
import sys

import numpy as np
from PIL import Image
import PIL
import numpy as np

from torch.autograd import Variable

import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from PIL import Image
import PIL

from torch.autograd import Variable
dtype = torch.cuda.FloatTensor

from . import transforms as transform
from .helpers import var_to_np,np_to_var

import numpy
import scipy.signal
import scipy.ndimage


def ksp2measurement(ksp):
    return np_to_var( np.transpose( np.array([np.real(ksp),np.imag(ksp)]) , (1, 2, 3, 0)) )   

def lsreconstruction(measurement,mode='both'):
    # measurement has dimension (1, num_slices, x, y, 2)
    fimg = transform.ifft2(measurement)
    normimag = torch.norm(fimg[:,:,:,:,0])
    normreal = torch.norm(fimg[:,:,:,:,1])
    #print("real/img parts: ",normimag, normreal)
    if mode == 'both':
        return torch.sqrt(fimg[:,:,:,:,0]**2 + fimg[:,:,:,:,1]**2)
    elif mode == 'real':
        return torch.tensor(fimg[:,:,:,:,0]) #torch.sqrt(fimg[:,:,:,:,0]**2)
    elif mode == 'imag':
        return torch.sqrt(fimg[:,:,:,:,1]**2)

def root_sum_of_squares2(lsimg):
    out = np.zeros(lsimg[0].shape)
    for img in lsimg:
        out += img**2
    return np.sqrt(out)

def crop_center2(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

def channels2imgs(out):
    sh = out.shape
    chs = int(sh[0]/2)
    imgs = np.zeros( (chs,sh[1],sh[2]) )
    for i in range(chs):
        imgs[i] = np.sqrt( out[2*i]**2 + out[2*i+1]**2 )
    return imgs

def forwardm(img,mask):
    # img has dimension (2*num_slices, x,y)
    # output has dimension (1, num_slices, x, y, 2)
    mask = np_to_var(mask)[0].type(dtype)
    s = img.shape
    ns = int(s[1]/2) # number of slices
    fimg = Variable( torch.zeros( (s[0],ns,s[2],s[3],2 ) ) ).type(dtype)
    for i in range(ns):
        fimg[0,i,:,:,0] = img[0,2*i,:,:]
        fimg[0,i,:,:,1] = img[0,2*i+1,:,:]
    Fimg = transform.fft2(fimg) # dim: (1,num_slices,x,y,2)
    for i in range(ns):
        Fimg[0,i,:,:,0] *= mask
        Fimg[0,i,:,:,1] *= mask
    return Fimg

def get_scale_factor(net,num_channels,in_size,slice_ksp,scale_out=1,scale_type="norm"): 
    ### get norm of deep decoder output
    # get net input, scaling of that is irrelevant
    shape = [1,num_channels, in_size[0], in_size[1]]
    ni = Variable(torch.zeros(shape)).type(dtype)
    ni.data.uniform_()
    # generate random image
    try:
        out_chs = net( ni.type(dtype),scale_out=scale_out ).data.cpu().numpy()[0]
    except:
        out_chs = net( ni.type(dtype) ).data.cpu().numpy()[0]
    out_imgs = channels2imgs(out_chs)
    out_img_tt = transform.root_sum_of_squares( torch.tensor(out_imgs) , dim=0)

    ### get norm of least-squares reconstruction
    ksp_tt = transform.to_tensor(slice_ksp)
    orig_tt = transform.ifft2(ksp_tt)           # Apply Inverse Fourier Transform to get the complex image
    orig_imgs_tt = transform.complex_abs(orig_tt)   # Compute absolute value to get a real image
    orig_img_tt = transform.root_sum_of_squares(orig_imgs_tt, dim=0)
    orig_img_np = orig_img_tt.cpu().numpy()
    
    if scale_type == "norm":
        s = np.linalg.norm(out_img_tt) / np.linalg.norm(orig_img_np)
    if scale_type == "mean":
        s = (out_img_tt.mean() / orig_img_np.mean()).numpy()[np.newaxis][0]
    return s,ni

def data_consistency(parnet, parni, mask1d, slice_ksp_torchtensor1):    
    img = parnet(parni.type(dtype))
    s = img.shape
    ns = int(s[1]/2) # number of slices
    fimg = Variable( torch.zeros( (s[0],ns,s[2],s[3],2 ) ) ).type(dtype)
    for i in range(ns):
        fimg[0,i,:,:,0] = img[0,2*i,:,:]
        fimg[0,i,:,:,1] = img[0,2*i+1,:,:]
    Fimg = transform.fft2(fimg) # dim: (1,num_slices,x,y,2)
    # ksp has dim: (num_slices,x,y)
    meas = slice_ksp_torchtensor1.unsqueeze(0) # dim: (1,num_slices,x,y,2)
    mask = torch.from_numpy(np.array(mask1d, dtype=np.uint8))
    ksp_dc = Fimg.clone()
    ksp_dc = ksp_dc.detach().cpu()
    ksp_dc[:,:,:,mask==1,:] = meas[:,:,:,mask==1,:] # after data consistency block

    img_dc = transform.ifft2(ksp_dc)[0]
    out = []
    for img in img_dc.detach().cpu():
        out += [ img[:,:,0].numpy() , img[:,:,1].numpy() ]

    par_out_chs = np.array(out)
    par_out_imgs = channels2imgs(par_out_chs)

    # deep decoder reconstruction
    prec = root_sum_of_squares2(par_out_imgs)
    if prec.shape[0] > 320:
        prec = crop_center2(prec,320,320)
    return prec
def data_consistency_sense(net,ni,mask,mask1d,ksp,sens_maps,post_process=True):
    if post_process:
        out = net(ni.type(dtype))
        
        S = transform.to_tensor(sens_maps).type(dtype)
        imgs = torch.zeros(S.shape).type(dtype)
        for j,s in enumerate(S):
            imgs[j,:,:,0] = out[0,0,:,:] * s[:,:,0] - out[0,1,:,:] * s[:,:,1]
            imgs[j,:,:,1] = out[0,0,:,:] * s[:,:,1] + out[0,1,:,:] * s[:,:,0]
        Fimg = transform.fft2(imgs[None,:])
        #Fimg,_ = transform.apply_mask(Fimg, mask = mask.type(dtype))
        
        # ksp has dim: (num_slices,x,y)
        meas = ksp2measurement(ksp) # dim: (1,num_slices,x,y,2)
        maskk = torch.from_numpy(np.array(mask1d, dtype=np.uint8))
        ksp_dc = Fimg.clone()
        ksp_dc = ksp_dc.detach().cpu()
        ksp_dc[:,:,:,maskk==1,:] = meas[:,:,:,maskk==1,:] # after data consistency block
        
        img_dc = transform.ifft2(ksp_dc)[0]
        out = []
        for img in img_dc.detach().cpu():
            out += [ img[:,:,0].numpy() , img[:,:,1].numpy() ]
        
        out_chs = np.array(out)
    else:
        out_chs = net( ni.type(dtype) ).data.cpu().numpy()[0]
    out_imgs = channels2imgs(out_chs)
    
    # deep decoder reconstruction
    rec = crop_center2(root_sum_of_squares2(out_imgs),320,320)
    return rec