from torch.autograd import Variable
import torch
import torch.optim
import copy
import numpy as np
from scipy.linalg import hadamard
from skimage.metrics import structural_similarity as ssim

from .helpers import *
from .mri_helpers import *
from .transforms import *

dtype = torch.FloatTensor
#dtype = torch.FloatTensor
           

def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=500):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.65**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

def sqnorm(a):
    return np.sum( a*a )

def get_distances(initial_maps,final_maps):
    results = []
    for a,b in zip(initial_maps,final_maps):
        res = sqnorm(a-b)/(sqnorm(a) + sqnorm(b))
        results += [res]
    return(results)

def get_weights(net):
    weights = []
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            weights += [m.weight.data.cpu().numpy()]
    return weights

class rMSELoss(torch.nn.Module):
    def __init__(self):
        super(rMSELoss,self).__init__()

    def forward(self,net_input,x,y,lam1,lam2):
        criterion = nn.MSELoss()
        loss = -lam1*criterion(x,y) + lam2*torch.norm(net_input) # 0.1 is the regularizer parameter
        return loss

def runner(net,
        ksp,
        gt,
        num_iter = 5000,
        LR = 0.01,
        lam1 = 0.1,
        lam2 = 0.1,
        OPTIMIZER='adam',
        mask = None,
        devices = [torch.device("cuda:3")],
        lr_decay_epoch = 0,
        weight_decay=0,
        loss_type='MSE',
        model_type='unet',
        retain_graph = False,
        find_best = True,
       ):

    shape = ksp.shape
    print("input shape: ", shape)
    net_input = Variable(torch.zeros(shape)).type(dtype).to(devices[0])
    #net_input.data.uniform_()
    net_input.data.normal_()
    #net_input.data = torch.nn.init.kaiming_uniform_(net_input.data)
    net_input.data *= torch.norm(ksp)/torch.norm(net_input)/100#1./1e3

    net_input = net_input.type(dtype).to(devices[0])
    net_input_saved = net_input.data.clone()
    noise = net_input.data.clone()
    
    net_input.requires_grad = True
    p = [net_input]

    # set model grads to false
    for param in net.parameters():
        param.requires_grad = False
        
    mse_ = np.zeros(num_iter)
    
    
    if OPTIMIZER == 'SGD':
        print("optimize with SGD", LR)
        optimizer = torch.optim.SGD(p, lr=LR,momentum=0.9,weight_decay=weight_decay)
    elif OPTIMIZER == 'adam':
        print("optimize with adam", LR)
        optimizer = torch.optim.Adam(p, lr=LR,weight_decay=weight_decay)
    elif OPTIMIZER == 'LBFGS':
        print("optimize with LBFGS", LR)
        optimizer = torch.optim.LBFGS(p, lr=LR)
    elif OPTIMIZER == "adagrad":
        print("optimize with adagrad", LR)
        optimizer = torch.optim.Adagrad(p, lr=LR,weight_decay=weight_decay)

    mse = rMSELoss()
    
    masked_kspace, _ = transform.apply_mask(ksp, mask = mask.type(dtype).to(devices[0]))
    if model_type == 'unet':
        ### fixed reconstruction from non-perturbed data
        sampled_image2 = transform.ifft2(masked_kspace)
        crop_size = (320, 320)
        sampled_image = transform.complex_center_crop(sampled_image2, crop_size)
        # Absolute value
        sampled_image = transform.complex_abs(sampled_image)
        # Apply Root-Sum-of-Squares if multicoil data
        sampled_image = transform.root_sum_of_squares(sampled_image)
        # Normalize input
        sampled_image, mean, std = transform.normalize_instance(sampled_image, eps=1e-11)
        sampled_image = sampled_image.clamp(-6, 6)
        inp2 = sampled_image.unsqueeze(0)
        out2 = net(inp2.type(dtype).to(devices[0]))
    elif model_type == 'varnet':
        with torch.no_grad():
            out2 = net(masked_kspace[None,:].type(dtype).to(devices[0]),mask.type(torch.cuda.ByteTensor).to(devices[0])).to(devices[-1])
        torch.cuda.empty_cache()
    pert_recs = []
    R = []
    for i in range(num_iter):
        inp = net_input + ksp
        masked_kspace, _ = transform.apply_mask(inp, mask = mask.type(dtype).to(devices[0]))
        
        if model_type == 'unet':
            sampled_image2 = transform.ifft2(masked_kspace)
            crop_size = (320, 320)
            sampled_image = transform.complex_center_crop(sampled_image2, crop_size)
            # Absolute value
            sampled_image = transform.complex_abs(sampled_image)
            # Apply Root-Sum-of-Squares if multicoil data
            sampled_image = transform.root_sum_of_squares(sampled_image)
            # Normalize input
            sampled_image, mean, std = transform.normalize_instance(sampled_image, eps=1e-11)
            sampled_image = sampled_image.clamp(-6, 6)

            inp = sampled_image.unsqueeze(0)
            out = net(inp.type(dtype).to(devices[0]))
            pert_recs.append(out.data.cpu().numpy()[0])
        elif model_type == 'varnet':
            #with torch.no_grad():
            out = net(masked_kspace[None,:].type(dtype).to(devices[0]),mask.type(torch.cuda.ByteTensor).to(devices[0]))
            pert_recs.append( crop_center2(out.data.cpu().numpy()[0],320,320) )
        
        def closure():
            
            optimizer.zero_grad()
            
            loss = mse(net_input.to(devices[-1]), out, out2,lam1,lam2)
            
            loss.backward(retain_graph=retain_graph)
            
            mse_[i] = loss.data.cpu().numpy()
            
            if i % 10 == 0:
                print ('Iteration %05d   loss %f' % (i, mse_[i]), '\r', end='')
            
            return loss   
        R.append(net_input.data.cpu())
        loss = optimizer.step(closure)
        
        ### discard buffers
        #del(out)
        #torch.cuda.empty_cache()
        ###
        
    return R,mse_,crop_center2(out2.data.cpu().numpy()[0],320,320),pert_recs