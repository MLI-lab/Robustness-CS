from torch.autograd import Variable
import torch
import torch.optim
import time
import copy
import pickle
import random
import numpy as np
from scipy.linalg import hadamard
from skimage.metrics import structural_similarity as ssim

from .helpers import *
from .mri_helpers import *
from .transforms import *
from .decoder_conv import *

dtype = torch.cuda.FloatTensor
#dtype = torch.FloatTensor
           

def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=500):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.65**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

class MSLELoss(torch.nn.Module):
    def __init__(self):
        super(MSLELoss,self).__init__()

    def forward(self,x,y):
        criterion = nn.MSELoss()
        loss = torch.log(criterion(x, y))
        return loss

class rMSELoss(torch.nn.Module):
    def __init__(self):
        super(rMSELoss,self).__init__()

    def forward(self,r,x,y,lam):
        criterion = nn.MSELoss()
        loss = -criterion(x,y) + lam*torch.norm(r) # 0.01 is the regularizer parameter
        return loss

class MyLoss(torch.nn.Module):
    #def __init__(self):
    #    super(MyLoss,self).__init__()
    def forward(self,r,y,yr,H,lam):
        loss = -(torch.norm(y-yr)**2) /np.prod(H.data.cpu().numpy().shape) + lam * (torch.norm(r)**2)
        return loss
    def get_derivs(self,y,yr,yd,r,H,lam):
        # y   : reconstruction from clean k-space
        # yr  : reconstruction from perturbed k-space
        # yd  : reconstruction from slightly perturbed version of the "perturbed" kspace (for numerical derivation)
        # r   : perturbation
        # H   : slight perturbation for computing numerical derivatives
        
        #grad1 = 1/np.prod(y.data.cpu().numpy().shape)
        grad1 = 1/np.prod(H.data.cpu().numpy().shape) 
        grad1 *= (torch.norm(y-yd)**2 - torch.norm(y-yr)**2) / (2*H)
        
        grad2 = lam*r
        
        print("\ngrad norms:",torch.norm(grad1),torch.norm(grad2))
        grad = -grad1 + grad2
        
        #self.grad = grad
        #del(y,yr,yd,r,H,grad1,grad2,grad)
        #torch.cuda.empty_cache()
        return grad
        
def get_scale_factor(net,num_channels,in_size,ksp_tt,ni=None,scale_out=1,scale_type="norm"): 
    ### get norm of deep decoder output
    # get net input, scaling of that is irrelevant
    if ni is None:
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

def myrunner_untrained( ksp,
                        num_iter = 20,
                        num_iter_inner = 10000,
                        LR = 0.01,
                        lam = 0.1,
                        eps = 1e2,
                        OPTIMIZER='adam',
                        mask = None,
                        mask1d = None,
                        mask2d = None,
                        lr_decay_epoch = 0,
                        weight_decay=0,
                        loss_type="MSE",
                        retain_graph = False,
                        find_best = True,
                      ):
    ################ main optimization steup: perturbation finder ################
    shape = ksp.shape
    print("perturbation shape: ", shape)
    r = Variable(torch.zeros(shape).cuda(),requires_grad=True).type(dtype)
    r.data.uniform_()
    #r.data *= 1/torch.norm(ksp)#1./1e3
    r.data *= torch.norm(ksp)/torch.norm(r)
    
    r = r.type(dtype)
    r.retain_grad()
    r_saved = r.data.clone()
    
    ####
    r.requires_grad = True
    optimizer = torch.optim.SGD([r], lr=LR,momentum=0.9,weight_decay=weight_decay)
    ####
    
    loss_ = np.zeros(num_iter)
    #loss = MyLoss()
    loss = rMSELoss()
    ################ ################
    
    ################ sub optimization: fitting ConvDecoder (or any untrained network) ################
    num_channels = 160 #256
    num_layers = 8
    strides = [1]*(num_layers-1)
    in_size = [8,4]
    kernel_size = 3
    output_depth = ksp.numpy().shape[0]*2
    out_size = ksp.numpy().shape[1:-1]
    width,height = in_size
    shape = [1,num_channels, width, height]
    print("network input shape: ", shape)
    net_input = Variable(torch.zeros(shape)).type(dtype)
    net_input.data.uniform_()
    net_input.data *= 1./10
    net_input = net_input.type(dtype)
    
    ##### fit the network for reconstruction without perturbation #####
    net = convdecoder(out_size,in_size,output_depth,
                         num_layers,strides,num_channels, act_fun = nn.ReLU(),
                         skips=False,need_sigmoid=False,bias=False, need_last = True,
                         kernel_size=kernel_size,upsample_mode="nearest").type(dtype)
    torch.save(net.state_dict(), "./init")
    ### fix scaling for ConvDecoder
    scaling_factor,_ = get_scale_factor(net,
                                       num_channels,
                                       in_size,
                                       ksp,
                                       ni=net_input)
    slice_ksp_torchtensor1 = ksp * scaling_factor
    masked_kspace, mask = transform.apply_mask(slice_ksp_torchtensor1, mask = mask)
    unders_measurement = np_to_var( masked_kspace.data.cpu().numpy() ).type(dtype)
    sampled_image2 = transform.ifft2(masked_kspace)
    measurement = slice_ksp_torchtensor1.unsqueeze(0).type(dtype) 
    lsimg = lsreconstruction(measurement)
    out = []
    for img in sampled_image2:
        out += [ img[:,:,0].numpy() , img[:,:,1].numpy() ]
    lsest = torch.tensor(np.array([out]))

    scale_out,sover,pover,par_mse_n, par_mse_t, parni, fixed_net = fitr( net,
                                                                        unders_measurement,
                                                                        Variable(lsest).type(dtype),
                                                                        mask2d,
                                                                        num_iter=num_iter_inner,
                                                                        LR=0.008,
                                                                        apply_f = forwardm,
                                                                        lsimg = lsimg,
                                                                        find_best=True,
                                                                        net_input = net_input,
                                                                        OPTIMIZER = "adam"
                                                                       )
    #out_chs = fixed_net( net_input.type(dtype) ).data.cpu().numpy()[0]
    #out_imgs = channels2imgs(out_chs)
    #fixed_rec = torch.from_numpy(crop_center2(root_sum_of_squares2(out_imgs),320,320)).type(dtype)
    fixed_rec = torch.from_numpy( data_consistency(fixed_net, net_input, mask1d, slice_ksp_torchtensor1) ).type(dtype)
    fixed_rec, mean, std = transform.normalize_instance(fixed_rec, eps=1e-11)
    fixed_rec = fixed_rec.clamp(-6, 6)
    
    pert_recs = []
    R = []
    
    for i in range(num_iter):
        ### prepare input for ConvDecoder
        # create the network
        net = convdecoder(out_size,in_size,output_depth,
                         num_layers,strides,num_channels, act_fun = nn.ReLU(),
                         skips=False,need_sigmoid=False,bias=False, need_last = True,
                         kernel_size=kernel_size,upsample_mode="nearest").type(dtype)
        net.load_state_dict(torch.load("./init"))
        # f(A(x+r)) recovery
        print("\n***fit ConvDecoder at i = {}***".format(i))
        print("norms:",torch.norm(r),torch.norm(ksp))
        inp = r + ksp.type(dtype)
        scaling_factor,_ = get_scale_factor(net,
                                           num_channels,
                                           in_size,
                                           inp.data,
                                           ni=net_input)
        slice_ksp_torchtensor1 = inp * scaling_factor
        masked_kspace, mask = transform.apply_mask(slice_ksp_torchtensor1, mask = mask.type(dtype))
        unders_measurement = np_to_var( masked_kspace.data.cpu().numpy() ).type(dtype)
        sampled_image2 = transform.ifft2(masked_kspace)
        measurement = slice_ksp_torchtensor1.unsqueeze(0).type(dtype) 
        lsimg = lsreconstruction(measurement)
        # fit the network
        out = []
        for img in sampled_image2:
            out += [ img[:,:,0].numpy() , img[:,:,1].numpy() ]
        lsest = torch.tensor(np.array([out]))

        scale_out,sover,pover,par_mse_n, par_mse_t, parni, pert_net = fitr(net,
                                                                        unders_measurement,
                                                                        Variable(lsest).type(dtype),
                                                                        mask2d,
                                                                        num_iter=num_iter_inner,
                                                                        LR=0.008,
                                                                        apply_f = forwardm,
                                                                        lsimg = lsimg,
                                                                        find_best=True,
                                                                        net_input = net_input,
                                                                        OPTIMIZER = "adam"
                                                                        )
        #out_chs = pert_net( net_input.type(dtype) ).data.cpu().numpy()[0]
        #out_imgs = channels2imgs(out_chs)
        #pert_rec = torch.from_numpy(crop_center2(root_sum_of_squares2(out_imgs),320,320)).type(dtype)
        pert_rec = torch.from_numpy( data_consistency(pert_net, net_input, mask1d, slice_ksp_torchtensor1) ).type(dtype)
        pert_rec, mean, std = transform.normalize_instance(pert_rec, eps=1e-11)
        pert_rec = pert_rec.clamp(-6, 6)
        pert_recs.append(pert_rec.data.cpu().numpy())
        
        def closure():
            
            optimizer.zero_grad()
            
            loss__ = loss(r,fixed_rec.type(dtype),pert_rec.type(dtype),lam)
            
            loss__.backward(retain_graph=retain_graph)
            
            loss_[i] = loss__.data.cpu().numpy()
            
            if i % 1 == 0:
                print ('Iteration %05d   loss %f' % (i, loss_[i]), '\r', end='')
            
            return loss__   
        #print("\n{}\n".format(r.requires_grad))
        R.append(r.data.cpu())
        loss__ = optimizer.step(closure)
        ### new network for computing derivatives
        """print("\n***fit ConvDecoder at i = {} for derivatives***".format(i))
        net = convdecoder(out_size,in_size,output_depth,
                         num_layers,strides,num_channels, act_fun = nn.ReLU(),
                         skips=False,need_sigmoid=False,bias=False, need_last = True,
                         kernel_size=kernel_size,upsample_mode="nearest").type(dtype)
        net.load_state_dict(torch.load("./init"))
        ### fit this network to get an approximate derivative of f(A(x+r)) with respect to A(x+r) --> slightly perturb it with epsilon
        shape = inp.shape
        #H = Variable(torch.zeros(shape)).type(dtype)
        #H.data.uniform_(0.1,0.2)
        H /= torch.norm(H)
        H *= torch.norm(inp) / eps
        print(H.shape,unders_measurement.shape)
        unders_meas = unders_measurement.clone()
        unders_meas += H
        scale_out,sover,pover,par_mse_n, par_mse_t, parni, der_net = fitr(net,
                                                                        unders_meas.type(dtype),
                                                                        Variable(lsest).type(dtype),
                                                                        mask2d,
                                                                        num_iter=num_iter_inner,
                                                                        LR=0.008,
                                                                        apply_f = forwardm,
                                                                        lsimg = lsimg,
                                                                        find_best=True,
                                                                        net_input = net_input,
                                                                        OPTIMIZER = "adam",
                                                                        retain_graph=True,
                                                                        )
        #out_chs = pert_net( net_input.type(dtype) ).data.cpu().numpy()[0]
        #out_imgs = channels2imgs(out_chs)
        #pert_rec = torch.from_numpy(crop_center2(root_sum_of_squares2(out_imgs),320,320)).type(dtype)
        der_rec = torch.from_numpy( data_consistency(der_net, net_input, mask1d, slice_ksp_torchtensor1) ).type(dtype)
        der_rec, mean, std = transform.normalize_instance(der_rec, eps=1e-11)
        der_rec = der_rec.clamp(-6, 6)
        ### compute loss and perform the optimization step
        loss_[i] = loss(r,fixed_rec,pert_rec,H,lam)
        grad = loss.get_derivs(fixed_rec,pert_rec,der_rec,r,H,lam)
        r -= LR*grad
        print("\nloss at iteration{}:".format(i),loss_[i])
        
        R.append(r.data.cpu())
        print(2*"\n")"""
        #loss = optimizer.step(closure)
        with open("./outputs/untrainedrunner_test/results","wb") as fn:
            pickle.dump([R,ksp,loss_,fixed_rec.data.cpu().numpy(),pert_recs],fn)
    return R,net_input, loss_, fixed_rec.data.cpu().numpy(), pert_recs

def myrunner_untrained_test3( ksp,
                        num_iter = 20,
                        num_iter_inner = 10000,
                        LR = 0.01,
                        lam = 0.1,
                        eps = 1e2,
                        OPTIMIZER='adam',
                        mask = None,
                        mask1d = None,
                        mask2d = None,
                        lr_decay_epoch = 0,
                        weight_decay=0,
                        loss_type="MSE",
                        retain_graph = False,
                        find_best = True,
                      ):
    ################ main optimization steup: perturbation finder ################
    shape = ksp.shape
    print("perturbation shape: ", shape)
    r = Variable(torch.zeros(shape)).type(dtype)
    r.data.uniform_()
    #r.data *= 1/torch.norm(ksp)#1./1e3
    r.data *= torch.norm(ksp)/torch.norm(r)
    
    r = r.type(dtype)
    #r.retain_grad()
    r_saved = r.data.clone()
    
    ####
    r.requires_grad = True
    optimizer = torch.optim.SGD([r], lr=LR,momentum=0.9,weight_decay=weight_decay)
    ####
    
    loss_ = np.zeros(num_iter)
    #loss = MyLoss()
    loss = rMSELoss()
    ################ ################
    
    ################ sub optimization: fitting ConvDecoder (or any untrained network) ################
    num_channels = 160 #256
    num_layers = 8
    strides = [1]*(num_layers-1)
    in_size = [8,4]
    kernel_size = 3
    output_depth = ksp.numpy().shape[0]*2
    out_size = ksp.numpy().shape[1:-1]
    width,height = in_size
    shape = [1,num_channels, width, height]
    print("network input shape: ", shape)
    net_input = Variable(torch.zeros(shape)).type(dtype)
    net_input.data.uniform_()
    net_input.data *= 1./10
    net_input = net_input.type(dtype)
    
    ##### fit the network for reconstruction without perturbation #####
    net = convdecoder(out_size,in_size,output_depth,
                         num_layers,strides,num_channels, act_fun = nn.ReLU(),
                         skips=False,need_sigmoid=False,bias=False, need_last = True,
                         kernel_size=kernel_size,upsample_mode="nearest").type(dtype)
    torch.save(net.state_dict(), "./init")
    ### fix scaling for ConvDecoder
    scaling_factor,_ = get_scale_factor(net,
                                       num_channels,
                                       in_size,
                                       ksp,
                                       ni=net_input)
    slice_ksp_torchtensor1 = ksp.type(dtype) * scaling_factor
    masked_kspace, mask = transform.apply_mask(slice_ksp_torchtensor1, mask = mask.type(dtype))
    #unders_measurement = np_to_var( masked_kspace.data.cpu().numpy() ).type(dtype)
    unders_measurement = Variable(masked_kspace[None,:])
    sampled_image2 = transform.ifft2(masked_kspace)
    measurement = slice_ksp_torchtensor1.unsqueeze(0).type(dtype) 
    lsimg = lsreconstruction(measurement)
    out = []
    for img in sampled_image2:
        out += [ img[:,:,0].data.cpu().numpy() , img[:,:,1].data.cpu().numpy() ]
    lsest = torch.tensor(np.array([out]))
    
    ######################### optimization for convdecoder
    img_noisy_var = unders_measurement.clone()
    img_clean_var = Variable(lsest).type(dtype)
    p = [x for x in net.parameters()]
    optimizer2 = torch.optim.Adam(p, lr=0.1,weight_decay=weight_decay)
    mse = torch.nn.MSELoss()
    mse_wrt_noisy = np.zeros(num_iter_inner)
    import copy
    if find_best:
        best_net = copy.deepcopy(net)
        best_mse = 1000000.0
    SSIMs = []
    PSNRs = []
    for i in range(num_iter_inner):

        def closure2():
            
            optimizer2.zero_grad()
            out = net(net_input.type(dtype))
                
            # training loss
            losss = mse( forwardm(out,mask2d) , img_noisy_var )
        
            losss.backward(retain_graph=retain_graph)
            
            mse_wrt_noisy[i] = losss.data.cpu().numpy()

            # the actual loss 
            true_loss = mse( Variable(out.data, requires_grad=False).type(dtype), img_clean_var.type(dtype) )
            
            if i % 100 == 0:
                if lsimg is not None:
                    ### compute ssim and psnr ###
                    out_chs = out.data.cpu().numpy()[0]
                    out_imgs = channels2imgs(out_chs)
                    # least squares reconstruciton
                    orig = crop_center2( root_sum_of_squares2(var_to_np(lsimg)) , 320,320)

                    # deep decoder reconstruction
                    rec = crop_center2(root_sum_of_squares2(out_imgs),320,320)

                    ssim_const = ssim(orig,rec,data_range=orig.max())
                    SSIMs.append(ssim_const)

                    psnr_const = psnr(orig,rec,np.max(orig))
                    PSNRs.append(psnr_const)
                    
                    ### ###
                
                trloss = losss.data
                true_loss = true_loss.data
                print ('Iteration %05d    Train loss %f  Actual loss %f' % (i, trloss,true_loss), '\r', end='')
            
            return losss   
        
        losss = optimizer2.step(closure2)
            
        # if training loss improves by at least one percent, we found a new best net
        lossval = losss.data
        if best_mse > 1.005*lossval:
            best_mse = lossval
            best_net = copy.deepcopy(net)
    net = best_net
    #scale_out,sover,pover,par_mse_n, par_mse_t, parni, fixed_net = fitr( net,
    #                                                                    unders_measurement,
    #                                                                    Variable(lsest).type(dtype),
    #                                                                    mask2d,
    #                                                                    num_iter=num_iter_inner,
    #                                                                    LR=0.008,
    #                                                                    apply_f = forwardm,
    #                                                                    lsimg = lsimg,
    #                                                                    find_best=True,
    #                                                                    net_input = net_input,
    #                                                                    OPTIMIZER = "adam"
    #                                                                   )
    out_chs = net( net_input.type(dtype) )[0]
    sh = out_chs.shape
    chs = int(sh[0]/2)
    imgs = torch.zeros( (chs,sh[1],sh[2]) ).type(dtype)
    for q in range(chs):
        imgs[q] = torch.sqrt( out_chs[2*q]**2 + out_chs[2*q+1]**2 )
    fixed_rec = root_sum_of_squares(imgs)
    #out_chs = fixed_net( net_input.type(dtype) ).data.cpu().numpy()[0]
    #out_imgs = channels2imgs(out_chs)
    #fixed_rec = torch.from_numpy(crop_center2(root_sum_of_squares2(out_imgs),320,320)).type(dtype)
    #fixed_rec = torch.from_numpy( data_consistency(net, net_input, mask1d, slice_ksp_torchtensor1) ).type(dtype)
    fixed_rec, mean, std = transform.normalize_instance(fixed_rec, eps=1e-11)
    fixed_rec = fixed_rec.clamp(-6, 6)
    
    pert_recs = []
    R = []
    
    for j in range(num_iter):
        ### prepare input for ConvDecoder
        # create the network
        net3 = convdecoder(out_size,in_size,output_depth,
                         num_layers,strides,num_channels, act_fun = nn.ReLU(),
                         skips=False,need_sigmoid=False,bias=False, need_last = True,
                         kernel_size=kernel_size,upsample_mode="nearest").type(dtype)
        net3.load_state_dict(torch.load("./init"))
        # f(A(x+r)) recovery
        print("\n***fit ConvDecoder at i = {}***".format(j))
        print("norms:",torch.norm(r),torch.norm(ksp))
        inp = r + ksp.type(dtype)
        #scaling_factor,_ = get_scale_factor(net3,
        #                                   num_channels,
        #                                   in_size,
        #                                   inp.data,
        #                                   ni=net_input)
        slice_ksp_torchtensor1 = inp * scaling_factor
        masked_kspace, mask = transform.apply_mask(slice_ksp_torchtensor1, mask = mask.type(dtype))
        #unders_measurement = np_to_var( masked_kspace.data.cpu().numpy() ).type(dtype)
        unders_measurement = masked_kspace[None,:]
        sampled_image2 = transform.ifft2(masked_kspace)
        measurement = slice_ksp_torchtensor1.unsqueeze(0).type(dtype) 
        lsimg = lsreconstruction(measurement)
        # fit the network
        out = []
        for img in sampled_image2:
            out += [ img[:,:,0].data.cpu().numpy() , img[:,:,1].data.cpu().numpy() ]
        lsest = torch.tensor(np.array([out]))

        ######################### optimization for convdecoder
        img_noisy_var = unders_measurement
        img_clean_var = Variable(lsest).type(dtype)
        p3 = [x for x in net3.parameters()]
        optimizer3 = torch.optim.Adam(p3, lr=0.1,weight_decay=weight_decay)
        #mse = torch.nn.MSELoss()
        mse_wrt_noisy = np.zeros(num_iter_inner)
        import copy
        if find_best:
            best_net = copy.deepcopy(net3)
            best_mse = 1000000.0
        SSIMs = []
        PSNRs = []
        for i in range(num_iter_inner):
            #def closure3():
            
            optimizer3.zero_grad()
            out = net3(net_input.type(dtype))

            # training loss
            losss = mse( forwardm(out,mask2d) , img_noisy_var )

            losss.backward(retain_graph=retain_graph)
            optimizer3.step()
            mse_wrt_noisy[i] = losss.data.cpu().numpy()

            # the actual loss 
            true_loss = mse( Variable(out.data, requires_grad=False).type(dtype), img_clean_var.type(dtype) )

            if i % 100 == 0:

                trloss = losss.data
                true_loss = true_loss.data
                print ('Iteration %05d    Train loss %f  Actual loss %f' % (i, trloss,true_loss), '\r', end='')

            #    return losss   
            #losss = optimizer3.step(closure3)

            # if training loss improves by at least one percent, we found a new best net
            lossval = losss.data
            if best_mse > 1.005*lossval:
                best_mse = lossval
                best_net = copy.deepcopy(net3)
        net3 = best_net
        #scale_out,sover,pover,par_mse_n, par_mse_t, parni, pert_net = fitr(net,
        #                                                                unders_measurement,
        ##                                                                Variable(lsest).type(dtype),
        #                                                                mask2d,
        #                                                                num_iter=num_iter_inner,
        #                                                                LR=0.008,
        #                                                                apply_f = forwardm,
        #                                                                lsimg = lsimg,
        #                                                                find_best=True,
        #                                                                net_input = net_input,
        #                                                                OPTIMIZER = "adam"
        #                                                                )
        out_chs = net3( net_input.type(dtype) )[0]
        sh = out_chs.shape
        chs = int(sh[0]/2)
        imgs = torch.zeros( (chs,sh[1],sh[2]) ).type(dtype)
        for q in range(chs):
            imgs[q] = torch.sqrt( out_chs[2*q]**2 + out_chs[2*q+1]**2 )
        pert_rec = root_sum_of_squares(imgs)
        #pert_rec = center_crop(pert_rec,(320,320))
        #out_imgs = channels2imgs(out_chs)
        #pert_rec = torch.from_numpy(crop_center2(root_sum_of_squares2(out_imgs),320,320)).type(dtype)
        #pert_rec = torch.from_numpy( data_consistency(net3, net_input, mask1d, slice_ksp_torchtensor1) ).type(dtype)
        pert_rec, mean, std = transform.normalize_instance(pert_rec, eps=1e-11)
        pert_rec = pert_rec.clamp(-6, 6)
        pert_recs.append(pert_rec.data.cpu().numpy())
        
        def closure():
            
            optimizer.zero_grad()
            
            loss__ = loss(r,fixed_rec.type(dtype),pert_rec.type(dtype),lam)
            
            loss__.backward(retain_graph=retain_graph)
            
            loss_[j] = loss__.data.cpu().numpy()
            
            if i % 1 == 0:
                print ('Iteration %05d   loss %f' % (j, loss_[j]), '\r', end='')
            
            return loss__ 
        #print("\n{}\n".format(r.requires_grad))
        R.append(r.data.cpu())
        loss__ = optimizer.step(closure)
        print(r.grad.nonzero())
        ### new network for computing derivatives
        """print("\n***fit ConvDecoder at i = {} for derivatives***".format(i))
        net = convdecoder(out_size,in_size,output_depth,
                         num_layers,strides,num_channels, act_fun = nn.ReLU(),
                         skips=False,need_sigmoid=False,bias=False, need_last = True,
                         kernel_size=kernel_size,upsample_mode="nearest").type(dtype)
        net.load_state_dict(torch.load("./init"))
        ### fit this network to get an approximate derivative of f(A(x+r)) with respect to A(x+r) --> slightly perturb it with epsilon
        shape = inp.shape
        #H = Variable(torch.zeros(shape)).type(dtype)
        #H.data.uniform_(0.1,0.2)
        H /= torch.norm(H)
        H *= torch.norm(inp) / eps
        print(H.shape,unders_measurement.shape)
        unders_meas = unders_measurement.clone()
        unders_meas += H
        scale_out,sover,pover,par_mse_n, par_mse_t, parni, der_net = fitr(net,
                                                                        unders_meas.type(dtype),
                                                                        Variable(lsest).type(dtype),
                                                                        mask2d,
                                                                        num_iter=num_iter_inner,
                                                                        LR=0.008,
                                                                        apply_f = forwardm,
                                                                        lsimg = lsimg,
                                                                        find_best=True,
                                                                        net_input = net_input,
                                                                        OPTIMIZER = "adam",
                                                                        retain_graph=True,
                                                                        )
        #out_chs = pert_net( net_input.type(dtype) ).data.cpu().numpy()[0]
        #out_imgs = channels2imgs(out_chs)
        #pert_rec = torch.from_numpy(crop_center2(root_sum_of_squares2(out_imgs),320,320)).type(dtype)
        der_rec = torch.from_numpy( data_consistency(der_net, net_input, mask1d, slice_ksp_torchtensor1) ).type(dtype)
        der_rec, mean, std = transform.normalize_instance(der_rec, eps=1e-11)
        der_rec = der_rec.clamp(-6, 6)
        ### compute loss and perform the optimization step
        loss_[i] = loss(r,fixed_rec,pert_rec,H,lam)
        grad = loss.get_derivs(fixed_rec,pert_rec,der_rec,r,H,lam)
        r -= LR*grad
        print("\nloss at iteration{}:".format(i),loss_[i])
        
        R.append(r.data.cpu())
        print(2*"\n")"""
        #loss = optimizer.step(closure)
        with open("./outputs/untrainedrunner_test/results","wb") as fn:
            pickle.dump([R,ksp,loss_,fixed_rec.data.cpu().numpy(),pert_recs],fn)
    return R,net_input, loss_, fixed_rec.data.cpu().numpy(), pert_recs

def myrunner_untrained_test2(ksp,
                            net=None,
                            num_iter = 20,
                            num_iter_inner = 10000,
                            LR = 0.01,
                            lam = 0.1,
                            eps = 1e2,
                            OPTIMIZER='adam',
                            mask = None,
                            mask1d = None,
                            mask2d = None,
                            lr_decay_epoch = 0,
                            weight_decay=0,
                            loss_type="MSE",
                            retain_graph = False,
                            find_best = True,
                          ):
    ################ main optimization steup: perturbation finder ################
    shape = ksp.shape
    print("perturbation shape: ", shape)
    r = Variable(torch.zeros(shape)).type(dtype)
    r.data.uniform_()
    r.data *= torch.norm(ksp)/torch.norm(r)
    
    r = r.type(dtype)
    r_saved = r.data.clone()
    
    #r.requires_grad = True

    loss_ = np.zeros(num_iter)
    loss = MyLoss()
    ################ ################
    """
    with open("masks","rb") as fn:
        [mask,mask1d,mask2d,net_input] = pickle.load(fn)
    in_size = [4,4]
    kernel_size = 3
    num_channels = 60#128
    num_layers = 4#6
    strides = [1]*(num_layers-1)
    output_depth = ksp.numpy().shape[0]*2
    out_size = ksp.numpy().shape[1:-1]
    width,height = in_size
    shape = [1,num_channels, width, height]
    print("network input shape: ", shape)
    net_input = Variable(torch.zeros(shape)).type(dtype)
    net_input.data.uniform_()
    net_input.data *= 1./10
    net_input = net_input.type(dtype)
    ##### fit the network for reconstruction without perturbation #####
    net = convdecoder(out_size,in_size,output_depth,
                         num_layers,strides,num_channels, act_fun = nn.ReLU(),
                         skips=False,need_sigmoid=False,bias=True, need_last = True,
                         kernel_size=kernel_size,upsample_mode="nearest").type(dtype)
    net.load_state_dict(torch.load("./init"))
    #torch.save(net.state_dict(), "./init")
    ### fix scaling for ConvDecoder
    scaling_factor,_ = get_scale_factor(net,
                                       num_channels,
                                       in_size,
                                       ksp,
                                       ni=net_input)
    slice_ksp_torchtensor1 = ksp * scaling_factor
    masked_kspace, mask = transform.apply_mask(slice_ksp_torchtensor1, mask = mask)
    unders_measurement = np_to_var( masked_kspace.data.cpu().numpy() ).type(dtype)
    sampled_image2 = transform.ifft2(masked_kspace)
    measurement = slice_ksp_torchtensor1.unsqueeze(0).type(dtype) 
    lsimg = lsreconstruction(measurement)
    out = []
    for img in sampled_image2:
        out += [ img[:,:,0].numpy() , img[:,:,1].numpy() ]
    lsest = torch.tensor(np.array([out]))

    scale_out,sover,pover,par_mse_n, par_mse_t, parni, fixed_net = fitr( net,
                                                                        unders_measurement,
                                                                        Variable(lsest).type(dtype),
                                                                        mask2d,
                                                                        num_iter=num_iter_inner,
                                                                        LR=0.1,
                                                                        apply_f = forwardm,
                                                                        lsimg = lsimg,
                                                                        find_best=True,
                                                                        net_input = net_input,
                                                                        OPTIMIZER = "adam"
                                                                       )
    #out_chs = fixed_net( net_input.type(dtype) ).data.cpu().numpy()[0]
    #out_imgs = channels2imgs(out_chs)
    #fixed_rec = torch.from_numpy(crop_center2(root_sum_of_squares2(out_imgs),320,320)).type(dtype)
    fixed_rec = torch.from_numpy( data_consistency(fixed_net, net_input, mask1d, slice_ksp_torchtensor1) ).type(dtype)
    fixed_rec, mean, std = transform.normalize_instance(fixed_rec, eps=1e-11)
    fixed_rec = fixed_rec.clamp(-6, 6)
    print(fixed_rec.shape,ksp.shape)
    """
    ### fixed reconstruction from non-perturbed data
    masked_kspace, mask = transform.apply_mask(ksp.type(dtype), mask = mask.type(dtype))
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
    fixed_rec = net(inp2.type(dtype))[0]
    
    pert_recs = []
    R = []
    
    
    for i in range(num_iter):
        print("perturbation norm:",torch.norm(net_input))
        inp = net_input + ksp
        masked_kspace, mask = transform.apply_mask(inp, mask = mask.type(dtype))
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
        pert_rec = net(inp2.type(dtype))
        pert_recs.append(pert_rec.data.cpu().numpy()[0])
        ### fit this network to get an approximate derivative of f(A(x+r)) with respect to A(x+r) --> slightly perturb it with epsilon
        shape = inp.shape
        #H = Variable(torch.zeros(shape)).type(dtype)
        #H.data.uniform_(0.1,0.2)
        
        H = torch.randn(shape).type(dtype)+1e-9
        #H = torch.zeros(shape).type(dtype) + 1
        H /= torch.norm(H)
        H *= torch.norm(inp) / eps
        inp2 = inp + H
        masked_kspace, mask = transform.apply_mask(inp2, mask = mask.type(dtype))
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
        der_rec = net(inp2.type(dtype))
        ########
        inp2 = inp - H
        masked_kspace, mask = transform.apply_mask(inp2, mask = mask.type(dtype))
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
        der_rec2 = net(inp2.type(dtype))
        
        ### compute loss and perform the optimization step
        loss_[i] = loss(net_input,fixed_rec,pert_rec,H,lam)
        grad = loss.get_derivs(fixed_rec.data.cpu(),der_rec2.data.cpu(),der_rec.data.cpu(),net_input.data.cpu(),H.data.cpu(),lam)
        print("\nloss at iteration{}:".format(i),loss_[i]) 
        """
    num_channels = 100 #256
    num_layers = 5
    strides = [1]*(num_layers-1)
    in_size = [4,4]
    kernel_size = 3
    output_depth = ksp.numpy().shape[0]*2
    out_size = ksp.numpy().shape[1:-1]
    width,height = in_size
    shape = [1,num_channels, width, height]
    print("network input shape: ", shape)
    net_input = Variable(torch.zeros(shape)).type(dtype)
    net_input.data.uniform_()
    net_input.data *= 1./10
    net_input = net_input.type(dtype)
    
    ##### fit the network for reconstruction without perturbation #####
    net = convdecoder(out_size,in_size,output_depth,
                         num_layers,strides,num_channels, act_fun = nn.ReLU(),
                         skips=False,need_sigmoid=False,bias=False, need_last = True,
                         kernel_size=kernel_size,upsample_mode="nearest").type(dtype)
    torch.save(net.state_dict(), "./init")
    ### fix scaling for ConvDecoder
    scaling_factor,_ = get_scale_factor(net,
                                       num_channels,
                                       in_size,
                                       ksp,
                                       ni=net_input)
    slice_ksp_torchtensor1 = ksp * scaling_factor
    masked_kspace, mask = transform.apply_mask(slice_ksp_torchtensor1, mask = mask)
    unders_measurement = np_to_var( masked_kspace.data.cpu().numpy() ).type(dtype)
    sampled_image2 = transform.ifft2(masked_kspace)
    measurement = slice_ksp_torchtensor1.unsqueeze(0).type(dtype) 
    lsimg = lsreconstruction(measurement)
    out = []
    for img in sampled_image2:
        out += [ img[:,:,0].numpy() , img[:,:,1].numpy() ]
    lsest = torch.tensor(np.array([out]))

    scale_out,sover,pover,par_mse_n, par_mse_t, parni, fixed_net = fitr( net,
                                                                        unders_measurement,
                                                                        Variable(lsest).type(dtype),
                                                                        mask2d,
                                                                        num_iter=num_iter_inner,
                                                                        LR=0.008,
                                                                        apply_f = forwardm,
                                                                        lsimg = lsimg,
                                                                        find_best=True,
                                                                        net_input = net_input,
                                                                        OPTIMIZER = "adam"
                                                                       )
    #out_chs = fixed_net( net_input.type(dtype) ).data.cpu().numpy()[0]
    #out_imgs = channels2imgs(out_chs)
    #fixed_rec = torch.from_numpy(crop_center2(root_sum_of_squares2(out_imgs),320,320)).type(dtype)
    fixed_rec = torch.from_numpy( data_consistency(fixed_net, net_input, mask1d, slice_ksp_torchtensor1) ).type(dtype)
    fixed_rec, mean, std = transform.normalize_instance(fixed_rec, eps=1e-11)
    fixed_rec = fixed_rec.clamp(-6, 6)
    pert_recs = []
    R = []
    
    indices = []
    for i in range(ksp.shape[1]):
        for j in range(ksp.shape[2]):
            for k in range(ksp.shape[3]):
                p = random.random()
                if p < 0.001:
                    indices.append((0,i,j,k))
    print("\n%{} of elements picked for perturbation".format(100*len(indices)/np.prod(ksp.numpy().shape)))
    for i in range(num_iter):
        ### prepare input for ConvDecoder
        # create the network
        net = convdecoder(out_size,in_size,output_depth,
                         num_layers,strides,num_channels, act_fun = nn.ReLU(),
                         skips=False,need_sigmoid=False,bias=False, need_last = True,
                         kernel_size=kernel_size,upsample_mode="nearest").type(dtype)
        net.load_state_dict(torch.load("./init"))
        # f(A(x+r)) recovery
        print("\n***fit ConvDecoder at i = {}***".format(i))
        print("norms:",torch.norm(r),torch.norm(ksp))
        inp = r + ksp.type(dtype)
        scaling_factor,_ = get_scale_factor(net,
                                           num_channels,
                                           in_size,
                                           inp.data,
                                           ni=net_input)
        slice_ksp_torchtensor1 = inp.data.cpu() * scaling_factor
        masked_kspace, mask = transform.apply_mask(slice_ksp_torchtensor1, mask = mask)
        unders_measurement = np_to_var( masked_kspace.data.cpu().numpy() ).type(dtype)
        sampled_image2 = transform.ifft2(masked_kspace)
        measurement = slice_ksp_torchtensor1.unsqueeze(0).type(dtype) 
        lsimg = lsreconstruction(measurement)
        # fit the network
        out = []
        for img in sampled_image2:
            out += [ img[:,:,0].numpy() , img[:,:,1].numpy() ]
        lsest = torch.tensor(np.array([out]))

        scale_out,sover,pover,par_mse_n, par_mse_t, parni, pert_net = fitr(net,
                                                                        unders_measurement,
                                                                        Variable(lsest).type(dtype),
                                                                        mask2d,
                                                                        num_iter=num_iter_inner,
                                                                        LR=0.008,
                                                                        apply_f = forwardm,
                                                                        lsimg = lsimg,
                                                                        find_best=True,
                                                                        net_input = net_input,
                                                                        OPTIMIZER = "adam"
                                                                        )
        #out_chs = pert_net( net_input.type(dtype) ).data.cpu().numpy()[0]
        #out_imgs = channels2imgs(out_chs)
        #pert_rec = torch.from_numpy(crop_center2(root_sum_of_squares2(out_imgs),320,320)).type(dtype)
        pert_rec = torch.from_numpy( data_consistency(pert_net, net_input, mask1d, slice_ksp_torchtensor1) ).type(dtype)
        pert_rec, mean, std = transform.normalize_instance(pert_rec, eps=1e-11)
        pert_rec = pert_rec.clamp(-6, 6)
        pert_recs.append(pert_rec.data.cpu().numpy())"""

    
        R.append(r.data.cpu())
        r -= LR*(-grad.type(dtype)+lam*r)
        print(2*"\n")
        #loss = optimizer.step(closure)
        with open("./outputs/untrainedrunner_test/results","wb") as fn:
            pickle.dump([R,ksp,loss_,fixed_rec.data.cpu().numpy(),pert_recs,mask,mask1d,mask2d],fn)
        #del(der_rec,pert_rec,inp,grad)
        #torch.cuda.empty_cache()
        
    return R,net_input, loss_, fixed_rec.data.cpu().numpy(), pert_recs

def myrunner_untrained_test(ksp,
                            net=None,
                            num_iter = 20,
                            num_iter_inner = 10000,
                            LR = 0.01,
                            lam = 0.1,
                            eps = 1e2,
                            OPTIMIZER='adam',
                            mask = None,
                            mask1d = None,
                            mask2d = None,
                            lr_decay_epoch = 0,
                            weight_decay=0,
                            loss_type="MSE",
                            retain_graph = False,
                            find_best = True,
                          ):
    ################ main optimization steup: perturbation finder ################
    shape = ksp.shape
    print("perturbation shape: ", shape)
    r = Variable(torch.zeros(shape)).type(dtype)
    r.data.uniform_()
    #r.data *= 1/torch.norm(ksp)#1./1e3
    #indices = []
    indices = torch.from_numpy(np.random.rand(shape[0],shape[1],shape[2],shape[3]))
    indices[indices>0.2] = 0
    indices[indices!=0] = 1
    r[indices==0] = 0
    #for l in range(ksp.shape[0]):
    #    for i in range(ksp.shape[1]):
    #        for j in range(ksp.shape[2]):
    #            for k in range(ksp.shape[3]):
    #                p = random.random()
    #                if p < 0.001:
    #                    indices.append((l,i,j,k))
    #                else:
    #                    r[l,i,j,k] = 0
    
    print("\n%{} of elements picked for perturbation".format(100*indices.sum()/np.prod(ksp.numpy().shape)))
    inds = torch.nonzero(r)
    r.data *= torch.norm(ksp)/torch.norm(r)
    
    r = r.type(dtype)
    r_saved = r.data.clone()
    
    #r.requires_grad = True

    loss_ = np.zeros(num_iter)
    loss = MyLoss()
    ################ ################
    
    with open("masks","rb") as fn:
        [mask,mask1d,mask2d,net_input] = pickle.load(fn)
    in_size = [4,4]
    kernel_size = 3
    num_channels = 60#128
    num_layers = 4#6
    strides = [1]*(num_layers-1)
    output_depth = ksp.numpy().shape[0]*2
    out_size = ksp.numpy().shape[1:-1]
    width,height = in_size
    shape = [1,num_channels, width, height]
    print("network input shape: ", shape)
    net_input = Variable(torch.zeros(shape)).type(dtype)
    net_input.data.uniform_()
    net_input.data *= 1./10
    net_input = net_input.type(dtype)
    ##### fit the network for reconstruction without perturbation #####
    net = convdecoder(out_size,in_size,output_depth,
                         num_layers,strides,num_channels, act_fun = nn.ReLU(),
                         skips=False,need_sigmoid=False,bias=True, need_last = True,
                         kernel_size=kernel_size,upsample_mode="nearest").type(dtype)
    net.load_state_dict(torch.load("./init"))
    #torch.save(net.state_dict(), "./init")
    ### fix scaling for ConvDecoder
    scaling_factor,_ = get_scale_factor(net,
                                       num_channels,
                                       in_size,
                                       ksp,
                                       ni=net_input)
    slice_ksp_torchtensor1 = ksp * scaling_factor
    masked_kspace, mask = transform.apply_mask(slice_ksp_torchtensor1, mask = mask)
    unders_measurement = np_to_var( masked_kspace.data.cpu().numpy() ).type(dtype)
    sampled_image2 = transform.ifft2(masked_kspace)
    measurement = slice_ksp_torchtensor1.unsqueeze(0).type(dtype) 
    lsimg = lsreconstruction(measurement)
    out = []
    for img in sampled_image2:
        out += [ img[:,:,0].numpy() , img[:,:,1].numpy() ]
    lsest = torch.tensor(np.array([out]))

    scale_out,sover,pover,par_mse_n, par_mse_t, parni, fixed_net = fitr( net,
                                                                        unders_measurement,
                                                                        Variable(lsest).type(dtype),
                                                                        mask2d,
                                                                        num_iter=num_iter_inner,
                                                                        LR=0.1,
                                                                        apply_f = forwardm,
                                                                        lsimg = lsimg,
                                                                        find_best=True,
                                                                        net_input = net_input,
                                                                        OPTIMIZER = "adam"
                                                                       )
    #out_chs = fixed_net( net_input.type(dtype) ).data.cpu().numpy()[0]
    #out_imgs = channels2imgs(out_chs)
    #fixed_rec = torch.from_numpy(crop_center2(root_sum_of_squares2(out_imgs),320,320)).type(dtype)
    fixed_rec = torch.from_numpy( data_consistency(fixed_net, net_input, mask1d, slice_ksp_torchtensor1) ).type(dtype)
    fixed_rec, mean, std = transform.normalize_instance(fixed_rec, eps=1e-11)
    fixed_rec = fixed_rec.clamp(-6, 6)
    print(fixed_rec.shape,ksp.shape)
    """
    ### fixed reconstruction from non-perturbed data
    masked_kspace, mask = transform.apply_mask(ksp.type(dtype), mask = mask.type(dtype))
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
    fixed_rec = net(inp2.type(dtype))[0]
    
    pert_recs = []
    R = []
    """
    
    """for i in range(num_iter):
        print("perturbation norm:",torch.norm(net_input))
        inp = net_input + ksp
        masked_kspace, mask = transform.apply_mask(inp, mask = mask.type(dtype))
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
        pert_rec = net(inp2.type(dtype))
        pert_recs.append(pert_rec.data.cpu().numpy()[0])
        ### fit this network to get an approximate derivative of f(A(x+r)) with respect to A(x+r) --> slightly perturb it with epsilon
        shape = inp.shape
        #H = Variable(torch.zeros(shape)).type(dtype)
        #H.data.uniform_(0.1,0.2)
        
        H = torch.randn(shape).type(dtype)+1e-9
        #H = torch.zeros(shape).type(dtype) + 1
        H /= torch.norm(H)
        H *= torch.norm(inp) / eps
        inp2 = inp + H
        masked_kspace, mask = transform.apply_mask(inp2, mask = mask.type(dtype))
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
        der_rec = net(inp2.type(dtype))
        ########
        inp2 = inp - H
        masked_kspace, mask = transform.apply_mask(inp2, mask = mask.type(dtype))
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
        der_rec2 = net(inp2.type(dtype))
        
        ### compute loss and perform the optimization step
        loss_[i] = loss(net_input,fixed_rec,pert_rec,H,lam)
        grad = loss.get_derivs(fixed_rec.data.cpu(),der_rec2.data.cpu(),der_rec.data.cpu(),net_input.data.cpu(),H.data.cpu(),lam)
        print("\nloss at iteration{}:".format(i),loss_[i])""" """
    num_channels = 100 #256
    num_layers = 5
    strides = [1]*(num_layers-1)
    in_size = [4,4]
    kernel_size = 3
    output_depth = ksp.numpy().shape[0]*2
    out_size = ksp.numpy().shape[1:-1]
    width,height = in_size
    shape = [1,num_channels, width, height]
    print("network input shape: ", shape)
    net_input = Variable(torch.zeros(shape)).type(dtype)
    net_input.data.uniform_()
    net_input.data *= 1./10
    net_input = net_input.type(dtype)
    
    ##### fit the network for reconstruction without perturbation #####
    net = convdecoder(out_size,in_size,output_depth,
                         num_layers,strides,num_channels, act_fun = nn.ReLU(),
                         skips=False,need_sigmoid=False,bias=False, need_last = True,
                         kernel_size=kernel_size,upsample_mode="nearest").type(dtype)
    torch.save(net.state_dict(), "./init")
    ### fix scaling for ConvDecoder
    scaling_factor,_ = get_scale_factor(net,
                                       num_channels,
                                       in_size,
                                       ksp,
                                       ni=net_input)
    slice_ksp_torchtensor1 = ksp * scaling_factor
    masked_kspace, mask = transform.apply_mask(slice_ksp_torchtensor1, mask = mask)
    unders_measurement = np_to_var( masked_kspace.data.cpu().numpy() ).type(dtype)
    sampled_image2 = transform.ifft2(masked_kspace)
    measurement = slice_ksp_torchtensor1.unsqueeze(0).type(dtype) 
    lsimg = lsreconstruction(measurement)
    out = []
    for img in sampled_image2:
        out += [ img[:,:,0].numpy() , img[:,:,1].numpy() ]
    lsest = torch.tensor(np.array([out]))

    scale_out,sover,pover,par_mse_n, par_mse_t, parni, fixed_net = fitr( net,
                                                                        unders_measurement,
                                                                        Variable(lsest).type(dtype),
                                                                        mask2d,
                                                                        num_iter=num_iter_inner,
                                                                        LR=0.008,
                                                                        apply_f = forwardm,
                                                                        lsimg = lsimg,
                                                                        find_best=True,
                                                                        net_input = net_input,
                                                                        OPTIMIZER = "adam"
                                                                       )
    #out_chs = fixed_net( net_input.type(dtype) ).data.cpu().numpy()[0]
    #out_imgs = channels2imgs(out_chs)
    #fixed_rec = torch.from_numpy(crop_center2(root_sum_of_squares2(out_imgs),320,320)).type(dtype)
    fixed_rec = torch.from_numpy( data_consistency(fixed_net, net_input, mask1d, slice_ksp_torchtensor1) ).type(dtype)
    fixed_rec, mean, std = transform.normalize_instance(fixed_rec, eps=1e-11)
    fixed_rec = fixed_rec.clamp(-6, 6)
    pert_recs = []
    R = []
    
    indices = []
    for i in range(ksp.shape[1]):
        for j in range(ksp.shape[2]):
            for k in range(ksp.shape[3]):
                p = random.random()
                if p < 0.001:
                    indices.append((0,i,j,k))
    print("\n%{} of elements picked for perturbation".format(100*len(indices)/np.prod(ksp.numpy().shape)))
    for i in range(num_iter):
        ### prepare input for ConvDecoder
        # create the network
        net = convdecoder(out_size,in_size,output_depth,
                         num_layers,strides,num_channels, act_fun = nn.ReLU(),
                         skips=False,need_sigmoid=False,bias=False, need_last = True,
                         kernel_size=kernel_size,upsample_mode="nearest").type(dtype)
        net.load_state_dict(torch.load("./init"))
        # f(A(x+r)) recovery
        print("\n***fit ConvDecoder at i = {}***".format(i))
        print("norms:",torch.norm(r),torch.norm(ksp))
        inp = r + ksp.type(dtype)
        scaling_factor,_ = get_scale_factor(net,
                                           num_channels,
                                           in_size,
                                           inp.data,
                                           ni=net_input)
        slice_ksp_torchtensor1 = inp.data.cpu() * scaling_factor
        masked_kspace, mask = transform.apply_mask(slice_ksp_torchtensor1, mask = mask)
        unders_measurement = np_to_var( masked_kspace.data.cpu().numpy() ).type(dtype)
        sampled_image2 = transform.ifft2(masked_kspace)
        measurement = slice_ksp_torchtensor1.unsqueeze(0).type(dtype) 
        lsimg = lsreconstruction(measurement)
        # fit the network
        out = []
        for img in sampled_image2:
            out += [ img[:,:,0].numpy() , img[:,:,1].numpy() ]
        lsest = torch.tensor(np.array([out]))

        scale_out,sover,pover,par_mse_n, par_mse_t, parni, pert_net = fitr(net,
                                                                        unders_measurement,
                                                                        Variable(lsest).type(dtype),
                                                                        mask2d,
                                                                        num_iter=num_iter_inner,
                                                                        LR=0.008,
                                                                        apply_f = forwardm,
                                                                        lsimg = lsimg,
                                                                        find_best=True,
                                                                        net_input = net_input,
                                                                        OPTIMIZER = "adam"
                                                                        )
        #out_chs = pert_net( net_input.type(dtype) ).data.cpu().numpy()[0]
        #out_imgs = channels2imgs(out_chs)
        #pert_rec = torch.from_numpy(crop_center2(root_sum_of_squares2(out_imgs),320,320)).type(dtype)
        pert_rec = torch.from_numpy( data_consistency(pert_net, net_input, mask1d, slice_ksp_torchtensor1) ).type(dtype)
        pert_rec, mean, std = transform.normalize_instance(pert_rec, eps=1e-11)
        pert_rec = pert_rec.clamp(-6, 6)
        pert_recs.append(pert_rec.data.cpu().numpy())"""

    pert_recs = []
    R = []
    
    for i in range(num_iter):
        print("perturbation norm:",torch.norm(r))
        if i>0:
            print("grad norm, l2 norm:",torch.norm(grad),lam*torch.norm(r))
        """inp = r + ksp.type(dtype)
        masked_kspace, mask = transform.apply_mask(inp, mask = mask.type(dtype))
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
        pert_rec = net(inp2.type(dtype))[0]
        pert_recs.append(pert_rec.data.cpu().numpy())
        """
        net = convdecoder(out_size,in_size,output_depth,
                         num_layers,strides,num_channels, act_fun = nn.ReLU(),
                         skips=False,need_sigmoid=False,bias=True, need_last = True,
                         kernel_size=kernel_size,upsample_mode="nearest").type(dtype)
        net.load_state_dict(torch.load("./init"))
        # f(A(x+r)) recovery
        print("\n***fit ConvDecoder at i = {}***".format(i))
        print("norms:",torch.norm(r),torch.norm(ksp))
        inp = r + ksp.type(dtype)
        scaling_factor,_ = get_scale_factor(net,
                                           num_channels,
                                           in_size,
                                           inp.data,
                                           ni=net_input)
        slice_ksp_torchtensor1 = inp.data.cpu() * scaling_factor
        masked_kspace, mask = transform.apply_mask(slice_ksp_torchtensor1, mask = mask)
        unders_measurement = np_to_var( masked_kspace.data.cpu().numpy() ).type(dtype)
        sampled_image2 = transform.ifft2(masked_kspace)
        measurement = slice_ksp_torchtensor1.unsqueeze(0).type(dtype) 
        lsimg = lsreconstruction(measurement)
        # fit the network
        out = []
        for img in sampled_image2:
            out += [ img[:,:,0].numpy() , img[:,:,1].numpy() ]
        lsest = torch.tensor(np.array([out]))

        scale_out,sover,pover,par_mse_n, par_mse_t, parni, pert_net = fitr(net,
                                                                        unders_measurement,
                                                                        Variable(lsest).type(dtype),
                                                                        mask2d,
                                                                        num_iter=num_iter_inner,
                                                                        LR=0.1,
                                                                        apply_f = forwardm,
                                                                        lsimg = lsimg,
                                                                        find_best=True,
                                                                        net_input = net_input,
                                                                        OPTIMIZER = "adam"
                                                                        )
        #out_chs = pert_net( net_input.type(dtype) ).data.cpu().numpy()[0]
        #out_imgs = channels2imgs(out_chs)
        #pert_rec = torch.from_numpy(crop_center2(root_sum_of_squares2(out_imgs),320,320)).type(dtype)
        pert_rec = torch.from_numpy( data_consistency(pert_net, net_input, mask1d, slice_ksp_torchtensor1) ).type(dtype)
        pert_rec, mean, std = transform.normalize_instance(pert_rec, eps=1e-11)
        pert_rec = pert_rec.clamp(-6, 6)
        pert_recs.append(pert_rec.data.cpu().numpy())
        
        ctr = 0
        s = time.time()
        grad = torch.zeros(ksp.shape)
        #for j in range(ksp.shape[0]):
            #print( "j={}".format(j) )
        #    for m in range(ksp.shape[1]):
        #        for n in range(ksp.shape[2]):
        #            for q in range(ksp.shape[3]):
        #                if (j,m,n,q) not in indices:
        #                    continue
        for ind in inds:
            """h = inp.mean()/eps
            inp2 = inp.clone()
            inp2[ind[0],ind[1],ind[2],ind[3]] += h
            masked_kspace, mask = transform.apply_mask(inp2, mask = mask.type(dtype))
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
            der_rec = net(inp2.type(dtype)).data.cpu()[0]
            ### new network for computing derivatives
            ### right side
            #print("\n***fit ConvDecoder at i = {} for derivatives***".format(i))
            """
            net = convdecoder(out_size,in_size,output_depth,
                             num_layers,strides,num_channels, act_fun = nn.ReLU(),
                             skips=False,need_sigmoid=False,bias=True, need_last = True,
                             kernel_size=kernel_size,upsample_mode="nearest").type(dtype)
            net.load_state_dict(torch.load("./init"))
            ### fit this network to get an approximate derivative of f(A(x+r)) with respect to A(x+r) --> slightly perturb it with epsilon
            h = inp.mean()/eps
            unders_meas = unders_measurement.clone()
            unders_meas[0,ind[0],ind[1],ind[2],ind[3]] += h
            scale_out,sover,pover,par_mse_n, par_mse_t, parni, der_net = fitr(net,
                                                                            unders_meas.type(dtype),
                                                                            Variable(lsest).type(dtype),
                                                                            mask2d,
                                                                            num_iter=num_iter_inner,
                                                                            LR=0.1,
                                                                            apply_f = forwardm,
                                                                            lsimg = lsimg,
                                                                            find_best=True,
                                                                            net_input = net_input,
                                                                            OPTIMIZER = "adam",
                                                                            retain_graph=True,
                                                                            )
            #out_chs = pert_net( net_input.type(dtype) ).data.cpu().numpy()[0]
            #out_imgs = channels2imgs(out_chs)
            #pert_rec = torch.from_numpy(crop_center2(root_sum_of_squares2(out_imgs),320,320)).type(dtype)
            der_rec = torch.from_numpy( data_consistency(der_net, net_input, mask1d, slice_ksp_torchtensor1) ).type(dtype)
            der_rec, mean, std = transform.normalize_instance(der_rec, eps=1e-11)
            der_rec = der_rec.clamp(-6, 6).data.cpu()

            """### left side
            net = convdecoder(out_size,in_size,output_depth,
                             num_layers,strides,num_channels, act_fun = nn.ReLU(),
                             skips=False,need_sigmoid=False,bias=False, need_last = True,
                             kernel_size=kernel_size,upsample_mode="nearest").type(dtype)
            net.load_state_dict(torch.load("./init"))
            ### fit this network to get an approximate derivative of f(A(x+r)) with respect to A(x+r) --> slightly perturb it with epsilon
            unders_meas = unders_measurement.clone()
            unders_meas[0,j,m,n,q] -= h
            scale_out,sover,pover,par_mse_n, par_mse_t, parni, der_net = fitr(net,
                                                                            unders_meas.type(dtype),
                                                                            Variable(lsest).type(dtype),
                                                                            mask2d,
                                                                            num_iter=num_iter_inner,
                                                                            LR=0.008,
                                                                            apply_f = forwardm,
                                                                            lsimg = lsimg,
                                                                            find_best=True,
                                                                            net_input = net_input,
                                                                            OPTIMIZER = "adam",
                                                                            retain_graph=True,
                                                                            )
            #out_chs = pert_net( net_input.type(dtype) ).data.cpu().numpy()[0]
            #out_imgs = channels2imgs(out_chs)
            #pert_rec = torch.from_numpy(crop_center2(root_sum_of_squares2(out_imgs),320,320)).type(dtype)
            der_rec2 = torch.from_numpy( data_consistency(der_net, net_input, mask1d, slice_ksp_torchtensor1) ).type(dtype)
            der_rec2, mean, std = transform.normalize_instance(der_rec2, eps=1e-11)
            der_rec2 = der_rec2.clamp(-6, 6).data.cpu()"""
            grad[ind[0],ind[1],ind[2],ind[3]] = (torch.norm(fixed_rec.data.cpu()-der_rec)**2 - torch.norm(fixed_rec.data.cpu()-pert_rec.data.cpu())**2) / (h) / np.prod(fixed_rec.data.cpu().numpy().shape)
            #if ctr % 100 == 0:
            #    print('%',ctr*100/np.prod(ksp.data.cpu().numpy().shape), time.time()-s,"seconds")
            #    s = time.time()
            #ctr += 1
        #r[grad==0] = 0
        R.append(r.data.cpu())
        r -= LR*(-grad.type(dtype)+lam*r)
        print(2*"\n")
        #loss = optimizer.step(closure)
        with open("./outputs/untrainedrunner_test/results","wb") as fn:
            pickle.dump([R,ksp,loss_,fixed_rec.data.cpu().numpy(),pert_recs,mask,mask1d,mask2d],fn)
        #del(der_rec,pert_rec,inp,grad)
        #torch.cuda.empty_cache()
        
    return R,net_input, loss_, fixed_rec.data.cpu().numpy(), pert_recs




def runner_untrained(ksp,
                    num_iter = 20,
                    num_iter_inner = 10000,
                    LR = 0.01,
                    OPTIMIZER='adam',
                    mask = None,
                    mask1d = None,
                    mask2d = None,
                    lr_decay_epoch = 0,
                    weight_decay=0,
                    loss_type="MSE",
                    retain_graph = False,
                    find_best = True,
                   ):
    ################ main optimization steup: perturbation finder ################
    shape = ksp.shape
    print("perturbation shape: ", shape)
    r = Variable(torch.zeros(shape)).type(dtype)
    r.data.uniform_()
    #r.data *= 1/torch.norm(ksp)#1./1e3
    r.data *= torch.norm(ksp)/torch.norm(r)
    
    r = r.type(dtype)
    r_saved = r.data.clone()
    
    r.requires_grad = True
    p = [r]

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
    ################ ################
    
    ################ sub optimization: fitting ConvDecoder (or any untrained network) ################
    num_channels = 160 #256
    num_layers = 8
    strides = [1]*(num_layers-1)
    in_size = [8,4]
    kernel_size = 3
    output_depth = ksp.shape[0]*2
    out_size = ksp.shape[1:-1]
    width,height = in_size
    shape = [1,num_channels, width, height]
    print("network input shape: ", shape)
    net_input = Variable(torch.zeros(shape)).type(dtype)
    net_input.data.uniform_()
    net_input.data *= 1./10
    net_input = net_input.type(dtype)
    
    ##### fit the network for reconstruction without perturbation #####
    net = convdecoder(out_size,in_size,output_depth,
                         num_layers,strides,num_channels, act_fun = nn.ReLU(),
                         skips=False,need_sigmoid=False,bias=False, need_last = True,
                         kernel_size=kernel_size,upsample_mode="nearest").type(dtype)
    torch.save(net.state_dict(), "./init")
    ### fix scaling for ConvDecoder
    scaling_factor,_ = get_scale_factor(net,
                                       num_channels,
                                       in_size,
                                       ksp,
                                       ni=net_input)
    slice_ksp_torchtensor1 = ksp * scaling_factor
    masked_kspace, mask = transform.apply_mask(slice_ksp_torchtensor1, mask = mask)
    unders_measurement = np_to_var( masked_kspace.data.cpu().numpy() ).type(dtype)
    sampled_image2 = transform.ifft2(masked_kspace)
    measurement = slice_ksp_torchtensor1.unsqueeze(0).type(dtype) 
    lsimg = lsreconstruction(measurement)
    out = []
    for img in sampled_image2:
        out += [ img[:,:,0].numpy() , img[:,:,1].numpy() ]
    lsest = torch.tensor(np.array([out]))

    scale_out,sover,pover,par_mse_n, par_mse_t, parni, fixed_net = fitr( net,
                                                                        unders_measurement,
                                                                        Variable(lsest).type(dtype),
                                                                        mask2d,
                                                                        num_iter=num_iter_inner,
                                                                        LR=0.008,
                                                                        apply_f = forwardm,
                                                                        lsimg = lsimg,
                                                                        find_best=True,
                                                                        net_input = net_input,
                                                                        OPTIMIZER = "adam"
                                                                     )
    fixed_outs = fixed_net( net_input.type(dtype) )
    #outs = fixed_outs.data.cpu().numpy()[0]
    #fixed_imgs = channels2imgs(outs)
    #fixed_rec = torch.from_numpy(crop_center2(root_sum_of_squares2(fixed_imgs),320,320)).type(dtype)
    fixed_rec = torch.from_numpy( data_consistency(fixed_net, net_input, mask1d, slice_ksp_torchtensor1) ).type(dtype)
    
    pert_recs = []
    R = []
    
    for i in range(num_iter):
        ### prepare input for ConvDecoder
        # create the network
        net = convdecoder(out_size,in_size,output_depth,
                         num_layers,strides,num_channels, act_fun = nn.ReLU(),
                         skips=False,need_sigmoid=False,bias=False, need_last = True,
                         kernel_size=kernel_size,upsample_mode="nearest").type(dtype)
        net.load_state_dict(torch.load("./init"))
        # f(A(x+r)) recovery
        print("***fit ConvDecoder at i = {}***".format(i))
        print("norms:",torch.norm(r),torch.norm(ksp),'\n')
        inp = r + ksp.type(dtype)
        scaling_factor,_ = get_scale_factor(net,
                                           num_channels,
                                           in_size,
                                           inp.data,
                                           ni=net_input)
        slice_ksp_torchtensor1 = inp.data.cpu() * scaling_factor
        masked_kspace, mask = transform.apply_mask(slice_ksp_torchtensor1, mask = mask)
        unders_measurement = np_to_var( masked_kspace.data.cpu().numpy() ).type(dtype)
        sampled_image2 = transform.ifft2(masked_kspace)
        measurement = slice_ksp_torchtensor1.unsqueeze(0).type(dtype) 
        lsimg = lsreconstruction(measurement)
        # fit the network
        out = []
        for img in sampled_image2:
            out += [ img[:,:,0].numpy() , img[:,:,1].numpy() ]
        lsest = torch.tensor(np.array([out]))

        scale_out,sover,pover,par_mse_n, par_mse_t, parni, pert_net = fitr(net,
                                                                        unders_measurement,
                                                                        Variable(lsest).type(dtype),
                                                                        mask2d,
                                                                        num_iter=num_iter_inner,
                                                                        LR=0.008,
                                                                        apply_f = forwardm,
                                                                        lsimg = lsimg,
                                                                        find_best=True,
                                                                        net_input = net_input,
                                                                        OPTIMIZER = "adam"
                                                                        )
        pert_outs = pert_net( net_input.type(dtype) )
        #outs = pert_outs.data.cpu().numpy()[0]
        #pert_imgs = channels2imgs(outs)
        #pert_rec = torch.from_numpy(crop_center2(root_sum_of_squares2(pert_imgs),320,320)).type(dtype)
        pert_rec = torch.from_numpy( data_consistency(pert_net, net_input, mask1d, slice_ksp_torchtensor1) ).type(dtype)
        pert_recs.append(pert_rec.data.cpu().numpy())
        if lr_decay_epoch is not 0:
            optimizer = exp_lr_scheduler(optimizer, i, init_lr=LR, lr_decay_epoch=lr_decay_epoch)

        def closure():
            
            optimizer.zero_grad()
            #out = net(inp.type(dtype))
            #out2 = net(inp2.type(dtype))
            
            #loss = mse(r, fixed_rec, pert_rec)
            loss = mse(r, fixed_outs, pert_outs)
            
            loss.backward(retain_graph=retain_graph)
            
            mse_[i] = loss.data.cpu().numpy()
            
            if i % 1 == 0:
                print ('\nIteration %05d   loss %f\n\n' % (i, mse_[i]))
            
            return loss   
        R.append(r.data.cpu())
        loss = optimizer.step(closure)
        with open("./outputs/untrainedrunner1/results","wb") as fn:
            pickle.dump([R,ksp,mse_,fixed_rec.data.cpu().numpy(),pert_recs],fn)
    return R,net_input, mse_, fixed_rec.data.cpu().numpy(), pert_recs
    

def fitr(net,
        img_noisy_var,
        img_clean_var,
        mask,
        net_input = None,
        num_iter = 5000,
        LR = 0.01,
        OPTIMIZER='adam',
        mask_var = None,
        apply_f = None,
        lr_decay_epoch = 0,
        lsimg = None,
        target_img = None,
        find_best=False,
        weight_decay=0,
        totalupsample = 1,
        loss_type="MSE",
        retain_graph = False,
        scale_out=1,
       ):
    import copy
    p = [x for x in net.parameters() ]

    mse_wrt_noisy = np.zeros(num_iter)
    mse_wrt_truth = np.zeros(num_iter)
    
    if OPTIMIZER == 'SGD':
        print("optimize with SGD", LR)
        optimizer1 = torch.optim.SGD(p, lr=LR,momentum=0.9,weight_decay=weight_decay)
    elif OPTIMIZER == 'adam':
        print("optimize with adam", LR)
        optimizer1 = torch.optim.Adam(p, lr=LR, weight_decay=weight_decay)
    elif OPTIMIZER == 'LBFGS':
        print("optimize with LBFGS", LR)
        optimizer1 = torch.optim.LBFGS(p, lr=LR)
    elif OPTIMIZER == "adagrad":
        print("optimize with adagrad", LR)
        optimizer1 = torch.optim.Adagrad(p, lr=LR,weight_decay=weight_decay)
    mse1 = torch.nn.MSELoss()
    
    if find_best:
        best_net = copy.deepcopy(net)
        best_mse = 1000000.0
    
    PSNRs = []
    SSIMs = []
    for i in range(num_iter):
         
        if lr_decay_epoch is not 0 and i % lr_decay_epoch == 0:
            optimizer1 = exp_lr_scheduler(optimizer, i, init_lr=LR, lr_decay_epoch=lr_decay_epoch)
        
        def closure():
            
            optimizer1.zero_grad()
            try:
                out = net(net_input.type(dtype),scale_out=scale_out)
            except:
                out = net(net_input.type(dtype))
                
            # training loss
            if mask_var is not None:
                loss = mse1( out * mask_var , img_noisy_var * mask_var )
            elif apply_f:
                loss = mse1( apply_f(out,mask) , img_noisy_var )
            else:
                loss = mse1(out, img_noisy_var)
        
            loss.backward(retain_graph=retain_graph)
            
            mse_wrt_noisy[i] = loss.data.cpu().numpy()
            # the actual loss 
            true_loss = mse1( Variable(out.data, requires_grad=False).type(dtype), img_clean_var.type(dtype) )
            mse_wrt_truth[i] = true_loss.data.cpu().numpy()
            
            if i % 100 == 0:
                if lsimg is not None:
                    ### compute ssim and psnr ###
                    out_chs = out.data.cpu().numpy()[0]
                    out_imgs = channels2imgs(out_chs)
                    # least squares reconstruciton
                    orig = crop_center2( root_sum_of_squares2(var_to_np(lsimg)) , 320,320)

                    # deep decoder reconstruction
                    rec = crop_center2(root_sum_of_squares2(out_imgs),320,320)

                    ssim_const = ssim(orig,rec,data_range=orig.max())
                    SSIMs.append(ssim_const)

                    psnr_const = psnr(orig,rec,np.max(orig))
                    PSNRs.append(psnr_const)
                    
                    ### ###
                
                trloss = loss.data
                true_loss = true_loss.data
                try:
                    out2 = net(Variable(net_input).type(dtype),scale_out=scale_out)
                except:
                    out2 = net(Variable(net_input).type(dtype))
                loss2 = mse1(out2, img_clean_var).data
                print ('Iteration %05d    Train loss %f  Actual loss %f Actual loss orig %f' % (i, trloss,true_loss,loss2), '\r', end='')
            return loss
        
        
        loss = optimizer1.step(closure)
        
        if find_best:
            # if training loss improves by at least one percent, we found a new best net
            lossval = loss.data
            if best_mse > 1.005*lossval:
                best_mse = lossval
                best_net = copy.deepcopy(net)
                net_input_saved = net_input.data.clone()
       
        
    if find_best:
        net = best_net
    
    return scale_out,SSIMs,PSNRs,mse_wrt_noisy, mse_wrt_truth,net_input, net      