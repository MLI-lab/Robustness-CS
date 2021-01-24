### fit an un-trained network with estimated sensitivity maps

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

dtype = torch.cuda.FloatTensor
#dtype = torch.FloatTensor
           
def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=500):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    
    lr = init_lr + 1e-4*epoch #(1.0001**epoch) #* (0.65**(epoch // lr_decay_epoch))
    """c = init_lr
    b = 0.1
    d = 200
    lr = c + b - (b/d)*abs(d - epoch%(2*d))
    
    if epoch % ((d//4)*lr_decay_epoch) == 0:
        print('LR is set to {}'.format(lr))"""

    for i,param_group in enumerate(optimizer.param_groups):
        """if i == 0:
            param_group['lr'] = lr * 5
        if i == 1:
            param_group['lr'] = lr * 8000
        if i == 2:
            param_group['lr'] = lr * 200
        if i == 3:
            param_group['lr'] = lr
        """
        param_group['lr'] = lr
        #print(param_group['lr'])
    return optimizer


'''
def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=500):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.65**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer
'''


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

class MSLELoss(torch.nn.Module):
    def __init__(self):
        super(MSLELoss,self).__init__()

    def forward(self,x,y):
        criterion = nn.MSELoss()
        loss = torch.log(criterion(x, y))
        return loss
def fits(net,
        img_noisy_var,
        num_channels,
        img_clean_var,
        slice_ksp_torchtensor,
        sens_maps,
        num_iter = 5000,
        LR = 0.01,
        OPTIMIZER='adam',
        devices=None,
        opt_input = False,
        reg_noise_std = 0,
        reg_noise_decayevery = 100000,
        mask = None,
        mask1d = None,
        mask_var = None,
        lr_decay_epoch = 0,
        net_input = None,
        net_input_gen = "random",
        lsimg = None,
        target_img = None,
        find_best=False,
        weight_decay=0,
        upsample_mode = "bilinear",
        totalupsample = 1,
        loss_type="MSE",
        output_gradients=False,
        output_weights=False,
        show_images=False,
        plot_after=None,
        in_size=None,
        scale_out=1,
       ):

    if net_input is not None:
        print("input provided")
    else:
        
        if upsample_mode=="bilinear":
            # feed uniform noise into the network 
            totalupsample = 2**len(num_channels)
            width = int(img_clean_var.data.shape[2]/totalupsample)
            height = int(img_clean_var.data.shape[3]/totalupsample)
        elif upsample_mode=="deconv":
            # feed uniform noise into the network 
            totalupsample = 2**(len(num_channels)-1)
            width = int(img_clean_var.data.shape[2]/totalupsample)
            height = int(img_clean_var.data.shape[3]/totalupsample)
        elif upsample_mode=="free":
            width,height = in_size
            
        
        shape = [1,num_channels[0], width, height]
        print("input shape: ", shape)
        net_input = Variable(torch.zeros(shape)).type(dtype).to(devices[0])
        net_input.data.uniform_()
        net_input.data *= 1./10
    
    net_input = net_input.type(dtype).to(devices[0])
    net_input_saved = net_input.data.clone()
    noise = net_input.data.clone()
    p = [x for x in net.parameters() ]

    if(opt_input == True): # optimizer over the input as well
        net_input.requires_grad = True
        p += [net_input]

    mse_wrt_noisy = np.zeros(num_iter)
    mse_wrt_truth = np.zeros(num_iter)
    
    
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

    if loss_type=="MSE":
        mse = torch.nn.MSELoss()
    if loss_type == "MSLE":
        mse = MSLELoss()
    if loss_type=="L1":
        mse = nn.L1Loss()
    
    if find_best:
        best_net = copy.deepcopy(net)
        best_mse = 1000000.0

    nconvnets = 0
    for p in list(filter(lambda p: len(p.data.shape)>2, net.parameters())):
        nconvnets += 1
    
    out_grads = np.zeros((nconvnets,num_iter))
        
    init_weights = get_weights(net)
    out_weights = np.zeros(( len(init_weights) ,num_iter))
    
    out_imgs = np.zeros((1,1))
    
    if plot_after is not None:
        try:
            out_img_np = net( net_input_saved.type(dtype).to(devices[0]),scale_out=scale_out ).data.cpu().numpy()[0]
        except:
            out_img_np = net( net_input_saved.type(dtype).to(devices[0]) ).data.cpu().numpy()[0]
        out_imgs = np.zeros( (len(plot_after),) + out_img_np.shape )
    
    PSNRs = []
    SSIMs = []
    norm_ratio = []
    S = transform.to_tensor(sens_maps).type(dtype).to(devices[0])
    for i in range(num_iter):
        
        def closure():
            
            optimizer.zero_grad()
            try:
                out = net(net_input.type(dtype).to(devices[0]),scale_out=scale_out)
            except:
                out = net(net_input.type(dtype).to(devices[0]))
            
            ### apply coil sensitivity maps and forward model
            imgs = torch.zeros(S.shape).type(dtype).to(devices[0])
            for j,s in enumerate(S):
                imgs[j,:,:,0] = out[0,0,:,:] * s[:,:,0] - out[0,1,:,:] * s[:,:,1]
                imgs[j,:,:,1] = out[0,0,:,:] * s[:,:,1] + out[0,1,:,:] * s[:,:,0]
            Fimg = transform.fft2(imgs[None,:])
            Fimg,_ = transform.apply_mask(Fimg, mask = mask)
            #print(imgs.shape,mask.shape)
            
            # training loss
            loss = mse(Fimg,img_noisy_var)
            loss.backward()
            
            mse_wrt_noisy[i] = loss.data.cpu().numpy()

            # the actual loss 
            true_loss = mse( Variable(out.data, requires_grad=False).type(dtype).to(devices[0]), img_clean_var.type(dtype).to(devices[0]) )
            mse_wrt_truth[i] = true_loss.data.cpu().numpy()
            
            if output_gradients:
                for ind,p in enumerate(list(filter(lambda p: p.grad is not None and len(p.data.shape)>2, net.parameters()))):
                    out_grads[ind,i] = p.grad.data.norm(2).item()
                    #print(p.grad.data.norm(2).item())
                    #su += p.grad.data.norm(2).item()
                    #mse_wrt_noisy[i] = su
            
            if i % 100 == 0:
                #if lsimg is not None:
                    #norm_ratio.append( np.linalg.norm(root_sum_of_squares2(out_imgs)) / np.linalg.norm(root_sum_of_squares2(var_to_np(lsimg))) )
                    ### ###
                
                trloss = loss.data
                true_loss = true_loss.data
                try:
                    out2 = net(Variable(net_input_saved).type(dtype).to(devices[0]),scale_out=scale_out)
                except:
                    out2 = net(Variable(net_input_saved).type(dtype).to(devices[0]))
                loss2 = mse(out2, img_clean_var).data
                print ('Iteration %05d    Train loss %f  Actual loss %f Actual loss orig %f' % (i, trloss,true_loss,loss2), '\r', end='')
            
            if show_images:
                if i % 50 == 0:
                    print(i)
                    try:
                        out_img_np = net( ni.type(dtype).to(devices[0]),scale_out=scale_out ).data.cpu().numpy()[0]
                    except:
                        out_img_np = net( ni.type(dtype).to(devices[0]) ).data.cpu().numpy()[0]
                    myimgshow(plt,out_img_np)
                    plt.show()
                    
            if plot_after is not None:
                if i in plot_after:
                    try:
                        out_imgs[ plot_after.index(i) ,:] = net( net_input_saved.type(dtype),scale_out=scale_out ).data.cpu().numpy()[0]
                    except:
                        out_imgs[ plot_after.index(i) ,:] = net( net_input_saved.type(dtype),scale_out=scale_out ).data.cpu().numpy()[0]
            if output_weights:
                out_weights[:,i] = np.array( get_distances( init_weights, get_weights(net) ) )
            
            return loss   
        
        loss = optimizer.step(closure)
            
        if find_best:
            # if training loss improves by at least one percent, we found a new best net
            lossval = loss.data
            if best_mse > 1.005*lossval:
                best_mse = lossval
                best_net = copy.deepcopy(net)
                if opt_input:
                    best_ni = net_input.data.clone()
                else:
                    best_ni = net_input_saved.clone()
            if i % 100 == 0:
                if lsimg is not None:
                    orig = lsimg.copy()
                    ### data consistency ###
                    out = best_net(net_input.type(dtype).to(devices[0]))
                    img = out.clone()
                    s = img.shape
                    ns = int(s[1]/2) # number of slices
                    fimg = Variable( torch.zeros( (s[0],ns,s[2],s[3],2 ) ) ).type(dtype)
                    for a in range(ns):
                        fimg[0,a,:,:,0] = img[0,2*a,:,:]
                        fimg[0,a,:,:,1] = img[0,2*a+1,:,:]
                    Fimg = transform.fft2(fimg) # dim: (1,num_slices,x,y,2)
                    # ksp has dim: (num_slices,x,y)
                    meas = slice_ksp_torchtensor.unsqueeze(0) # dim: (1,num_slices,x,y,2)
                    maskk = torch.from_numpy(np.array(mask1d, dtype=np.uint8))
                    ksp_dc = Fimg.clone()
                    ksp_dc = ksp_dc.detach().cpu()
                    ksp_dc[:,:,:,maskk==1,:] = meas[:,:,:,maskk==1,:] # after data consistency block

                    img_dc = transform.ifft2(ksp_dc)[0]
                    outt = []
                    for img in img_dc.detach().cpu():
                        outt += [ img[:,:,0].numpy() , img[:,:,1].numpy() ]

                    par_out_chs = np.array(outt)
                    par_out_imgs = channels2imgs(par_out_chs)

                    # deep decoder reconstruction
                    recc = root_sum_of_squares2(par_out_imgs)
                    orig = crop_center2(orig,min(orig.shape[1],recc.shape[1],320),min(orig.shape[0],recc.shape[0],320))
                    rec = crop_center2(recc,min(orig.shape[1],recc.shape[1],320),min(orig.shape[0],recc.shape[0],320))
                    
                    ### compute ssim and psnr ###
                    #out_chs = out.data.cpu().numpy()[0]
                    #out_imgs = channels2imgs(out_chs)
                    # least squares reconstruciton
                     #crop_center2( root_sum_of_squares2(var_to_np(lsimg)) , 320,320)

                    # deep decoder reconstruction
                    #rec = crop_center2(root_sum_of_squares2(out_imgs),320,320)
                    rec = (rec-rec.mean()) / rec.std()
                    rec *= orig.std()
                    rec += orig.mean()
                    
                    ssim_const = ssim(np.array([orig]), np.array([rec]))
                    psnr_const = psnr(np.array([orig]), np.array([rec]))
                    SSIMs.append(ssim_const)
                    PSNRs.append(psnr_const)
                    #print(ssim_const,orig.mean(),rec.mean(),rec.shape, orig.shape)
       
        
    if find_best:
        net = best_net
        net_input_saved = best_ni
    if output_gradients and output_weights:
        return scale_out,SSIMs,PSNRs,norm_ratio,mse_wrt_noisy, mse_wrt_truth,net_input_saved, net, out_grads
    elif output_gradients:
        return scale_out,SSIMs,PSNRs,norm_ratio,mse_wrt_noisy, mse_wrt_truth,net_input_saved, net, out_grads      
    elif output_weights:
        return scale_out,SSIMs,PSNRs,norm_ratio,mse_wrt_noisy, mse_wrt_truth,net_input_saved, net, out_weights
    elif plot_after is not None:
        return scale_out,SSIMs,PSNRs,norm_ratio,mse_wrt_noisy, mse_wrt_truth,net_input_saved, net, out_imgs
    else:
        return scale_out,SSIMs,PSNRs,norm_ratio,mse_wrt_noisy, mse_wrt_truth,net_input_saved, net       
        

