from __future__ import division

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable

import  os
import time
import scipy.io as sio

from dataset import DatasetFromHdf5
from resblock import resblock,conv_bn_relu_res_block
from utils import AverageMeter,initialize_logger,save_checkpoint_material,record_loss
from loss import rrmse_loss, L1_Loss, Angle_Loss, Divergence_Loss
from utils import save_matv73
from scipy.io import  loadmat
import numpy as np
import pytorch_msssim

def main():
    
    cudnn.benchmark = True
    # Dataset
    train_data = DatasetFromHdf5('./Data/train_Material_.h5')
    print(len(train_data))
    val_data = DatasetFromHdf5('./Data/valid_Material_.h5')
    print(len(val_data))

    # Data Loader (Input Pipeline)
    train_data_loader = DataLoader(dataset=train_data, 
                                   num_workers=1,  
                                   batch_size=64,
                                   shuffle=True,
                                   pin_memory=True)
    val_loader = DataLoader(dataset=val_data,
                            num_workers=1, 
                            batch_size=1,
                            shuffle=False,
                           pin_memory=True)

    # Model   

    model = resblock(conv_bn_relu_res_block,10,25,25)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    if torch.cuda.is_available():
        model.cuda()
  
    # Parameters, Loss and Optimizer
    start_epoch = 0
    end_epoch = 100
    init_lr = 0.0001
    iteration = 0
    record_test_loss = 1000
    # criterion_RRMSE = torch.nn.L1Loss()
    criterion_RRMSE = rrmse_loss
    criterion_Angle = Angle_Loss
    criterion_MSE = torch.nn.MSELoss()
    criterion_SSIM = pytorch_msssim.SSIM()
    # criterion_Div = Divergence_Loss
    criterion_Div = torch.nn.KLDivLoss()
    optimizer=torch.optim.Adam(model.parameters(), lr=init_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
    
    model_path = './models/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    loss_csv = open(os.path.join(model_path,'loss_material.csv'), 'w+')
    
    log_dir = os.path.join(model_path,'train_material.log')
    logger = initialize_logger(log_dir)
    
    # Resume
    resume_file = '' 
    if resume_file:
        if os.path.isfile(resume_file):
            print("=> loading checkpoint '{}'".format(resume_file))
            checkpoint = torch.load(resume_file)
            start_epoch = checkpoint['epoch']
            iteration = checkpoint['iter']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
       
    for epoch in range(start_epoch+1, end_epoch):
        
        start_time = time.time()         
        train_loss, iteration, lr = train(train_data_loader, model, criterion_MSE,criterion_RRMSE, criterion_Angle, criterion_SSIM, criterion_Div, optimizer, iteration, init_lr, end_epoch,epoch)
        test_loss, loss_angle, loss_reconstruct, loss_SSIM, loss_Div = validate(val_loader, model, criterion_MSE, criterion_RRMSE, criterion_Angle, criterion_SSIM, criterion_Div)
        
        # xxx_loss = validate_save(val_loader, model, criterion_MSE, criterion_RRMSE, epoch)
 
        

        save_checkpoint_material(model_path, epoch, iteration, model, optimizer)     

        # print loss 
        end_time = time.time()
        epoch_time = end_time - start_time
        print ("Epoch [%d], Iter[%d], Time:%.9f, learning rate : %.9f, Train Loss: %.9f Test Loss: %.9f , Angle Loss: %.9f, Recon Loss: %.9f, SSIM Loss: %.9f ,  Div Loss: %.9f" %(epoch, iteration, epoch_time, lr, train_loss, test_loss, loss_angle, loss_reconstruct, loss_SSIM, loss_Div))
        
        
        # save loss
        record_loss(loss_csv,epoch, iteration, epoch_time, lr, train_loss, test_loss)     
        logger.info("Epoch [%d], Iter[%d], Time:%.9f, learning rate : %.9f, Train Loss: %.9f Test Loss: %.9f, Angle Loss: %.9f, Recon Loss: %.9f, SSIM Loss: %.9f,  Div Loss: %.9f " %(epoch, iteration, epoch_time, lr, train_loss, test_loss, loss_angle, loss_reconstruct, loss_SSIM, loss_Div))
    
# Training 
def train(train_data_loader, model, criterion,criterion_RRMSE, criterion_Angle, criterion_SSIM, criterion_Div, optimizer, iteration, init_lr ,end_epoch,epoch):
    losses = AverageMeter()
    for i, (images, labels) in enumerate(train_data_loader):
        labels = labels.cuda(non_blocking =True)
        images = images.cuda(non_blocking =True)
        images = Variable(images)
        labels = Variable(labels)    
        lr = poly_lr_scheduler(optimizer, init_lr, iteration, max_iter=968000, power=1.5) 
        iteration = iteration + 1
        # Forward + Backward + Optimize       
        output = model(images)

        loss_MSE_1 = criterion(output, labels)

        loss_RRMSE_1 = criterion_RRMSE(output, labels)

        loss_angle = criterion_Angle(output, labels)* 0.01 / 5.0


        loss_SSIM = 1 - criterion_SSIM(output[:,0:3,:,:], labels[:,0:3,:,:]) 
        loss_SSIM = loss_SSIM + 1 - criterion_SSIM(output[:,3:6,:,:], labels[:,3:6,:,:]) 
        loss_SSIM = loss_SSIM + 1 - criterion_SSIM(output[:,6:9,:,:], labels[:,6:9,:,:]) 
        loss_SSIM = loss_SSIM + 1 - criterion_SSIM(output[:,9:12,:,:], labels[:,9:12,:,:]) 
        loss_SSIM = loss_SSIM + 1 - criterion_SSIM(output[:,12:15,:,:], labels[:,12:15,:,:]) 
        loss_SSIM = loss_SSIM + 1 - criterion_SSIM(output[:,15:18,:,:], labels[:,15:18,:,:]) 
        loss_SSIM = loss_SSIM + 1 - criterion_SSIM(output[:,18:21,:,:], labels[:,18:21,:,:]) 
        loss_SSIM = loss_SSIM + 1 - criterion_SSIM(output[:,21:24,:,:], labels[:,21:24,:,:]) 


        a = torch.log_softmax(output, dim=1)
        b = torch.softmax(labels, dim=1)

        a2 = torch.log_softmax(labels, dim=1)
        b2 = torch.softmax(output, dim=1)
        loss_Div = (criterion_Div(a, b) + criterion_Div(a2, b2))

        loss_reconstruct = loss_RRMSE_1
        loss = loss_reconstruct  + loss_angle + loss_Div + loss_SSIM * 0.01 
        

        optimizer.zero_grad()
        loss.backward()
        
        # Calling the step function on an Optimizer makes an update to its parameters
        optimizer.step()
        
        #  record loss
        losses.update(loss.item())

            
    return losses.avg, iteration, lr

# Validate
def validate(val_loader, model, criterion,criterion_RRMSE, criterion_Angle, criterion_SSIM, criterion_Div):
    
    
    model.eval()
    losses = AverageMeter()
    losses_angle = AverageMeter()
    losses_recons = AverageMeter()
    losses_SSIM = AverageMeter()
    losses_Div = AverageMeter()
    for i, (input, target) in enumerate(val_loader):
        input = input.cuda(non_blocking =True)
        target = target.cuda(non_blocking =True)
        with torch.no_grad():
          input_var = torch.autograd.Variable(input)
          target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)      
        loss_MSE_1 = criterion(output, target_var)


        loss_RRMSE_1 = criterion_RRMSE(output, target_var)

        loss_angle = criterion_Angle(output, target_var) * 0.01 / 5.0

        loss_SSIM = 1 - criterion_SSIM(output[:,0:3,:,:], target_var[:,0:3,:,:]) 
        loss_SSIM = loss_SSIM + 1 - criterion_SSIM(output[:,3:6,:,:], target_var[:,3:6,:,:]) 
        loss_SSIM = loss_SSIM + 1 - criterion_SSIM(output[:,6:9,:,:], target_var[:,6:9,:,:]) 
        loss_SSIM = loss_SSIM + 1 - criterion_SSIM(output[:,9:12,:,:], target_var[:,9:12,:,:]) 
        loss_SSIM = loss_SSIM + 1 - criterion_SSIM(output[:,12:15,:,:], target_var[:,12:15,:,:]) 
        loss_SSIM = loss_SSIM + 1 - criterion_SSIM(output[:,15:18,:,:], target_var[:,15:18,:,:]) 
        loss_SSIM = loss_SSIM + 1 - criterion_SSIM(output[:,18:21,:,:], target_var[:,18:21,:,:]) 
        loss_SSIM = loss_SSIM + 1 - criterion_SSIM(output[:,21:24,:,:], target_var[:,21:24,:,:]) 

        a = torch.log_softmax(target_var, dim=1)
        b = torch.softmax(output, dim=1)
        a2 = torch.log_softmax(output, dim=1)
        b2 = torch.softmax(target_var, dim=1)
        loss_Div =(criterion_Div(a, b) + criterion_Div(a2, b2) ) 
        
        
        loss_reconstruct =   loss_RRMSE_1
        loss = loss_reconstruct  + loss_angle + loss_Div + loss_SSIM * 0.01  
        
        losses.update(loss.item())
        losses_angle.update(loss_angle.item())
        losses_recons.update(loss_reconstruct.item())
        losses_Div.update(loss_Div.item())
        # losses_SSIM.update(loss_SSIM)

    return losses.avg, losses_angle.avg, losses_recons.avg, loss_SSIM, losses_Div.avg


def validate_save(val_loader, model, criterion, criterion_RRMSE,epoch):
    
    
    model.eval()
    losses = AverageMeter()

    for i, (input, target) in enumerate(val_loader):
        if (i % 10000 == 1):
            input = input.cuda(non_blocking =True)
            target = target.cuda(non_blocking =True)
            f = loadmat('./ValidObjectInput_4.mat')
            gt = f.get('CompData')
            # gt = gt.cuda(non_blocking =True)
            # input_var = gt
            gt = np.expand_dims(np.transpose(gt,[2,1,0]), axis=0).copy() 
            
            # gt = gt.astype(np.float64)/(4096-1)
            gt = torch.from_numpy(gt * 20.000 / 6.000).float()
            gt = gt.cuda(non_blocking =True)


            with torch.no_grad():
                input_var = torch.autograd.Variable(gt)
                target_var = torch.autograd.Variable(target)
            output = model(input_var)      
            loss = criterion(output, output)
            
            result = np.transpose(np.squeeze(output.cpu().detach().numpy()))
            save_matv73('./results/validobj' +str(epoch)+ '.mat', 'rad',result)
            losses.update(loss.item())


    for i, (input, target) in enumerate(val_loader):
        if (i % 10000 == 1):
            input = input.cuda(non_blocking =True)
            target = target.cuda(non_blocking =True)
            f = loadmat('./ValidDATInput_2')
            gt = f.get('CompData')
            gt = np.expand_dims(np.transpose(gt,[2,1,0]), axis=0).copy() 
            gt = torch.from_numpy(gt * 20.000 / 6.000 ).float()
            gt = gt.cuda(non_blocking =True)
            with torch.no_grad():
                input_var = torch.autograd.Variable(gt)
                target_var = torch.autograd.Variable(target)

            # compute output
            
            
            output = model(input_var[:,:,:,:])      
            loss = criterion(output, output)
            
            result = np.transpose(np.squeeze(output.cpu().detach().numpy()))
            save_matv73('./results/validapp' +str(epoch)+ '.mat', 'rad',result)
            #  record loss
            losses.update(loss.item())

    return losses.avg


# Learning rate
def poly_lr_scheduler(optimizer, init_lr, iteraion, lr_decay_iter=1,
                      max_iter=100, power=0.9):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power

    """
    if iteraion % lr_decay_iter or iteraion > max_iter:
        return optimizer

    lr = init_lr*(1 - iteraion/max_iter)**power
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

if __name__ == '__main__':
    main()



