from __future__ import division
import torch
import torch.nn as nn

import os
import time

from scipy.io import  loadmat
import numpy as np
from imageio import imread
import PIL.Image
from resblock import resblock,conv_bn_relu_res_block
from utils import save_matv73,reconstruction,load_mat,mrae,rmse, Angle_Loss

#98
model_path = './models/HS_Fruit_Model.pkl'
img_path = './Data/Fruit/test_inputs/'
result_path = './Data/Fruit/test_results/'
if not os.path.exists(result_path):
        os.makedirs(result_path)
gt_path = './Data/Fruit/test_labels/'


var_name = 'CompData'

save_point = torch.load(model_path)
model_param = save_point['state_dict']
model = resblock(conv_bn_relu_res_block,10,25,25)
model.load_state_dict(model_param)

model = model.cuda()
model.eval()




criterion_Angle = Angle_Loss
# criterion_MSE = torch.nn.MSELoss()
# criterion_SSIM = pytorch_msssim.SSIM()

Loss_SID = 0
ANG_loss = 0

Total_MRAE = 0
Total_RRMSE = 0

images_number = 0
for img_name in sorted(os.listdir(img_path)):
    images_number = images_number +1
    img_path_name = os.path.join(img_path, img_name)


    f = loadmat(img_path_name)
    inputt = f.get('CompData')
    inputt = np.expand_dims(np.transpose(inputt,[2,1,0]), axis=0).copy() 
    inputt = torch.from_numpy(inputt /6.0000).float()
    inputt = inputt.cuda(non_blocking =True) 
    with torch.no_grad():
        input_var = torch.autograd.Variable(inputt)
    output = model(input_var)


    mat_name = img_name
    mat_dir= os.path.join(result_path, mat_name)
    
    gt_name =  img_name
    gt_dir= os.path.join(gt_path, gt_name)

    f2 = loadmat(gt_dir)
    gt = f2.get('CompData')


    gt = np.expand_dims(np.transpose(gt,[2,1,0]), axis=0).copy() 
    gt = torch.from_numpy(gt/6.000).float()
    gt = gt.cuda(non_blocking =True)

    criterion_Div = torch.nn.KLDivLoss()
    a = torch.log_softmax(output*6.00, dim=1)
    b = torch.softmax(gt*6.00, dim=1)

    a2 = torch.log_softmax(gt*6.00, dim=1)
    b2 = torch.softmax(output *6.00, dim=1)

    loss_Div = (criterion_Div(a, b) + criterion_Div(a2, b2))

    ANG_loss = ANG_loss + criterion_Angle(output*6.00, gt*6.00).item()/(512*512)
    
    output = np.transpose(np.squeeze(output.cpu().detach().numpy()))
    inputt = np.transpose(np.squeeze(inputt.cpu().detach().numpy()))
    gt = np.transpose(np.squeeze(gt.cpu().detach().numpy()))
    save_matv73(mat_dir, 'rad',output)



    Loss_SID = Loss_SID + loss_Div.item()
    mrae_error =  mrae(output*6.00, gt*6.00)
    # rrmse_error = rmse(inputt, gt)

    Total_MRAE = Total_MRAE + mrae_error
    # Total_RRMSE = Total_RRMSE + rrmse_error

    print("[%s]" %(img_name))
    # print(Total_MRAE)
    

print("#############################################")
print(images_number)
print("Average SAM Loss is:")
print(ANG_loss/images_number)

print("Average MRAE Loss is:")
print(Total_MRAE/images_number)

print("Average SID Loss is:")
print(Loss_SID/images_number)

print("#############################################")




    
