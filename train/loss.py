#!/usr/local/bin/python

from __future__ import division
import torch
import random
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

class reconstruct_loss(nn.Module):
    """the loss between the input and synthesized input"""
    def __init__(self, cie_matrix, batchsize):
        super(reconstruct_loss, self).__init__()
        self.cie = Variable(torch.from_numpy(cie_matrix).float().cuda(), requires_grad=False)
        self.batchsize = batchsize
    def forward(self, network_input, network_output):
        network_output = network_output.permute(3, 2, 0, 1)
        network_output = network_output.contiguous().view(-1, 31)
        reconsturct_input = torch.mm(network_output,self.cie)
        reconsturct_input = reconsturct_input.view(50, 50, 64, 3)
        reconsturct_input = reconsturct_input.permute(2,3,1,0)
        reconstruction_loss = torch.mean(torch.abs(reconsturct_input - network_input))                  
        return reconstruction_loss

def rrmse_loss(outputs, label):
    """Computes the rrmse value"""
    error = torch.abs(outputs-label)/(label + 0.003)
    rrmse = torch.mean(error.view(-1))
    return rrmse

def L1_Loss(outputs, label):
    """Computes the rrmse value"""
    # error = torch.abs(outputs-label)/(label + 0.004)
    # rrmse = torch.mean(error.view(-1))
    print(label.size())
    print(outputs.size())
    Loss = torch.nn.MSELoss()
    # print(Loss)
    return Loss(outputs, label)

def Angle_Loss(outputs, label):
    # print(label.size())
    # print(outputs.size())
    normalize_out = F.normalize(outputs,p=2, dim=1)
    normalize_lab = F.normalize(label,p=2, dim=1)
    mult = normalize_out * normalize_lab
    # print("mult")
    # print(mult.size())
    # print(torch.sum(normalize_out * normalize_lab,dim=1 ).size())
    if(torch.isnan(outputs).any()):
        print("Fuckkkkk")
    # loss = torch.sum( torch.acos(torch.sum(normalize_out * normalize_lab,dim=1 ) ) )
    loss = torch.sum( 1.0 - (torch.sum(normalize_out * normalize_lab,dim=1 ) ) )
    # print(loss.size())
    # exit()
    return loss




def Divergence_Loss(outputs, label):
    # spectral information divergence loss
    # print(label.size())
    # print(outputs.size())
    # print(torch.sum( label ,dim = 1).size())
    # print(torch.sum(outputs,dim = 1).size()) 
    sum1 = torch.sum( label ,dim = 1)
    sum1 = torch.unsqueeze(torch.sum( label ,dim = 1), dim = 1)
    sum2 = torch.sum( label ,dim = 1)
    sum2 = torch.unsqueeze(torch.sum( outputs ,dim = 1), dim = 1)

    # if(torch.isnan(sum1).any()):
    #     print("Fuckkkkk")
    # print(sum1.size()) 
    t = torch.div( label, sum1.repeat(1,25,1,1) + 0.0123 )
    r = torch.div( outputs , sum2.repeat(1,25,1,1) + 0.0123 )

    if(torch.isnan(torch.log(torch.abs(t+0.0123))).any()):
        print("Fuckkkkk1")

    if(torch.isnan(torch.log(torch.abs(outputs))).any()):
        print("Fuckkkkk2")

    # if(torch.isnan(((torch.div(t+1000000,r+10000)))).any()):
    #     print("Fuckkkkk")
    # print(t.size())
    # print(r.size())
    # loss = torch.sum( torch.sum( t *torch.log(torch.abs(torch.div(t+0.0001,r+0.0001))) , dim = 1)
    #                         + torch.sum( r * torch.log(torch.abs(torch.div(r+0.0001,t+0.0001))) , dim = 1) )
    loss = torch.sum( torch.sum( r * (torch.log(torch.abs(r+0.0123)) - torch.log(torch.abs(t+0.0123)) )  , dim = 1) )

    return loss
