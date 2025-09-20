import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from tsf.models.Gaussianblur import gaussian_pooling
from numpy.ma.core import squeeze

lambda_per = 1
lambda_ave = 0.5
def prm_losses(result, target):

        # we expect input and target to be in [0, 1] range
        predicted = torch.round(result)
        print(torch.max(result),torch.min(result))
        emphysema = (target==3)
        fsad = (target==2)
        normal = (target==1)
        predicted_emphysema = (predicted == 3)
        predicted_fsad = (predicted == 2)
        predicted_normal = (predicted == 1)
        print('emphysema: ', torch.sum(emphysema),'predicted emphysema: ', torch.sum(predicted_emphysema))
        print('fSAD: ', torch.sum(fsad),'predicted fSAD: ', torch.sum(predicted_fsad))

        print('normal: ', torch.sum(normal),'predicted normal: ', torch.sum(predicted_normal))

        pooled_predicted_emph = gaussian_pooling(torch.squeeze(predicted_emphysema.float(),dim=1))
        pooled_target_emph = gaussian_pooling(torch.squeeze(emphysema.float(),dim=1))
        pooled_predicted_fsad = gaussian_pooling(torch.squeeze(predicted_fsad.float(),dim=1))
        pooled_target_fsad = gaussian_pooling(torch.squeeze(fsad.float(),dim=1))
        pooled_predicted_normal = gaussian_pooling(torch.squeeze(predicted_normal.float(),dim=1))
        pooled_target_normal = gaussian_pooling(torch.squeeze(normal.float(),dim=1))

        vsum = torch.sum(emphysema)+torch.sum(fsad)+torch.sum(normal)
        losses_predicted = torch.abs(torch.abs(torch.sum(emphysema)-torch.sum(predicted_emphysema))/vsum) + torch.abs(torch.abs(torch.sum(fsad)-torch.sum(predicted_fsad))/vsum)+ torch.abs(torch.abs(torch.sum(normal)-torch.sum(predicted_normal)) /vsum)
        losses_predicted = losses_predicted / 3
        losses_ave= nn.L1Loss()(pooled_predicted_emph, pooled_target_emph)+ nn.L1Loss()(pooled_predicted_fsad, pooled_target_fsad)+ nn.L1Loss()(pooled_predicted_normal, pooled_target_normal)
        losses = losses_predicted*lambda_per #+ losses_ave*lambda_ave
        print('losses_predicted: ', losses_predicted,'losses_ave: ', losses_ave)
        print('prm_losses: ', losses)
        return losses
