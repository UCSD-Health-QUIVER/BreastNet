# -*- encoding: utf-8 -*-
import torch.nn as nn
import torch



def gradient3d(inputTensor: torch.Tensor) -> torch.Tensor:
    """
    This function computes Sobel edge maps on 3D images
    inputTensor: input 3D images, with size of [batchsize,W,H,D,1]
    output: output 3D edge maps, with size of [batchsize,W,H,D,3], each channel represents edge map in one dimension 
    
    """   


    epsilon = 1.e-10 # added epsilon to prevent nan error
    device = inputTensor.device
    
    # Sobel filter in x dimension
    x_filter = torch.tensor([[[-1,0,1],
                        [-2,0,2],
                        [-1,0,1]],
                        
                        [[-2,0,2],
                        [-4,0,4],
                        [-2,0,2]],
                        
                        [[-1,0,1],
                        [-2,0,2],
                        [-1,0,1]]],
                                dtype = torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    # Sobel filter in y dimension
    y_filter = torch.tensor([[[1,2,1],
                                [0,0,0],
                                [-1,-2,-1]],
                                
                                [[2,4,2],
                                [0,0,0],
                                [-2,-4,-2]],
                                
                                [[1,2,1],
                                [0,0,0],
                                [-1,-2,-1]]],
                                dtype = torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    # Sobel filter in z dimension
    z_filter  = torch.tensor([[[1,2,1],
                        [2,4,2],
                        [1,2,1]],
                        
                        [[0,0,0],
                        [0,0,0],
                        [0,0,0]],
                        
                        [[-1,-2,-1],
                        [-2,-4,-2],
                        [-1,-2,-1]]],
                                dtype = torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    # Convolution for each dimension
    outputTensor_x = torch.nn.functional.conv3d(inputTensor, x_filter, stride=1, padding=1)    
    outputTensor_y = torch.nn.functional.conv3d(inputTensor, y_filter, stride=1, padding=1)   
    outputTensor_z = torch.nn.functional.conv3d(inputTensor, z_filter, stride=1, padding=1)   

    xyz_grad = torch.concat([outputTensor_x,outputTensor_y,outputTensor_z],1)
    return torch.sqrt(torch.sum(torch.square(xyz_grad), keepdim=True, dim=1) + epsilon) # added epsilon to prevent nan error

def MGE(dose:torch.Tensor, pred:torch.Tensor, reduction:str='mean')->torch.Tensor:
    """
    Computes the mean gradient error between the ground truth dose and the predicted dose
    dose: ground truth dose, with size of [batchsize,W,H,D]
    pred: predicted dose, with size of [batchsize,W,H,D]
    reduction: reduction method, 'mean' or 'sum' or 'none'
    returns: mean gradient error
    """
    grad = gradient3d(dose)
    pred_grad = gradient3d(pred)

    if reduction == 'mean':
        return torch.mean(torch.square(grad - pred_grad))
    elif reduction == 'sum':
        return torch.sum(torch.square(grad - pred_grad))
    elif reduction == 'none':
        return torch.square(grad - pred_grad)



class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.L1_loss_func = nn.L1Loss(reduction='none')
        self.smoothL1 = nn.SmoothL1Loss(reduction='mean', beta=1.0)
        self.MSELoss_func_no_reduce = nn.MSELoss(reduction='none')
        self.MSELoss_func = nn.MSELoss()

    def forward(self, pred, gt_dose, body_mask):
        pred_A = pred
        gt_dose = gt_dose.to(pred_A.device)


        MSE_loss = self.MSELoss_func_no_reduce(pred_A, gt_dose)
        
        #body_masked MSE
        MSE_loss = MSE_loss * body_mask

        dose_gradient_loss = MGE(gt_dose,pred_A, reduction="none") # (batch, 3, 128,128,128)
        dose_gradient_loss = dose_gradient_loss * body_mask
        weighted_loss = torch.mean(MSE_loss + torch.tanh(dose_gradient_loss)*MSE_loss)

        return weighted_loss



