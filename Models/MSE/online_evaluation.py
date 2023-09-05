# -*- encoding: utf-8 -*-

# from evaluate_openKBP import *
import torch
import os
import numpy as np
import torch.nn as nn
import pandas as pd
import custom_loss
import torchio as tio
from custom_loss import MGE, gradient3d
def get_3D_Dose_dif(pred, gt, possible_dose_mask=None):
    if possible_dose_mask is not None:
        pred = pred[possible_dose_mask > 0]
        gt = gt[possible_dose_mask > 0]

    dif = np.mean(np.abs(pred - gt))
    return dif

def online_evaluation(trainer):
    
    list_Dose_score = []
    for batch_idx, list_loader_output in enumerate(trainer.setting.val_loader):
        with torch.no_grad():
            trainer.setting.network.eval()
            input_, gt_dose= trainer.prepare_batch(list_loader_output)
            input_ = input_.to(trainer.setting.device)
            prediction_B = trainer.setting.network(input_)
            pred_B = prediction_B.cpu().data[:, :, :, :, :]
            gt_dose = gt_dose

            MSE_no_reduction = nn.MSELoss(reduction='none')
            body_mask = list_loader_output["body"][tio.DATA].float()
            MSE_loss = MSE_no_reduction(pred_B,gt_dose)
            MSE_loss = MSE_loss*body_mask

            weighted_loss = torch.mean(MSE_loss)
            Dose_score = weighted_loss

            list_Dose_score.append(Dose_score.item())

            try:
                trainer.print_log_to_file('========> ' + str(list_loader_output['patient_id'][0]) + ':  ' + str(Dose_score.item()), 'a')
                
            except Exception as e:
                with open(trainer.setting.output_dir + '/errors.txt', 'a', encoding="utf-8") as file:
                    file.write(str(e))

    try:
        trainer.print_log_to_file('===============================================> Average MSE Dose Val (percent Rx): '
                                  + str(np.mean(list_Dose_score)), 'a')                              
    except Exception as e:
        with open(trainer.setting.output_dir + '/errors.txt', 'a', encoding="utf-8") as file:
            file.write(str(e))
    
    
    
    trainer.log.val_curve.append(np.mean(list_Dose_score))
    
    trainer.log.train_curve.append(trainer.log.average_train_loss)

    csv_dir = trainer.setting.output_dir
    df = pd.DataFrame({"train":trainer.log.train_curve, "val":trainer.log.val_curve})

    df.to_csv(os.path.join(csv_dir,"training_curves.csv"))
    

    return np.mean(list_Dose_score)

