# -*- encoding: utf-8 -*-
from genericpath import exists
import os
import argparse
from model import *
from network_trainer import *
import numpy as np
import SimpleITK as sitk
import TorchIODataloader as evalloader
import matplotlib.pyplot as plt





def plotMSE_tot_volume(dose_arrays:torch.Tensor, dict_volumes:dict, predicted_dose_array:torch.Tensor, plots_dir:str = None) -> None:
    """
    Plots the MSE of the total volume of the predicted dose and the true dose for each patient
    dose_arrays: dictionary of dose arrays for each patient
    dict_volumes: dictionary of volume arrays for each patient
    predicted_dose_array: predicted dose array for each patient
    plots_dir: directory to save plots

    """

    for name, volume_arrays in dict_volumes["test"].items():
        labels = []
        plot_vals = []
        for ttv in ['test','train','val']:
            volume = dict_volumes[ttv][name]
            volume_dose_true = torch.where(volume == 1, dose_arrays[ttv], torch.zeros_like(dose_arrays[ttv]) )
            volume_dose_pred = torch.where(volume == 1, predicted_dose_array[ttv], torch.zeros_like(predicted_dose_array[ttv]) )
            volume_dose_true_count = torch.count_nonzero(volume_dose_true > 0, dim=(1,2,3))
            volume_dose_pred_count = torch.count_nonzero(volume_dose_pred > 0, dim=(1,2,3))
            volume_dose_true_sum = torch.sum(volume_dose_true, dim=(1,2,3))
            volume_dose_pred_sum = torch.sum(volume_dose_pred, dim=(1,2,3))
            volume_dose_true_mean = volume_dose_true_sum/volume_dose_true_count
            volume_dose_pred_mean = volume_dose_pred_sum/volume_dose_pred_count
            plot_vals.append(volume_dose_true_mean)
            labels.append(ttv+"_Dose")
            plot_vals.append(volume_dose_pred_mean)
            labels.append(ttv+"_Pred")
        plt.boxplot(plot_vals, labels=labels)
        plt.title("Average Dose " + name)
        plt.grid()
        plt.savefig(os.path.join(plots_dir, "AvgDose_"+name+".png"))
        plt.clf()
        plt.close()
        
def plotMAE_tot(dose_arrays:torch.Tensor, dict_volumes:dict, predicted_dose_array:torch.Tensor, plots_dir:str = None) -> None:
    """
    Plots the MAE of the total volume of the predicted dose and the true dose for each patient
    Also plots the MAE of the total volume of the predicted dose and the true dose for each patient at a threshold value
    dose_arrays: dictionary of dose arrays for each patient
    dict_volumes: dictionary of volume arrays for each patient
    predicted_dose_array: predicted dose array for each patient
    plots_dir: directory to save plots   
    """

    color_list = ['r','b','c','m','y','k','w']
    color_map = dict(zip(dict_volumes.keys(),color_list))
    labels = []
    labels_threshold = []
    plot_vals = []
    plot_vals_threshold = []
    threshold_value = 0.1
    for ttv in ['test','train','val']:

        abs_error = torch.abs(dose_arrays[ttv] - predicted_dose_array[ttv])

        mean_error_pat = torch.mean(abs_error, dim=(1,2,3))
        plot_vals.append(mean_error_pat)
        labels.append(ttv+"_MAE")

        SE_threshold = torch.where(dose_arrays[ttv] > threshold_value, abs_error, torch.zeros_like(abs_error))
        mean_error_threshold = torch.sum(SE_threshold, dim=(1,2,3))/torch.count_nonzero(dose_arrays[ttv] > threshold_value, dim=(1,2,3))
        plot_vals_threshold.append(mean_error_threshold)
        labels_threshold.append(ttv + "_MAE_threshold")
    plt.boxplot(plot_vals, labels=labels)
    plt.title("MAE")
    plt.grid()
    plt.savefig(os.path.join(plots_dir, "MAE_plots.png"))
    plt.clf()
    plt.close()

    plt.boxplot(plot_vals_threshold, labels=labels_threshold)
    plt.title("MAE_threshold at " + str(threshold_value))
    plt.grid()
    plt.savefig(os.path.join(plots_dir, "MAE_threshold_plots.png"))
    plt.clf()
    plt.close()
    
  


def plotDVH_LCM(dose_array:torch.Tensor, dict_volumes:dict, predicted_dose_array:torch.Tensor = None, plots_dir:str = None, fold:str = None)-> None:
        """
        Plots the combined average DVH for the the dataset
        dose_array: dose array for each patient
        dict_volumes: dictionary of volume arrays for each patient
        predicted_dose_array: predicted dose array for each patient
        plots_dir: directory to save plots
        fold: name of the fold (train, val, test)

        """
        color_list = ['r','b','c','m','y','k','w']
        color_map = dict(zip(dict_volumes.keys(),color_list))


        for name,volume in dict_volumes.items():
            volume_bin = []
            

            masked_dose = dose_array[volume == 1]
            volume_voxels_count = np.count_nonzero(volume[volume == 1])
            x_axis = np.linspace(-0.1,1.2,500)
            for i in x_axis:
                bin_height = np.count_nonzero(masked_dose > i)/volume_voxels_count
                
                volume_bin.append(bin_height)
            plt.plot(x_axis,volume_bin, label=name, color=color_map[name])
        
        if predicted_dose_array is not None:
            for name,volume in dict_volumes.items():
                volume_bin = []
                

                masked_dose = predicted_dose_array[volume == 1]
                volume_voxels_count = np.count_nonzero(volume[volume == 1])

                x_axis = np.linspace(-0.1,1.2,500)
                for i in x_axis:
                    bin_height = np.count_nonzero(masked_dose > i)/volume_voxels_count
                    
                    volume_bin.append(bin_height)
                plt.plot(x_axis,volume_bin, label="predicted_"+name, linestyle='--', color=color_map[name])
        plt.legend()
        plt.grid()
        plt.xlabel("Dose (% Rx)")
        plt.ylabel("Volume (%)")
        title =  "DVH_" + fold if fold else "DVH" 
        plt.title(title)
        # plt.show()
        plt.savefig(os.path.join(plots_dir,title+".png"))
        plt.clf()
        plt.close()



if __name__ == "__main__":
    print("running")

    parser = argparse.ArgumentParser()
    parser.add_argument('--GPU_id', type=int, default=[0],
                        help='GPU id used for testing (default: 0)')
    parser.add_argument('--model_path', type=str, default= os.path.join(os.path.dirname(os.path.realpath(__file__)),os.path.join('Output','best_val_evaluation_index.pkl')))
    parser.add_argument('--TTA', type=bool, default=False,
                        help='do test-time augmentation, default True')
    parser.add_argument('--slurm_dir', type=str,default='/scratch/$USER/job_$SLURM_JOBID')   
    args = parser.parse_args()
    slurm_dir = args.slurm_dir
    
    if os.name == 'nt':
        slurm_dir = None
            
    model_path = args.model_path

    ### model path is directory to checkpoint file
    trainer = NetworkTrainer()
    trainer.setting.project_name = 'Alpha'


    trainer.setting.output_dir = os.path.join(os.path.dirname(model_path),'Predictions')



    if not os.path.exists(trainer.setting.output_dir):
        os.mkdir(trainer.setting.output_dir)

    trainer.setting.network = Model(in_ch=4, out_ch=1,

                                    list_ch_A=[-1, 24, 48, 96, 192, 384, 768],
    
                                    dropout_prob=0.3
                                    ).float()

    trainer.init_trainer(ckpt_file=model_path,
                list_GPU_ids=[0],
                only_network=True)
    train, val = evalloader.get_loader(slurm_dir=slurm_dir)
    test = evalloader.get_test_loader(slurm_dir=slurm_dir)
    combined_dose = None
    combined_pred = None
    count = 0
    if not os.path.exists(os.path.join(os.path.dirname(model_path),'Plots')):
        os.makedirs(os.path.join(os.path.dirname(model_path),'Plots'))
    plots_dir = os.path.join(os.path.dirname(model_path),'Plots')

    TTA = False

    fold = "Test"

    if not os.path.exists(os.path.join(trainer.setting.output_dir,fold)):
        os.mkdir(os.path.join(trainer.setting.output_dir,fold))
    val_dir = os.path.join(trainer.setting.output_dir,fold)
    list_loaders = []
    dose_arrays, dict_volumes, predicted_dose_arrays = {}, {}, {}
    for batch_idx, list_loader_output in enumerate(test):
     
 
        direction = (0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0)
        with torch.no_grad():
            trainer.setting.network.eval()
            input_, gt_dose= trainer.prepare_batch(list_loader_output)
            
            if not os.path.exists(os.path.join(val_dir,list_loader_output['patient_id'][0])):
                os.mkdir(os.path.join(val_dir,list_loader_output['patient_id'][0]))
            # else:
            #     continue
            full_pat_dir = os.path.join(val_dir,list_loader_output['patient_id'][0]) 
            

            prediction_B = trainer.setting.network(input_.to(device=trainer.setting.device))

            prediction_B = prediction_B.cpu().data[:, :, :, :, :]
        

            prediction_B = prediction_B.squeeze()

            body = list_loader_output["body"][tio.DATA].squeeze()


            prediction_B = torch.where(body != 0,prediction_B,torch.zeros_like(prediction_B))


            list_loader_output["pred"] = prediction_B
            prediction_B = torch.swapaxes(prediction_B, 0, 2)


            gt_dose = gt_dose.squeeze()
            gt_dose = torch.swapaxes(gt_dose, 0, 2)

            x_nii = sitk.GetImageFromArray(prediction_B.squeeze())
            # x_nii.SetDirection(direction)
            sitk.WriteImage(x_nii, os.path.join(full_pat_dir, "prediction.nii.gz"))

            
            y_nii = sitk.GetImageFromArray(gt_dose.squeeze())

            # y_nii.SetDirection(direction)
            sitk.WriteImage(y_nii, os.path.join(full_pat_dir, "true_dose.nii.gz"))
            
            
            diff = gt_dose.squeeze() - prediction_B.squeeze()

            diff = sitk.GetImageFromArray(diff)
            # diff.SetDirection(direction)
            sitk.WriteImage(diff, os.path.join(full_pat_dir, "Actual_Minus_Pred.nii.gz"))
         
        list_loaders.append(list_loader_output)

    preds = torch.cat([x["pred"].unsqueeze(0) for x in list_loaders])
    doses = torch.cat([x["dose"][tio.DATA].squeeze().unsqueeze(0) for x in list_loaders])
    hearts = torch.cat([x["heart_glowing"][tio.DATA].squeeze().unsqueeze(0) for x in list_loaders])
    lungs = torch.cat([x["lung_glowing"][tio.DATA].squeeze().unsqueeze(0) for x in list_loaders])
    tumor_beds = torch.cat([x["tumor_glowing"][tio.DATA].squeeze().unsqueeze(0) for x in list_loaders])
    CTs = torch.cat([x["CT"][tio.DATA].squeeze().unsqueeze(0) for x in list_loaders])
    dvs = {"heart": hearts, "lung": lungs, "tumor_bed": tumor_beds}
    patient_ids = [x["patient_id"][0] for x in list_loaders]
    print("Test")

    plotDVH_LCM(doses, dvs, preds, plots_dir=plots_dir, fold=fold)
    
    dose_arrays["test"] = doses
    predicted_dose_arrays["test"] = preds
    dict_volumes["test"] = dvs

    preds = None
    doses = None
    hearts = None
    lungs = None
    tumor_beds = None
    CTs = None
    dvs = None
    list_loaders = []
 


    fold = "Validation"
    if not os.path.exists(os.path.join(trainer.setting.output_dir,fold)):
        os.mkdir(os.path.join(trainer.setting.output_dir,fold))
    val_dir = os.path.join(trainer.setting.output_dir,fold)
    for batch_idx, list_loader_output in enumerate(val):

        with torch.no_grad():
            trainer.setting.network.eval()
            input_, gt_dose= trainer.prepare_batch(list_loader_output)
            
            if not os.path.exists(os.path.join(val_dir,list_loader_output['patient_id'][0])):
                os.mkdir(os.path.join(val_dir,list_loader_output['patient_id'][0]))
            # else:
            #     continue
            full_pat_dir = os.path.join(val_dir,list_loader_output['patient_id'][0]) 
            

            prediction_B = trainer.setting.network(input_.to(device=trainer.setting.device))

            prediction_B = prediction_B.cpu().data[:, :, :, :, :]
            prediction_B = prediction_B.squeeze()

            body = list_loader_output["body"][tio.DATA].squeeze()


            prediction_B = torch.where(body != 0,prediction_B,torch.zeros_like(prediction_B))

            list_loader_output["pred"] = prediction_B.squeeze()
            prediction_B = torch.swapaxes(prediction_B, 0, 2)
            gt_dose = gt_dose.squeeze()
            gt_dose = torch.swapaxes(gt_dose, 0, 2)

            x_nii = sitk.GetImageFromArray(prediction_B.squeeze())
            # x_nii.SetDirection(direction)
            sitk.WriteImage(x_nii, os.path.join(full_pat_dir, "prediction.nii.gz"))

            
            y_nii = sitk.GetImageFromArray(gt_dose.squeeze())
            # y_nii.SetDirection(direction)
            sitk.WriteImage(y_nii, os.path.join(full_pat_dir, "true_dose.nii.gz"))
            
            
            diff = gt_dose.squeeze() - prediction_B.squeeze()
            diff = sitk.GetImageFromArray(diff)
            # diff.SetDirection(direction)
            sitk.WriteImage(diff, os.path.join(full_pat_dir, "Actual_Minus_Pred.nii.gz"))
        list_loaders.append(list_loader_output)

    preds = torch.cat([x["pred"].unsqueeze(0) for x in list_loaders])
    doses = torch.cat([x["dose"][tio.DATA].squeeze().unsqueeze(0) for x in list_loaders])
    hearts = torch.cat([x["heart_glowing"][tio.DATA].squeeze().unsqueeze(0) for x in list_loaders])
    lungs = torch.cat([x["lung_glowing"][tio.DATA].squeeze().unsqueeze(0) for x in list_loaders])
    tumor_beds = torch.cat([x["tumor_glowing"][tio.DATA].squeeze().unsqueeze(0) for x in list_loaders])
    CTs = torch.cat([x["CT"][tio.DATA].squeeze().unsqueeze(0) for x in list_loaders])
    dvs = {"heart": hearts, "lung": lungs, "tumor_bed": tumor_beds}
    patient_ids = [x["patient_id"][0] for x in list_loaders]
    print("Val")


    plotDVH_LCM(doses, dvs, preds, plots_dir=plots_dir, fold=fold)
    
    dose_arrays["val"] = doses
    predicted_dose_arrays["val"] = preds
    dict_volumes["val"] = dvs

    preds = None
    doses = None
    hearts = None
    lungs = None
    tumor_beds = None
    CTs = None
    dvs = None
    list_loaders = []


    fold = "Train"
    if not os.path.exists(os.path.join(trainer.setting.output_dir,fold)):
        os.mkdir(os.path.join(trainer.setting.output_dir,fold))
    val_dir = os.path.join(trainer.setting.output_dir,fold)
    for batch_idx, list_loader_output in enumerate(train):
   

        with torch.no_grad():
            trainer.setting.network.eval()
            input_, gt_dose= trainer.prepare_batch(list_loader_output)
            
            if not os.path.exists(os.path.join(val_dir,list_loader_output['patient_id'][0])):
                os.mkdir(os.path.join(val_dir,list_loader_output['patient_id'][0]))
            # else:
            #     continue
            full_pat_dir = os.path.join(val_dir,list_loader_output['patient_id'][0]) 
          
            
            try:
                prediction_B = trainer.setting.network(input_.to(device=trainer.setting.device))
            except Exception as e:
                print(e)
                print(list_loader_output['patient_id'])
                continue

            prediction_B = prediction_B.cpu().data[:, :, :, :, :]
            prediction_B = prediction_B.squeeze()

            body = list_loader_output["body"][tio.DATA].squeeze()


            prediction_B = torch.where(body != 0,prediction_B,torch.zeros_like(prediction_B))
            list_loader_output["pred"] = prediction_B.squeeze()
            prediction_B = torch.swapaxes(prediction_B, 0, 2)
            gt_dose = gt_dose.squeeze()
            gt_dose = torch.swapaxes(gt_dose, 0, 2)
            x_nii = sitk.GetImageFromArray(prediction_B.squeeze())
            # x_nii.SetDirection(direction)
            sitk.WriteImage(x_nii, os.path.join(full_pat_dir, "prediction.nii.gz"))

            
            y_nii = sitk.GetImageFromArray(gt_dose.squeeze())
            # y_nii.SetDirection(direction)
            sitk.WriteImage(y_nii, os.path.join(full_pat_dir, "true_dose.nii.gz"))
            
            
            diff = gt_dose.squeeze() - prediction_B.squeeze()

            diff = sitk.GetImageFromArray(diff)
            # diff.SetDirection(direction)
            sitk.WriteImage(diff, os.path.join(full_pat_dir, "Actual_Minus_Pred.nii.gz"))
        list_loaders.append(list_loader_output)

    preds = torch.cat([x["pred"].unsqueeze(0) for x in list_loaders])
    doses = torch.cat([x["dose"][tio.DATA].squeeze().unsqueeze(0) for x in list_loaders])
    hearts = torch.cat([x["heart_glowing"][tio.DATA].squeeze().unsqueeze(0) for x in list_loaders])
    lungs = torch.cat([x["lung_glowing"][tio.DATA].squeeze().unsqueeze(0) for x in list_loaders])
    tumor_beds = torch.cat([x["tumor_glowing"][tio.DATA].squeeze().unsqueeze(0) for x in list_loaders])
    CTs = torch.cat([x["CT"][tio.DATA].squeeze().unsqueeze(0) for x in list_loaders])
    dvs = {"heart": hearts, "lung": lungs, "tumor_bed": tumor_beds}
    patient_ids = [x["patient_id"] for x in list_loaders]
    print("Train")

    plotDVH_LCM(doses, dvs, preds, plots_dir=plots_dir, fold=fold)
    
    dose_arrays["train"] = doses
    predicted_dose_arrays["train"] = preds
    dict_volumes["train"] = dvs

    preds = None
    doses = None
    hearts = None
    lungs = None
    tumor_beds = None
    CTs = None
    dvs = None
    list_loaders = []





    plotMSE_tot_volume(dose_arrays, dict_volumes, predicted_dose_arrays, plots_dir = plots_dir)
    plotMAE_tot(dose_arrays, dict_volumes, predicted_dose_arrays, plots_dir = plots_dir)  


