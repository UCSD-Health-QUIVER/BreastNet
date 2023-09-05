import os
import SimpleITK as sitk
import numpy as np




def recast(folder:str)->None:
    """
    This function recasts the dtype of the images in the folder to float32.
    The images masks are originally generated using a uint8 dtype, which is not
    compatible with the model/torchIO transformations. This function is used to recast the images to float32.
    Future models should use sitk.Cast(image, sitk.sitkFloat32) to cast the images
    to float32 during the dataloader, likely faster and more memory efficient
    """
    if folder is None:
        folder = r'{your default folder here}'
    for train_val in os.listdir(folder): # folder is slurm_dir train/val/test

        out_directory_train_val = os.path.join(folder,train_val)# train/val/test
        for patient_folder in os.listdir(out_directory_train_val):
            patFolder = os.path.join(out_directory_train_val, patient_folder)
            for file in os.listdir(patFolder):

                if file not in ["body.nii"]:
                    continue
                image = sitk.ReadImage(os.path.join(patFolder,file))
                array = sitk.GetArrayFromImage(image).astype(float)
                image_out = sitk.GetImageFromArray(array)
                image_out.SetDirection(image.GetDirection())
                image_out.SetSpacing(image.GetSpacing())
                image_out.SetOrigin(image.GetOrigin())
                sitk.WriteImage(image_out,os.path.join(patFolder,file))

if __name__ == "__main__":    

    recast(None)