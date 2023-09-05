from DicomRTTool.ReaderWriter import DicomReaderWriter
import SimpleITK as sitk
import os
import numpy as np
import torchio as tio
import pydicom as dcm
from scipy.ndimage import center_of_mass
from makeGlowingMasks import makeGlowingMask
import pandas as pd
import shutil
from glob import glob
from ARTemisMetrics import *
import traceback


def find_box(cube:np.ndarray) -> tuple:
    """
    Finds the bounding box of a cube of data.
    This centers the data around the breast.
    :param cube: 3D numpy array
    :return: tuple of voxel indicies
    """

    cube = np.where(cube > 5, cube, 0)
    cube_dim = cube.shape

    non_zero_indx = np.nonzero(cube)

    buffer = 5
    min1 = np.min(non_zero_indx[1]) - buffer
    if min1 + 128 > cube_dim[1]:
        min1 = cube_dim[1] - 128
    max2 = np.max(non_zero_indx[2]) + buffer
    if max2 - 128 < 0:
        max2 = 128


    return (max(0,min1),max(128,max2))





def makeDataset(in_dir:str, out_dir:str, key_dir:str = None) -> None:
    """
    This function takes in a directory of DICOM files and outputs a directory of nifti files.
    This function is designed to work with the anonymization key.
    :param in_dir: directory of DICOM files
    :param out_dir: directory to output nifti files
    :param key_dir: directory of anonymization key
    :return: None
    """
    if key_dir is not None:
        translation_key = pd.read_excel(key_dir, dtype=str)
    errors = [
            # used for troubleshooting individual patients 
            ]
    errors = [str(i) for i in errors]

    Contour_names = ['heart', 'lung - lt','tumor bed','body', "tumorbed","tumor bed 1", "lung - lt", "l lung", "lt lung", "lung- lt", "l tumor_bed","lumpectomy","dibh tb", "gtv","lung rt", "tumor bedsd"]
    nifti_path_top = out_dir

    dict_error = {"Error_Pats":[]}
    for patient_folder in os.listdir(in_dir):
        # trouble shoot individual patients
        # if patient_folder not in errors:
        #     continue

        brute_force = False
        artemis = None

 

        try:
            anon_pat = translation_key[translation_key["ID"].str.match(patient_folder)]["Anon_ID"].values[0]
            Dicom_path = os.path.join(in_dir,patient_folder)
            nifti_path = os.path.join(nifti_path_top,anon_pat)
            if not os.path.exists(nifti_path):
                os.makedirs(nifti_path)
            Dicom_reader = DicomReaderWriter(description='Examples', arg_max=False, get_dose_output=True, Contour_Names=Contour_names)

            Dicom_reader.walk_through_folders(Dicom_path) # This will parse through all DICOM present in the folder and subfolders
       
            all_rois = Dicom_reader.return_rois(print_rois=True) # Return a list of all rois present
            
       
            if True: # switched to always brute force during development
                brute_force = True
   
                artemis = make_session(Dicom_path, Contour_names)
                # plansums have multiple plans in them, they have to be loaded and ...summed 
                plan_name = list(artemis.grouped_dicom.keys())[0]
                second_plan_name = list(artemis.grouped_dicom.keys())[1] if len(artemis.grouped_dicom.keys()) > 1 else None
                third_plan_name = list(artemis.grouped_dicom.keys())[2] if len(artemis.grouped_dicom.keys()) > 2 else None
                # if there are more than three something is probably not right
                if third_plan_name is not None:
                    print("Warning: more than 2 plans present in patient {}".format(patient_folder))
        


            else:
                brute_force = True


            Transformation_masks = tio.Resample((2.,2.,2.), image_interpolation='nearest') # Define the transformation you want to apply to the image
            Transformation_Scalar = tio.Resample((2.,2.,2.), image_interpolation='linear') # Define the transformation you want to apply to the image


            Dicom_reader.get_images()

            if brute_force:
        

                dose = artemis.grouped_dicom[plan_name]['dose_image']
          
                origin = dose.GetOrigin()
                origin = np.round(np.array(origin),1)
                dose.SetOrigin(origin)

                second_origin = artemis.grouped_dicom[second_plan_name]['dose_image'].GetOrigin() if second_plan_name is not None else None
                second_origin = np.round(np.array(second_origin),1) if second_origin is not None else None
                artemis.grouped_dicom[second_plan_name]['dose_image'].SetOrigin(second_origin) if second_plan_name is not None else None
                dose += artemis.grouped_dicom[second_plan_name]['dose_image'] if second_plan_name is not None else 0
                dose += artemis.grouped_dicom[third_plan_name]['dose_image'] if third_plan_name is not None else 0

            else:
                dose = Dicom_reader.dose_handle

         
            dose_resampled = Transformation_Scalar(dose)
            dose_resampled_origin = dose_resampled.GetOrigin()
            dose_resampled_spacing = dose_resampled.GetSpacing()
            dose_resampled_direction = dose_resampled.GetDirection()

            

            dose_resample_array = sitk.GetArrayFromImage(dose_resampled)
            dose_resample_array = np.pad(dose_resample_array, ((0,100),(0,100),(0,100)), 'constant', constant_values=(0,0))
            dose_resampled = sitk.GetImageFromArray(dose_resample_array)
            dose_resampled.SetOrigin(dose_resampled_origin)
            dose_resampled.SetSpacing(dose_resampled_spacing)
            dose_resampled.SetDirection(dose_resampled_direction)


            assert dose_resample_array.shape[0] >=128, "Dose array {} is too small ({}, instead of >128)".format("zero", dose_resample_array.shape[0])
            assert dose_resample_array.shape[1] >=128, "Dose array {} is too small ({}, instead of >128)".format("one", dose_resample_array.shape[1])
            assert dose_resample_array.shape[2] >=128, "Dose array {} is too small ({}, instead of >128)".format("two", dose_resample_array.shape[2])

            comx,comy,comz = center_of_mass(np.where(dose_resample_array > 5, dose_resample_array, 0))
            comx = int(comx)

            if comx + 64 > dose_resample_array.shape[0]: ###hit top border
                top = dose_resample_array.shape[0]
                bot = top - 128


            elif comx < 64:
                bot = 0
                top = 128
            else:
                top = comx + 64
                bot = comx - 64
                
            dose_img_small = dose_resampled[:,:,bot:top]
            dose_array = sitk.GetArrayFromImage(dose_img_small)
            max_indicies = find_box(dose_array)

            dose_cropped = dose_img_small[max_indicies[1] - 128:max_indicies[1],max_indicies[0]:max_indicies[0]+128,:]
            assert dose_cropped.GetSize() == (128,128,128)
            sitk.WriteImage(dose_cropped, os.path.join(nifti_path, 'dose.nii'))


            dicom_sitk_handle = Dicom_reader.dicom_handle
        


            CT_image = Transformation_Scalar(dicom_sitk_handle)

            # write out the CT image
            sitk.WriteImage(CT_image, os.path.join(nifti_path, 'CT.nii'))

            diff_vector = np.array(dose_cropped.GetOrigin()) - np.array(CT_image.GetOrigin()) 

            diff_vector_scaled = np.array(diff_vector)/np.array(dose_cropped.GetSpacing())
            diff_vector_scaled = np.round(diff_vector_scaled).astype(int)
    
            CT_array = sitk.GetArrayFromImage(CT_image)
            CT_array = np.pad(CT_array, ((0,128),(0,128),(0,128)), 'constant', constant_values = 0)
            CT_image_padded = sitk.GetImageFromArray(CT_array)
            CT_image_padded.SetOrigin(CT_image.GetOrigin())
            CT_image_padded.SetSpacing(CT_image.GetSpacing())
            CT_image = CT_image_padded


            CT_image_cropped = CT_image[diff_vector_scaled[0]:diff_vector_scaled[0]+128,diff_vector_scaled[1]:diff_vector_scaled[1]+128,diff_vector_scaled[2]:diff_vector_scaled[2]+128]
            sitk.WriteImage(CT_image_cropped, os.path.join(nifti_path, 'CT.nii'))


            if brute_force:
                keys = artemis.grouped_dicom[plan_name].keys()
                for key in keys:
                    if "lung" in key and "image" in key:
                        lung_key = key
                    if ("tumor" in key and "image" in key) or ("lump" in key and "image" in key) or ("dibh tb" in key and "image" in key) or ("gtv" in key and "image" in key):
                        tumor_key = key
                        
                heart_mask = artemis.grouped_dicom[plan_name]['heart_mask_image']
                lung_mask = artemis.grouped_dicom[plan_name][lung_key]
                tumor_mask = artemis.grouped_dicom[plan_name][tumor_key]
                body_mask = artemis.grouped_dicom[plan_name]['body_mask_image']
  

                heart_mask = Transformation_masks(heart_mask)
                lung_mask = Transformation_masks(lung_mask)
                tumor_mask = Transformation_masks(tumor_mask)
                body_mask = Transformation_masks(body_mask)

                heart_array = sitk.GetArrayFromImage(heart_mask)
                lung_array = sitk.GetArrayFromImage(lung_mask)
                tumor_array = sitk.GetArrayFromImage(tumor_mask)
                body_array = sitk.GetArrayFromImage(body_mask)

                heart_array = np.pad(heart_array, ((0,128),(0,128),(0,128)), 'constant', constant_values = 0)
                lung_array = np.pad(lung_array, ((0,128),(0,128),(0,128)), 'constant', constant_values = 0)
                tumor_array = np.pad(tumor_array, ((0,128),(0,128),(0,128)), 'constant', constant_values = 0)
                body_array = np.pad(body_array, ((0,128),(0,128),(0,128)), 'constant', constant_values = 0)

                heart_mask_padded = sitk.GetImageFromArray(heart_array)
                lung_mask_padded = sitk.GetImageFromArray(lung_array)
                tumor_mask_padded = sitk.GetImageFromArray(tumor_array)
                body_mask_padded = sitk.GetImageFromArray(body_array)

                heart_mask_padded.SetOrigin(heart_mask.GetOrigin())
                lung_mask_padded.SetOrigin(lung_mask.GetOrigin())
                tumor_mask_padded.SetOrigin(tumor_mask.GetOrigin())
                body_mask_padded.SetOrigin(body_mask.GetOrigin())

                heart_mask_padded.SetSpacing(heart_mask.GetSpacing())
                lung_mask_padded.SetSpacing(lung_mask.GetSpacing())
                tumor_mask_padded.SetSpacing(tumor_mask.GetSpacing())
                body_mask_padded.SetSpacing(body_mask.GetSpacing())

                heart_mask = heart_mask_padded
                lung_mask = lung_mask_padded
                tumor_mask = tumor_mask_padded
                body_mask = body_mask_padded


                heart_mask_cropped = heart_mask[max_indicies[1] - 128:max_indicies[1],max_indicies[0]:max_indicies[0]+128,bot:top]
                lung_mask_cropped = lung_mask[max_indicies[1] - 128:max_indicies[1],max_indicies[0]:max_indicies[0]+128,bot:top]
                tumor_mask_cropped = tumor_mask[max_indicies[1] - 128:max_indicies[1],max_indicies[0]:max_indicies[0]+128,bot:top]
                body_mask_cropped = body_mask[max_indicies[1] - 128:max_indicies[1],max_indicies[0]:max_indicies[0]+128,bot:top]

                sitk.WriteImage(heart_mask_cropped, os.path.join(nifti_path, 'heart.nii'))
                sitk.WriteImage(lung_mask_cropped, os.path.join(nifti_path, 'lung.nii'))
                sitk.WriteImage(tumor_mask_cropped, os.path.join(nifti_path, 'tumor.nii'))
                sitk.WriteImage(body_mask_cropped, os.path.join(nifti_path, 'body.nii'))

                ### use body mask to cancel out the bed in the CT images
                # CT_image_array = sitk.GetArrayFromImage(CT_image_cropped)
                # body_mask_array = sitk.GetArrayFromImage(body_mask_cropped)
                # CT_image_array = CT_image_array*body_mask_array
                # CT_image_cropped2 = sitk.GetImageFromArray(CT_image_array)
                # CT_image_cropped2.SetOrigin(CT_image_cropped.GetOrigin())
                # CT_image_cropped2.SetSpacing(CT_image_cropped.GetSpacing())
                # CT_image_cropped = CT_image_cropped2
                # sitk.WriteImage(CT_image_cropped, os.path.join(nifti_path, 'CT.nii'))
                # doesn't work very well...

                heart_mask_glowing = makeGlowingMask(heart_mask_cropped, mesa = True )
                lung_mask_glowing = makeGlowingMask(lung_mask_cropped, mesa = True)
                tumor_mask_glowing = makeGlowingMask(tumor_mask_cropped, mesa = True)

                sitk.WriteImage(heart_mask_glowing, os.path.join(nifti_path, 'heart_glowing.nii'))
                sitk.WriteImage(lung_mask_glowing, os.path.join(nifti_path, 'lung_glowing.nii'))
                sitk.WriteImage(tumor_mask_glowing, os.path.join(nifti_path, 'tumor_glowing.nii'))


            else:    
                for key, value in Dicom_reader.mask_dictionary.items():

                    Dicom_reader.mask_dictionary[key] = Transformation_masks(Dicom_reader.mask_dictionary[key])
        

                    cropped_image = Dicom_reader.mask_dictionary[key][diff_vector_scaled[0]:diff_vector_scaled[0]+128,diff_vector_scaled[1]:diff_vector_scaled[1]+128,diff_vector_scaled[2]:diff_vector_scaled[2]+128]




                    if "lung" in key.lower():
                        key = "lung"
                    elif "heart" in key.lower():
                        key = "heart"
                    elif "tumor" in key.lower():
                        key = "tumor"
                    sitk.WriteImage(cropped_image, os.path.join(nifti_path, '{}.nii'.format(key))   )
                    cropped_imge_glowing = makeGlowingMask(cropped_image)
                    sitk.WriteImage(cropped_imge_glowing, os.path.join(nifti_path, '{}_glowing.nii'.format(key))   )
        except Exception as e:
            print(e)
            traceback.print_exc()
            print("Error in: ", patient_folder)
            dict_error["Error_Pats"].append(patient_folder)
            # delete the folder
            shutil.rmtree(nifti_path)    
            continue
    df = pd.DataFrame.from_dict(dict_error)
    df.to_csv(os.path.join(nifti_path_top, 'Error_Pats.csv'))





nifti_path = r'{output directory for the nifti images}'
Dicom_path = r'{the input dicom directory}'

anon_key = r'{path to the anonymization key}'
makeDataset(Dicom_path, nifti_path, anon_key)