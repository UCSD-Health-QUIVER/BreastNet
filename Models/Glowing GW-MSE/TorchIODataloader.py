# # -*- encoding: utf-8 -*-
from ast import Raise
from pickletools import uint8
import torch.utils.data as data
import numpy as np
import random
import torch
import torchio as tio
import os
# import SimpleITK as sitk

def get_subject_list(patients_dir):

    patient_list = []
    for patient_folder in os.listdir(patients_dir):
        full_dir = os.path.join(patients_dir,patient_folder)
        list_structures = ['CT',
                    'heart_glowing',
                    'lung_glowing',
                    'tumor_glowing',
                    'dose',
                    'body'
                    ]

        subject_dict = {}

        for structure_name in list_structures:
            if os.path.exists(os.path.join(full_dir,structure_name+".nii")):
                if structure_name == "CT" or structure_name == 'dose' or structure_name == 'heart_glowing' or structure_name == 'lung_glowing' or structure_name == 'tumor_glowing':
                    
                    # load to RAM by explicitly converting to tensor
                    # image = sitk.ReadImage(os.path.join(full_dir,structure_name+".nii.gz"))
                    # array = sitk.GetArrayFromImage(image)
                    # tensor = torch.from_numpy(np.expand_dims(array,0))
                    # subject_dict[structure_name]=tio.ScalarImage(tensor = tensor)

                    # load from hard disk using lazy loading (generally preferred)                 
                    subject_dict[structure_name]=tio.ScalarImage(os.path.join(full_dir,structure_name+".nii"))


                else:
                    #load to RAM
                    # image = sitk.ReadImage(os.path.join(full_dir,structure_name+".nii.gz"))
                    # array = sitk.GetArrayFromImage(image)
                    # tensor = torch.from_numpy(np.expand_dims(array,0))
                    # subject_dict[structure_name]=tio.LabelMap(tensor = tensor)

                    # from hard disk
                    subject_dict[structure_name]=tio.LabelMap(os.path.join(full_dir,structure_name+".nii"))
            

                        
            else:
                raise Exception("Error in data loading structure name: {}, for {}".format(structure_name, patient_folder))



            
        subject_dict["patient_id"] = patient_folder
        subject = tio.Subject(subject_dict)
        patient_list.append(subject)
    return patient_list

def cutout3D_block(volume):
    """
    Randomly cuts out a block of size 32,48,64 from the volume
    """

    size = np.random.choice([32,48,64])
    x_max = volume.shape[-3]
    y_max = volume.shape[-2]
    z_max = volume.shape[-1]

    x = np.random.choice(np.arange(x_max))
    y = np.random.choice(np.arange(y_max))
    z = np.random.choice(np.arange(z_max))
 
    volume[0,max(x-size,0):min(x+size,x_max), max(y-size,0):min(y+size,y_max),max(z-size,0):min(z+size,z_max)] = 0
    return volume

def cutout3D_many(volume):
    """
    Randomly cuts out 10 blocks of size 4,8,16 from the volume
    """
    x_max = volume.shape[-3]
    y_max = volume.shape[-2]
    z_max = volume.shape[-1]
    for i in range(10):
        size = np.random.choice([4,8,16])
        x = np.random.choice(np.arange(x_max))
        y = np.random.choice(np.arange(y_max))
        z = np.random.choice(np.arange(z_max))
    
        volume[0,max(x-size,0):min(x+size,x_max), max(y-size,0):min(y+size,y_max),max(z-size,0):min(z+size,z_max)] = 0
    return volume

def drop_channel(volume):
    """
    Channel dropout with probability 0.25.
    Could probably achive the same thing by
    adding a dropout layer to the model.
    """
    if random.random() < 0.25:
        volume[0,...] = 0
    return volume
def drop_channel_hard(volume):
    """
    Same as drop_channel but with probability 0.75.
    Could probably just make the probability a parameter
    """
    if random.random() < 0.75:
        volume[:] = 0
    return volume




def get_loader(train_bs=1, val_bs=1, train_num_samples_per_epoch=1, val_num_samples_per_epoch=1, num_works=0, slurm_dir=None, training=True):

    if slurm_dir is not None: # used when running on slurm, slurm_dir is the node local scratch dir
        data_dir_train = os.path.join(slurm_dir,"Train")
        data_dir_val = os.path.join(slurm_dir,"Validation")
    else: 
        data_dir_train = r'{your data dir here}'
        data_dir_val = r'{your data dir here}'   

    train_subjects = get_subject_list(data_dir_train)# *10 if your training set is small
    val_subjects = get_subject_list(data_dir_val)


    if training:
        transforms = tio.Compose([
        tio.transforms.Clamp(p=1.0, include="CT", out_min=-1000, out_max=1000),
        tio.transforms.RescaleIntensity(p=1.0, include="CT", in_min_max=(-1000, 1000),  out_min_max = (0,1)), # https://arxiv.org/pdf/1809.10486.pdf
        tio.RandomFlip(p=0.9,axes=(0,1,2)),
        tio.RandomAffine(p=0.9,scales=0.,degrees=(60),translation=10, isotropic=True, default_pad_value='minimum')
        ])
    else:
        transforms = tio.Compose([
        tio.transforms.Clamp(p=1.0, include="CT", out_min=-1000, out_max=1000),
        tio.transforms.RescaleIntensity(p=1.0, include="CT", in_min_max=(-1000, 1000), out_min_max = (0,1)), # https://arxiv.org/pdf/1809.10486.pdf
        ])

    
    val_transforms = tio.Compose([
    tio.transforms.Clamp(p=1.0, include="CT", out_min=-1000, out_max=1000),
    tio.transforms.RescaleIntensity(p=1.0, include="CT", in_min_max=(-1000, 1000), out_min_max = (0,1)), # https://arxiv.org/pdf/1809.10486.pdf
    ])

    train_dataset = tio.SubjectsDataset(train_subjects, transform=transforms)
    val_dataset = tio.SubjectsDataset(val_subjects, transform=val_transforms)



    train_loader = data.DataLoader(dataset=train_dataset, batch_size=train_bs, shuffle=True, num_workers=num_works)
                                   #pin_memory=False,collate_fn=tio.utils.history_collate) fix for memory crash issues 
    val_loader = data.DataLoader(dataset=val_dataset, batch_size=val_bs, shuffle=False, num_workers=num_works)
                                 #pin_memory=False,collate_fn=tio.utils.history_collate)


    return train_loader, val_loader 
