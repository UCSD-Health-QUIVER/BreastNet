# -*- encoding: utf-8 -*-
import os
import argparse
import torch
from TorchIODataloader import get_loader
from network_trainer import NetworkTrainer
from model import Model
from online_evaluation import online_evaluation
from custom_loss import Loss
from torchinfo import summary

if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=24,
                        help='batch size for training (default: 2)')
    parser.add_argument('--list_GPU_ids', nargs='+', type=int, default=[3,2,1, 0], 
                        help='list_GPU_ids for training (default: [1, 0])')
    parser.add_argument('--max_iter',  type=int, default=1000000,
                        help='training iterations(default: 100000)')
    parser.add_argument('--slurm_dir', type=str,default='/scratch/$USER/job_$SLURM_JOBID')         
                  

    args = parser.parse_args()


    #  Start training
    trainer = NetworkTrainer()
    trainer.setting.project_name = 'GW_MSE_'
    duplicate_count = 0
    while os.path.isdir(os.path.join(os.path.dirname(os.path.realpath(__file__)),'Output_' + trainer.setting.project_name + str(duplicate_count))):
      duplicate_count += 1
      
    outpath = os.path.join(os.path.dirname(os.path.realpath(__file__)),'Output_'+ trainer.setting.project_name + str(duplicate_count))
    trainer.setting.output_dir = outpath
    trainer.setting.save_per_epoch = 50
    slurm_dir = args.slurm_dir
    list_GPU_ids = args.list_GPU_ids
    if os.name == "nt":
        slurm_dir = None

    ### recast the byte masks if needed
    # recast(slurm_dir)

    trainer.setting.network = Model(in_ch=4, out_ch=1,

                                    list_ch_A=[-1, 24, 48, 96, 192, 384, 768],

                                    dropout_prob=0.3
                                    ).float()

    trainer.setting.max_iter = args.max_iter

    trainer.setting.train_loader, trainer.setting.val_loader = get_loader(
        train_bs=args.batch_size,
        val_bs=1,
        train_num_samples_per_epoch=args.batch_size * 500,  # 500 iterations per epoch
        val_num_samples_per_epoch=1,
        num_works=6,
        slurm_dir = slurm_dir,
        training=True
    )

    trainer.log.iters = len(trainer.setting.train_loader)



    trainer.setting.eps_train_loss = 0.01
    trainer.setting.lr_scheduler_update_on_iter = False
    trainer.setting.loss_function = Loss()

    trainer.setting.online_evaluation_function_val = online_evaluation

    trainer.set_optimizer(optimizer_type='AdamW',
                          args={
                              'lr': 1e-3,
                              'weight_decay': 1e-4
                          }
                          )

    trainer.set_lr_scheduler(lr_scheduler_type='ReduceLROnPlateau',
                            args={
                                'factor': 0.1,
                                'patience': 50,
                                'threshold': 1e-8
                                
                            }
                            )
    lr_scheduler_args = "None"

    if not os.path.exists(trainer.setting.output_dir):
        os.mkdir(trainer.setting.output_dir)
    
    with open(trainer.setting.output_dir + '/ModelStats.txt', 'a', encoding="utf-8") as file:
        file.write("-------------------------Model Summary----------------------------")
        file.write(str(summary(trainer.setting.network,input_size=(args.batch_size//len(list_GPU_ids),4,128,128,128), verbose=0)))
        file.write("\n")
        file.write(str(trainer.setting.optimizer))
        file.write("\n")
        file.write(str(trainer.setting.lr_scheduler_type))
        file.write("\n")
        file.write(str(lr_scheduler_args))
        file.write("\n")
        file.write("-------------------- args --------------------------")
        file.write("\n")
        file.write(str(args))
    trainer.set_GPU_device(list_GPU_ids)
    trainer.run()

    trainer.print_log_to_file('# Done !\n', 'a')
