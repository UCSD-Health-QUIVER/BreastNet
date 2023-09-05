# -*- encoding: utf-8 -*-
from tabnanny import verbose
import time
import torchio as tio
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import os


class TrainerSetting:
    def __init__(self):
        self.project_name = None
        self.output_dir = None
        self.max_iter = 99999999
        self.max_epoch = 99999999
        self.save_per_epoch = 99999999
        self.eps_train_loss = 0.01
        self.network = None
        self.device = None
        self.list_GPU_ids = None
        self.train_loader = None
        self.val_loader = None
        self.train_no_trans_loader = None
        self.optimizer = None
        self.lr_scheduler = None
        self.lr_scheduler_type = None
        self.lr_scheduler_update_on_iter = False
        self.loss_function = None
        self.online_evaluation_function_val = None


class TrainerLog:
    def __init__(self):
        self.iter = -1
        self.epoch = -1
        self.last_update_epoch = -1
        self.epoch_iter = -1
        self.iters = -1
        self.lr_history = []
        self.moving_train_loss = None
        self.average_train_loss = 99999999.
        self.best_average_train_loss = 99999999.
        self.average_val_index = 99999999.
        self.best_average_val_index = 99999999.
        self.train_curve = []
        self.val_curve = []
        self.train_no_trans_curve = []
        self.list_average_train_loss_associate_iter = []
        self.list_average_val_index_associate_iter = []
        self.list_lr_associate_iter = []
        self.save_status = []


class TrainerTime:
    def __init__(self):
        self.train_time_per_epoch = 0.
        self.train_loader_time_per_epoch = 0.
        self.val_time_per_epoch = 0.
        self.val_loader_time_per_epoch = 0.


class NetworkTrainer:
    def __init__(self):
        self.log = TrainerLog()
        self.setting = TrainerSetting()
        self.time = TrainerTime()

    def set_GPU_device(self, list_GPU_ids):
        self.setting.list_GPU_ids = list_GPU_ids
        sum_device = len(list_GPU_ids)
        # cpu only
        if list_GPU_ids[0] == -1:
            self.setting.device = torch.device('cpu')
        # single GPU
        elif sum_device == 1:
            self.setting.device = torch.device('cuda:' + str(list_GPU_ids[0]))
        # multi-GPU
        else:
            self.setting.device = torch.device('cuda:' + str(list_GPU_ids[0]))
            self.setting.network = nn.DataParallel(self.setting.network, device_ids=list_GPU_ids)
        self.setting.network.to(self.setting.device)

    def set_optimizer(self, optimizer_type, args):
        if optimizer_type == 'AdamW':
                self.setting.optimizer = optim.AdamW([
                    {'params': self.setting.network.parameters(), 'lr': args['lr']},
  
                ],
                    weight_decay=args['weight_decay'],
                    amsgrad=False)
        
    def set_lr_scheduler(self, lr_scheduler_type, args):
        if lr_scheduler_type == 'step':
            self.setting.lr_scheduler_type = 'step'
            self.setting.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.setting.optimizer,
                                                                       milestones=args['milestones'],
                                                                       gamma=args['gamma'],
                                                                       last_epoch=args['last_epoch']
                                                                       )
        elif lr_scheduler_type == 'cosine':
            self.setting.lr_scheduler_type = 'cosine'
            self.setting.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.setting.optimizer,
                                                                             T_max=args['T_max'],
                                                                             eta_min=args['eta_min'],
                                                                             last_epoch=args['last_epoch']
                                                                             )
        elif lr_scheduler_type == 'cosine_warm_restart':
            self.setting.lr_scheduler_type = 'cosine_warm_restart'
            self.setting.lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.setting.optimizer,
                                                                                        T_0=args['T_0'],
                                                                                        T_mult=args['T_mult'],
                                                                                        eta_min=args['eta_min'],
                                                                                        last_epoch=args['last_epoch'],
                                                                                        verbose=args['verbose']

                                                                                        )
        elif lr_scheduler_type == 'ReduceLROnPlateau':
            self.setting.lr_scheduler_type = 'ReduceLROnPlateau'
            self.setting.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.setting.optimizer,
                                                                             mode='min',
                                                                             factor=args['factor'],
                                                                             patience=args['patience'],
                                                                             verbose=True,
                                                                             threshold=args['threshold'],
                                                                             threshold_mode='rel',
                                                                             cooldown=0,
                                                                             min_lr=0,
                                                                             eps=1e-08)
        elif lr_scheduler_type == "CyclicLR":
            self.setting.lr_scheduler_type = 'CyclicLR'
            self.setting.lr_scheduler = optim.lr_scheduler.CyclicLR(self.setting.optimizer,
                                                                            max_lr=args['max_lr'],
                                                                            base_lr=args["base_lr"],
                                                                            step_size_up=args['step_size_up'],
                                                                            mode=args['mode'],
                                                                            cycle_momentum=args['cycle_momentum'],
                                                                            verbose = args['verbose']

                                                                    )

                           

    def update_lr(self):
        if self.setting.lr_scheduler_type == 'ReduceLROnPlateau':
            self.setting.lr_scheduler.step(self.log.best_average_train_loss)
        elif self.setting.lr_scheduler_type == None:
            return    

        elif self.setting.lr_scheduler_type == 'cosine_warm_restart':
            if (self.log.epoch + ((self.log.epoch_iter)/self.log.iters))  % self.setting.lr_scheduler.T_0 == 0 and self.log.epoch >0:
                self.save_trainer("Cosine_Annealing_Save_Epoch_{}".format(str(self.log.epoch)))
            self.setting.lr_scheduler.step(self.log.epoch + self.log.epoch_iter/self.log.iters)
            self.log.lr_history.append(self.setting.lr_scheduler.get_last_lr())
      
        else:
            self.setting.lr_scheduler.step()

    def update_moving_train_loss(self, loss):
        if self.log.moving_train_loss is None:
            self.log.moving_train_loss = loss.item()
        else:
            self.log.moving_train_loss = \
                (1 - self.setting.eps_train_loss) * self.log.moving_train_loss \
                + self.setting.eps_train_loss * loss.item()

    def update_average_statistics(self, loss, phase='train'):
        if phase == 'train':
            self.log.average_train_loss = loss
            if loss < self.log.best_average_train_loss:
                self.log.best_average_train_loss = loss
                self.log.save_status.append('best_train_loss')
            self.log.list_average_train_loss_associate_iter.append([self.log.average_train_loss, self.log.iter])

        elif phase == 'val':
            self.log.average_val_index = loss
            if loss < self.log.best_average_val_index:
                self.log.best_average_val_index = loss
                self.log.save_status.append('best_val_evaluation_index')
            self.log.list_average_val_index_associate_iter.append([self.log.average_val_index, self.log.iter])

    def forward(self, input_, phase):
        time_start_load_data = time.time()
        input_ = input_.to(self.setting.device)
        self.time.train_loader_time_per_epoch += time.time() - time_start_load_data

        # Forward
        if phase == 'train':
            self.setting.optimizer.zero_grad()
        output = self.setting.network(input_)

        return output

    def backward(self, output, target,body_mask,heart_mask, lung_mask, tumor_mask):

        time_start_load_data = time.time()
        for target_i in range(len(target)):
            target[target_i] = target[target_i].to(self.setting.device)

        self.time.train_loader_time_per_epoch += time.time() - time_start_load_data

        loss = self.setting.loss_function(output, target, body_mask, heart_mask, lung_mask, tumor_mask)
        loss.backward()
        self.setting.optimizer.step()

        return loss
    def prepare_batch(self,batch):
        # ensure the batch dimension isn't dropped when batch size is 1
        # (weird error)
        if len(batch['CT'][tio.DATA].shape) == 4:
            batch['CT'][tio.DATA] = batch['CT'][tio.DATA].unsqueeze(0)
            batch['heart_glowing'][tio.DATA] = batch['heart_glowing'][tio.DATA].unsqueeze(0)
            batch['lung_glowing'][tio.DATA] = batch['lung_glowing'][tio.DATA].unsqueeze(0)
            batch['tumor_glowing'][tio.DATA] = batch['tumor_glowing'][tio.DATA].unsqueeze(0)
            batch['dose'][tio.DATA] = batch['dose'][tio.DATA].unsqueeze(0)
        inputs = torch.cat([
        batch['CT'][tio.DATA],
        batch['heart_glowing'][tio.DATA],
        batch['lung_glowing'][tio.DATA],
        batch['tumor_glowing'][tio.DATA],
        ],dim=1).float() 

        targets = batch['dose'][tio.DATA].float()
       
        return inputs, targets

    def train(self):
        time_start_train = time.time()

        self.setting.network.train()
        sum_train_loss = 0.
        count_iter = 0

        time_start_load_data = time.time()
        for batch_idx, list_loader_output in enumerate(self.setting.train_loader):

            if (self.setting.max_iter is not None) and (self.log.iter >= self.setting.max_iter - 1):
                break
            self.log.iter += 1
            self.log.epoch_iter = batch_idx
            input_, target= self.prepare_batch(list_loader_output)
 
            self.time.train_loader_time_per_epoch += time.time() - time_start_load_data

            # Forward
            output = self.forward(input_, phase='train')

            # Backward
            body_mask = list_loader_output["body"][tio.DATA].float().to(self.setting.device)
            heart_mask = list_loader_output["heart_glowing"][tio.DATA].float().to(self.setting.device)
            lung_mask = list_loader_output["lung_glowing"][tio.DATA].float().to(self.setting.device)
            tumor_mask = list_loader_output["tumor_glowing"][tio.DATA].float().to(self.setting.device)

            loss = self.backward(output, target, body_mask, heart_mask, lung_mask, tumor_mask)

            # Used for counting average loss of this epoch
            sum_train_loss += loss.item()
            count_iter += 1

            self.update_moving_train_loss(loss)
            if self.setting.lr_scheduler_type != 'ReduceLROnPlateau':
                self.update_lr()

            # Print loss during the first epoch
            if self.log.epoch == 0:
                if self.log.iter % 10 == 0:
                    self.print_log_to_file('                Iter %12d       %12.7f\n' %
                                           (self.log.iter, self.log.moving_train_loss), 'a')
            else:
                if self.log.iter % 100 == 0:
                    self.print_log_to_file('                Iter %12d       %12.7f\n' %
                                           (self.log.iter, self.log.moving_train_loss), 'a')

            time_start_load_data = time.time()

        if count_iter > 0:
            average_loss = sum_train_loss / count_iter
            self.update_average_statistics(average_loss, phase='train')

        self.time.train_time_per_epoch = time.time() - time_start_train

    def val(self):
        time_start_val = time.time()
        self.setting.network.eval()

        if self.setting.online_evaluation_function_val is None:
            self.print_log_to_file('===============================> No online evaluation method specified ! \n', 'a')
            raise Exception('No online evaluation method specified !')
        else:
            val_index = self.setting.online_evaluation_function_val(self)
            self.update_average_statistics(val_index, phase='val')

        self.time.val_time_per_epoch = time.time() - time_start_val

    def run(self):
        if self.log.iter == -1:
            self.print_log_to_file('Start training !\n', 'w')
        else:
            self.print_log_to_file('Continue training !\n', 'w')
        self.print_log_to_file(time.strftime('Local time: %H:%M:%S\n', time.localtime(time.time())), 'a')

        # Start training
        while (self.log.epoch < self.setting.max_epoch - 1) and (self.log.iter < self.setting.max_iter - 1):
            #
            time_start_this_epoch = time.time()
            self.log.epoch += 1
            # Print current learning rate
            self.print_log_to_file('Epoch: %d, iter: %d\n' % (self.log.epoch, self.log.iter), 'a')
            self.print_log_to_file('    Begin lr is %12.12f, %12.12f\n' % (
                self.setting.optimizer.param_groups[0]['lr'], self.setting.optimizer.param_groups[-1]['lr']), 'a')

            # Record initial learning rate for this epoch
            self.log.list_lr_associate_iter.append([self.setting.optimizer.param_groups[0]['lr'], self.log.iter])

            self.time.__init__()
            self.train()
            self.val()

            # If update learning rate per epoch
            if not self.setting.lr_scheduler_update_on_iter:
                self.update_lr()

            # Save trainer every "self.setting.save_per_epoch"
            if (self.log.epoch + 1) % self.setting.save_per_epoch == 0:
                self.log.save_status.append('iter_' + str(self.log.iter))
            self.log.save_status.append('latest')

            # Try save trainer
            if len(self.log.save_status) > 0:
                for status in self.log.save_status:
                    self.save_trainer(status=status)
                self.log.save_status = []

            self.print_log_to_file(
                '            Average train loss is             %12.12f,     best is           %12.12f\n' %
                (self.log.average_train_loss, self.log.best_average_train_loss), 'a')
            self.print_log_to_file(
                '            Average val evaluation index is   %12.12f,     best is           %12.12f\n'
                % (self.log.average_val_index, self.log.best_average_val_index), 'a')

            self.print_log_to_file('    Train use time %12.5f\n' % (self.time.train_time_per_epoch), 'a')
            self.print_log_to_file('    Train loader use time %12.5f\n' % (self.time.train_loader_time_per_epoch), 'a')
            self.print_log_to_file('    Val use time %12.5f\n' % (self.time.val_time_per_epoch), 'a')
            self.print_log_to_file('    Total use time %12.5f\n' % (time.time() - time_start_this_epoch), 'a')
            self.print_log_to_file('    End lr is %12.12f, %12.12f\n' % (
                self.setting.optimizer.param_groups[0]['lr'], self.setting.optimizer.param_groups[-1]['lr']), 'a')
            self.print_log_to_file(time.strftime('    time: %H:%M:%S\n', time.localtime(time.time())), 'a')

        self.print_log_to_file('===============================> End successfully\n', 'a')

    def print_log_to_file(self, txt, mode):
        if not os.path.exists(self.setting.output_dir):
            os.makedirs(self.setting.output_dir)
        with open(self.setting.output_dir + '/log.txt', mode) as log_:
            log_.write(txt)

        # Also display log in the terminal
        txt = txt.replace('\n', '')
        print(txt)

    def save_trainer(self, status='latest'):
        if len(self.setting.list_GPU_ids) > 1:
            network_state_dict = self.setting.network.module.state_dict()
        else:
            network_state_dict = self.setting.network.state_dict()

        optimizer_state_dict = self.setting.optimizer.state_dict()
        lr_scheduler_state_dict = self.setting.lr_scheduler.state_dict()

        ckpt = {
            'network_state_dict': network_state_dict,
            'lr_scheduler_state_dict': lr_scheduler_state_dict,
            'optimizer_state_dict': optimizer_state_dict,
            'log': self.log
        }

        torch.save(ckpt, self.setting.output_dir + '/' + status + '.pkl')
        self.print_log_to_file('        ==> Saving ' + status + ' model successfully !\n', 'a')

    def init_trainer(self, ckpt_file, list_GPU_ids, only_network=True):
        ckpt = torch.load(ckpt_file, map_location='cpu')

        self.setting.network.load_state_dict(ckpt['network_state_dict'])

        if not only_network:
            self.setting.lr_scheduler.load_state_dict(ckpt['lr_scheduler_state_dict'])
            self.setting.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            self.log = ckpt['log']

        self.set_GPU_device(list_GPU_ids)
        if type(self.setting.optimizer).__name__ == 'Adam':
            for key in self.setting.optimizer.state.items():
                key[1]['exp_avg'] = key[1]['exp_avg'].to(self.setting.device)
                key[1]['exp_avg_sq'] = key[1]['exp_avg_sq'].to(self.setting.device)
                key[1]['max_exp_avg_sq'] = key[1]['max_exp_avg_sq'].to(self.setting.device)

        self.print_log_to_file('==> Init trainer from ' + ckpt_file + ' successfully! \n', 'a')
