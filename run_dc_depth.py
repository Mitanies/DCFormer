import os
import glob
import torch
import random
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import scipy.io as scio
from common.opt import opts
from common.utils import *
from common.camera import get_uvd2xyz
from common.load_data_hm36_tds import Fusion
from common.h36m_dataset import Human36mDataset
from model.block.refine import refine
from model.DCFormer_Depth import Model as Model_dupl

import time

opt = opts().parse()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

def train(opt, actions, train_loader, model, optimizer, epoch):
    return step('train', opt, actions, train_loader, model, optimizer, epoch)

def val(opt, actions, val_loader, model):
    with torch.no_grad():
        return step('test',  opt, actions, val_loader, model)

def step(split, opt, actions, dataLoader, model, optimizer=None, epoch=None):
    model_trans = model['trans']
    model_refine = model['refine']

    if split == 'train':
        model_trans.train()
        model_refine.train()
    else:
        model_trans.eval()
        model_refine.eval()

    loss_all = {'loss': AccumLoss()}
    action_error_sum = define_error_list(actions)
    action_error_sum_refine = define_error_list(actions)


    for i, data in enumerate(tqdm(dataLoader, 0, ncols=70)):
        #if i ==5:
        #     break
        batch_cam, gt_3D, input_2D, action, subject, scale, bb_box, cam_ind = data
        [input_2D, gt_3D, batch_cam, scale, bb_box] = get_varialbe(split, [input_2D, gt_3D, batch_cam, scale, bb_box])

        if split =='train':           
            b = input_2D.size(0)  ##b, f, n, c
            f = opt.frames      
            output_z = model_trans(input_2D)

        else:
            input_2D, output_z= input_augmentation(input_2D, model_trans)
            #print(input_2D.shape)

        out_target = gt_3D.clone()
        out_target[:, :, 0] = 0
        out_target_z = out_target[:, :, :, -1].unsqueeze(-1)
        output_z = output_z[:, :, :, -1].unsqueeze(-1)
        output_z_single = output_z[:,opt.pad].unsqueeze(1)

        if out_target.size(1) > 1:
            out_target_single = out_target[:, opt.pad].unsqueeze(1)
            gt_3D_single = gt_3D[:, opt.pad].unsqueeze(1)           
        else:
            out_target_single = out_target
            gt_3D_single = gt_3D 

        if split == 'train':

            loss = mpjpe_cal(output_z, out_target_z)
            N = input_2D.size(0)
            loss_all['loss'].update(loss.detach().cpu().numpy() * N, N)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        elif split == 'test':
            output_z[:, :, 0, :] = 0
            action_error_sum = test_calculation(output_z_single, out_target_z, action, action_error_sum, opt.dataset, subject)
           
    if split == 'train':
        return loss_all['loss'].avg
    elif split == 'test':
        p1, p2 = print_error(opt.dataset, action_error_sum, opt.train)
        return p1, p2

def input_augmentation(input_2D, model_trans):
    joints_left = [4, 5, 6, 11, 12, 13]  
    joints_right = [1, 2, 3, 14, 15, 16]

    input_2D_non_flip = input_2D[:, 0]
    input_2D_flip = input_2D[:, 1]

    output_z_non_flip = model_trans(input_2D_non_flip)
    output_z_flip     = model_trans(input_2D_flip)
    output_z_flip[:, :, joints_left + joints_right, :] = output_z_flip[:, :, joints_right + joints_left, :]
    output_z = (output_z_non_flip + output_z_flip) / 2
    input_2D = input_2D_non_flip

    return input_2D, output_z[:, :, :, -1].unsqueeze(-1)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    #opt.manualSeed = 42

    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    if opt.train:
        logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
                            filename=os.path.join(opt.checkpoint, 'train.log'), level=logging.INFO)
            
    root_path = opt.root_path
    dataset_path = root_path + 'data_3d_' + opt.dataset + '.npz'

    dataset = Human36mDataset(dataset_path, opt)
    actions = define_actions(opt.actions)

    if opt.train:
        train_data = Fusion(opt=opt, train=True, dataset=dataset, root_path=root_path, tds=opt.t_downsample, use_camera=False) #use_camera: pixel CS
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=opt.batchSize//opt.stride,
                                                       shuffle=True, num_workers=int(opt.workers), pin_memory=True)

    test_data = Fusion(opt=opt, train=False,dataset=dataset, root_path =root_path, tds=opt.t_downsample, use_camera=False)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batchSize//opt.stride,
                                                  shuffle=False, num_workers=int(opt.workers), pin_memory=True)

    opt.out_joints = dataset.skeleton().num_joints()

    model = {}
    model['trans'] =  nn.DataParallel(Model_dupl(opt,2,1)).cuda()
    model['refine']=  nn.DataParallel(refine(opt)).cuda()

    model_dict = model['trans'].state_dict()
    if opt.reload:
        no_refine_path = opt.previous_dir
        pre_dict = torch.load(no_refine_path)
        #print(model_dict.keys(), pre_dict.keys())
        for name, key in model_dict.items():
            model_dict[name] = pre_dict[name]
        model['trans'].load_state_dict(model_dict)

    refine_dict = model['refine'].state_dict()
    if opt.refine_reload:
        refine_path = opt.previous_refine_name

        pre_dict_refine = torch.load(refine_path)
        for name, key in refine_dict.items():
            refine_dict[name] = pre_dict_refine[name]
        model['refine'].load_state_dict(refine_dict)

    all_param = []
    lr = opt.lr
    for i_model in model:
        all_param += list(model[i_model].parameters())
    optimizer_all = optim.Adam(all_param, lr=opt.lr, amsgrad=True)

    for epoch in range(1, opt.nepoch):
        if opt.train: 
            loss = train(opt, actions, train_dataloader, model, optimizer_all, epoch)
        
        p1, p2 = val(opt, actions, test_dataloader, model)

        # if opt.train and not opt.refine:
        #     save_model_epoch(opt.checkpoint, epoch, model['trans'])

        if opt.train and p1 < opt.previous_best_threshold:
            opt.previous_name = save_model(opt.previous_name, opt.checkpoint, epoch, p1, model['trans'], 'depth')

            if opt.refine:
                opt.previous_refine_name = save_model(opt.previous_refine_name, opt.checkpoint, epoch,
                                                      p1, model['refine'], 'refine_depth')
            opt.previous_best_threshold = p1

        if not opt.train:
            print('p1: %.2f, p2: %.2f' % (p1, p2))
            break
        else:
            logging.info('epoch: %d, lr: %.7f, loss: %.4f, p1: %.2f, p2: %.2f' % (epoch, lr, loss, p1, p2))
            print('e: %d, lr: %.7f, loss: %.4f, p1: %.2f, p2: %.2f' % (epoch, lr, loss, p1, p2))

        if epoch % opt.large_decay_epoch == 0: 
            for param_group in optimizer_all.param_groups:
                param_group['lr'] *= opt.lr_decay_large
                lr *= opt.lr_decay_large
        else:
            for param_group in optimizer_all.param_groups:
                param_group['lr'] *= opt.lr_decay
                lr *= opt.lr_decay





