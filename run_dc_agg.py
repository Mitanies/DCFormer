import os
import glob
import random
import logging
import numpy as np
from tqdm import tqdm
import scipy.io as scio
from common.opt import opts
from common.utils import *
from common.camera import get_uvd2xyz
from common.load_data_hm36_tds import Fusion
from common.h36m_dataset import Human36mDataset
from model.block.refine import refine
from model.DCFormer_Scale import Model as Model_scale
from model.DCFormer_Depth import Model as Model_depth
from model.DCFormer_Agg import Mlp as Model_agg
from common import gendb_tools

import time

opt = opts().parse()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim

def train(opt, actions, train_loader, model, optimizer, epoch):
    return step('train', opt, actions, train_loader, model, optimizer, epoch)

def val(opt, actions, val_loader, model):
    with torch.no_grad():
        return step('test',  opt, actions, val_loader, model)

def step(split, opt, actions, dataLoader, model, optimizer=None, epoch=None):
    model_trans = model['trans']
    model_refine = model['refine']
    model_pre = model['pre']
    model_agg = model['agg']

    if split == 'train':
        model_pre.eval() ##
        model_trans.train()
        model_agg.train()
        model_refine.train()
    else:
        model_pre.eval() ##
        model_trans.eval()
        model_agg.eval()
        model_refine.eval()

    loss_all = {'loss': AccumLoss()}
    action_error_sum = define_error_list(actions)
    action_error_sum_xy = define_error_list(actions)
    action_error_sum_z = define_error_list(actions)
    action_error_sum_refine = define_error_list(actions)
    action_error_sum_xy_refine = define_error_list(actions)
    action_error_sum_z_refine = define_error_list(actions)

    for i, data in enumerate(tqdm(dataLoader, 0, ncols=70)):
        #if i ==5:
        #     break
        batch_cam, gt_3D, input_2D, action, subject, scale, bb_box, cam_ind = data
        [input_2D, gt_3D, batch_cam, scale, bb_box] = get_varialbe(split, [input_2D, gt_3D, batch_cam, scale, bb_box])
        
        if split =='train':
            output_2D = model_pre(input_2D)[:, :, :, 0:2]
            output_3D_z = model_trans(input_2D)[:, :, :, -1].unsqueeze(-1) #    
            output_3D = model_agg(output_2D, output_3D_z, split) ##
        else:
            input_2D, output_3D = input_augmentation(input_2D, model_trans, model_pre, model_agg) ## z            
            #print(input_2D.shape)

        out_target = gt_3D.clone()
        output_3D_single = output_3D[:,opt.pad].unsqueeze(1)

        if split == 'test' and opt.use_camera:
            output_3D_single[:, :, 1:] +=  output_3D_single[:, :, :1]
            output_3D_single = gendb_tools.gen_camera(output_3D_single, batch_cam, out_target, direction='image_to_camera')  ##
            output_3D_single[:, :, 1:] -=  output_3D_single[:, :, :1]
            output_3D_single[:, :, 0] = 0
            out_target[:, :, 0] = 0
        elif opt.use_camera==False:
            out_target[:, :, 0] = 0

        out_target_xy = out_target[:,:,:,0:2]
        out_target_z = out_target[:, :, :, -1].unsqueeze(-1)
        output_3D_xy_single = output_3D_single[:,:,:,0:2]
        output_3D_z_single = output_3D_single[:,:,:,-1].unsqueeze(-1)

        if out_target.size(1) > 1:
            out_target_single = out_target[:, opt.pad].unsqueeze(1)
            gt_3D_single = gt_3D[:, opt.pad].unsqueeze(1)           
        else:
            out_target_single = out_target
            gt_3D_single = gt_3D

        if opt.refine:
            pred_uv = input_2D[:, opt.pad, :, :].unsqueeze(1)
            uvd = torch.cat((pred_uv, output_3D_single[:, :, :, 2].unsqueeze(-1)), -1)
            xyz = get_uvd2xyz(uvd, gt_3D_single, batch_cam) 
            xyz[:, :, 0, :] = 0
            post_out = model_refine(output_3D_single, xyz) 

        if split == 'train':
            if opt.refine:
                loss = mpjpe_cal(post_out, out_target_single)
            else:
                loss_b = b_mpjpe(output_3D, out_target) ##Better for multi
                loss = mpjpe_cal(output_3D[:,:,:,-1].unsqueeze(-1), out_target[:,:,:,-1].unsqueeze(-1))  ##
                loss = loss + 5e-7 * loss_b

            N = input_2D.size(0)
            loss_all['loss'].update(loss.detach().cpu().numpy() * N, N)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        elif split == 'test':
            output_3D[:, :, 0, :] = 0
            action_error_sum = test_calculation(output_3D_single, out_target, action, action_error_sum, opt.dataset, subject)
            action_error_sum_xy = test_calculation(output_3D_xy_single, out_target_xy, action, action_error_sum_xy, opt.dataset, subject)
            action_error_sum_z = test_calculation(output_3D_z_single, out_target_z, action, action_error_sum_z, opt.dataset, subject)      

            if opt.refine:
                output_3D[:, :, 0, :] = 0
                action_error_sum_refine = test_calculation(post_out, out_target_single, action, action_error_sum_refine, opt.dataset, subject)
                action_error_sum_xy_refine = test_calculation(post_out[:,:,:,0:2], out_target_single[:,:,:,0:2], action, action_error_sum_xy_refine, opt.dataset, subject)
                action_error_sum_z_refine = test_calculation(post_out[:,:,:,-1].unsqueeze(-1), out_target_single[:,:,:,-1].unsqueeze(-1), action, action_error_sum_z_refine, opt.dataset, subject)

    if split == 'train':
        return loss_all['loss'].avg
    elif split == 'test':
        if opt.refine:
            p1, p2 = print_error(opt.dataset, action_error_sum_refine, opt.train)
            p1_xy, p2_xy = print_error(opt.dataset, action_error_sum_xy_refine, opt.train)
            p1_z, p2_z = print_error(opt.dataset, action_error_sum_z_refine, opt.train)
        else:
            p1, p2 = print_error(opt.dataset, action_error_sum, opt.train)
            p1_xy, p2_xy = print_error(opt.dataset, action_error_sum_xy, opt.train)
            p1_z, p2_z = print_error(opt.dataset, action_error_sum_z, opt.train)
        return p1, p2, p1_xy, p2_xy, p1_z, p2_z

def input_augmentation(input_2D, model_trans, model_pre, model_agg, gt_3D=None, batch_cam=None):
    joints_left = [4, 5, 6, 11, 12, 13]  
    joints_right = [1, 2, 3, 14, 15, 16]

    input_2D_non_flip = input_2D[:, 0]
    input_2D_flip = input_2D[:, 1]

    output_3D_xy_non_flip = model_pre(input_2D_non_flip)[:,:,:,0:2]
    output_3D_z_non_flip = model_trans(input_2D_non_flip)[:,:,:,-1].unsqueeze(-1)
    output_3D_xy_flip         = model_pre(input_2D_flip)[:,:,:,0:2]
    output_3D_z_flip         = model_trans(input_2D_flip)[:,:,:,-1].unsqueeze(-1)
    output_3D_non_flip = model_agg(output_3D_xy_non_flip, output_3D_z_non_flip) ##
    output_3D_flip = model_agg(output_3D_xy_flip, output_3D_z_flip) ##
    output_3D_flip[:,:,:,0] *= -1
    output_3D_flip[:, :, joints_left + joints_right, :] = output_3D_flip[:, :, joints_right + joints_left, :]
    output_3D = (output_3D_non_flip + output_3D_flip) / 2
    input_2D = input_2D_non_flip

    return input_2D, output_3D 


if __name__ == '__main__':
    #opt.manualSeed = 42

    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    np.random.seed(opt.manualSeed) #
    torch.cuda.manual_seed_all(opt.manualSeed) #

    if opt.train:
        logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
                            filename=os.path.join(opt.checkpoint, 'train.log'), level=logging.INFO)
            
    root_path = opt.root_path
    dataset_path = root_path + 'data_3d_' + opt.dataset + '.npz'

    dataset = Human36mDataset(dataset_path, opt)
    actions = define_actions(opt.actions)

    if opt.train:
        train_data = Fusion(opt=opt, train=True, dataset=dataset, root_path=root_path, tds=opt.t_downsample, use_camera=opt.use_camera)
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=opt.batchSize//opt.stride,
                                                       shuffle=True, num_workers=int(opt.workers), pin_memory=True)

    test_data = Fusion(opt=opt, train=False,dataset=dataset, root_path =root_path, tds=opt.t_downsample, use_camera=False)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batchSize//opt.stride,
                                                  shuffle=False, num_workers=int(opt.workers), pin_memory=True)

    opt.out_joints = dataset.skeleton().num_joints()

    pre_path = opt.previous_dir_scale
    pre_dict_pre = torch.load(pre_path, map_location='cuda:0')
    no_refine_path = opt.previous_dir_depth
    pre_dict = torch.load(no_refine_path, map_location='cuda:0')

    model = {}
    model['pre'] =  nn.DataParallel(Model_scale(opt,2,pre_dict_pre['module.regress_head.weight'].shape[0])).cuda()
    model['trans'] =  nn.DataParallel(Model_depth(opt,2,pre_dict['module.regress_head.weight'].shape[0])).cuda()
    model['refine']=  nn.DataParallel(refine(opt)).cuda()
    model['agg'] =  nn.DataParallel(Model_agg(opt)).cuda()

    model_dict_pre = model['pre'].state_dict()
    for name, key in model_dict_pre.items():
        model_dict_pre[name] = pre_dict_pre[name]
    model['pre'].load_state_dict(model_dict_pre)

    model_dict = model['trans'].state_dict()
    for name, key in model_dict.items():
        model_dict[name] = pre_dict[name]
    model['trans'].load_state_dict(model_dict)

    model_dict_agg = model['agg'].state_dict()
    if opt.reload:
        agg_path = opt.previous_dir_agg
        pre_dict_agg = torch.load(agg_path, map_location='cuda:0')
        for name, key in model_dict_agg.items():
            model_dict_agg[name] = pre_dict_agg[name]
        model['agg'].load_state_dict(model_dict_agg)

    refine_dict = model['refine'].state_dict()
    if opt.refine_reload:
        refine_path = opt.previous_refine_name
        pre_dict_refine = torch.load(refine_path)
        for name, key in refine_dict.items():
            refine_dict[name] = pre_dict_refine[name]
        model['refine'].load_state_dict(refine_dict)

    all_param = []
    lr = opt.lr
    all_param += list(model['trans'].parameters())
    all_param += list(model['agg'].parameters())
    all_param += list(model['refine'].parameters())
    optimizer_all = optim.Adam(all_param, lr=opt.lr, amsgrad=True)

    for epoch in range(1, opt.nepoch):
        if opt.train: 
            loss = train(opt, actions, train_dataloader, model, optimizer_all, epoch)
            
        p1, p2, p1_xy, p2_xy, p1_z, p2_z = val(opt, actions, test_dataloader, model)

        #if opt.train and not opt.refine:
            #save_model(opt.previous_name, opt.checkpoint, epoch, p1, model['agg'], 'agg')

        if opt.train and p1 < opt.previous_best_threshold:
            opt.previous_name = save_model(opt.previous_name, opt.checkpoint, epoch, p1, model['pre'], 'agg_scale')
            opt.previous_name = save_model(opt.previous_name, opt.checkpoint, epoch, p1, model['trans'], 'agg_depth')
            opt.previous_name = save_model(opt.previous_name, opt.checkpoint, epoch, p1, model['agg'], 'agg')

            if opt.refine:
                opt.previous_refine_name = save_model(opt.previous_refine_name, opt.checkpoint, epoch,
                                                      p1, model['refine'], 'refine_agg')
            opt.previous_best_threshold = p1

        if not opt.train:
            print('p1: %.2f, p2: %.2f, p1_xy: %.2f, p2_xy: %.2f, p1_z: %.2f, p2_z: %.2f' % (p1, p2, p1_xy, p2_xy, p1_z, p2_z))
            break
        else:
            logging.info('epoch: %d, lr: %.7f, loss: %.4f, p1: %.2f, p2: %.2f, p1_xy: %.2f, p2_xy: %.2f, p1_z: %.2f, p2_z: %.2f' % (epoch, lr, loss, p1, p2, p1_xy, p2_xy, p1_z, p2_z))
            print('e: %d, lr: %.7f, loss: %.4f, p1: %.2f, p2: %.2f, p1_xy: %.2f, p2_xy: %.2f, p1_z: %.2f, p2_z: %.2f' % (epoch, lr, loss, p1, p2,  p1_xy, p2_xy, p1_z, p2_z))

        if epoch % opt.large_decay_epoch == 0: 
            for param_group in optimizer_all.param_groups:
                param_group['lr'] *= opt.lr_decay_large
                lr *= opt.lr_decay_large
        else:
            for param_group in optimizer_all.param_groups:
                param_group['lr'] *= opt.lr_decay
                lr *= opt.lr_decay





