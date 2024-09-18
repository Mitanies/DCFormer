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
from common.opt import opts
from common.utils import *
from common.camera import get_uvd2xyz
from common.load_data_3dhp_mae import Fusion
from common.h36m_dataset import Human36mDataset
from model.block.refine import refine
from model.DCFormer_Scale import Model as Model_scale
from model.DCFormer_Depth import Model as Model_depth
from model.DCFormer_Agg import Mlp as Model_agg

#from thop import clever_format
#from thop.profile import profile
import scipy.io as scio

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
    model_pre = model['pre']
    model_agg = model['agg']

    if split == 'train':
        model_pre.eval()
        model_trans.train()
        model_agg.train()
        model_refine.train()
    else:
        model_pre.eval() 
        model_trans.eval()
        model_agg.eval()
        model_refine.eval()

    loss_all = {'loss': AccumLoss()}
    error_sum = AccumLoss()
    error_sum_test = AccumLoss()
    error_sum_test_xy = AccumLoss()
    error_sum_test_z = AccumLoss()

    action_error_sum = define_error_list(actions)
    action_error_sum_xy = define_error_list(actions)
    action_error_sum_z = define_error_list(actions)
    action_error_sum_post_out = define_error_list(actions)
   
    auc_sum, pck_sum = [], []

    joints_left = [5, 6, 7, 11, 12, 13]
    joints_right = [2, 3, 4, 8, 9, 10]

    data_inference = {}

    for i, data in enumerate(tqdm(dataLoader, 0, ncols=70)):

        if split == "train":
            batch_cam, gt_3D, input_2D, seq, subject, scale, bb_box, cam_ind = data
        else:
            batch_cam, gt_3D, input_2D, seq, scale, bb_box = data

        [input_2D, gt_3D, batch_cam, scale, bb_box] = get_varialbe(split,
                                                                       [input_2D, gt_3D, batch_cam, scale, bb_box])

        N = input_2D.size(0)

        out_target = gt_3D.clone().view(N, -1, opt.out_joints, opt.out_channels)
        out_target[:, :, 14] = 0
        gt_3D = gt_3D.view(N, -1, opt.out_joints, opt.out_channels).type(torch.cuda.FloatTensor)

        if out_target.size(1) > 1:
            out_target_single = out_target[:, opt.pad].unsqueeze(1)
            gt_3D_single = gt_3D[:, opt.pad].unsqueeze(1)
        else:
            out_target_single = out_target
            gt_3D_single = gt_3D

        if opt.test_augmentation and split =='test':
            input_2D, output_3D = input_augmentation(input_2D, model_pre, model_trans, model_agg, joints_left, joints_right)
        else:
            input_2D = input_2D.view(N, -1, opt.n_joints, opt.in_channels, 1).permute(0, 3, 1, 2, 4).type(torch.cuda.FloatTensor)
            output_3D_z = model_trans(input_2D, is_3dhp=True)[:, :, :, -1].unsqueeze(-1) #    
            output_2D = model_pre(input_2D, is_3dhp=True)[:, :, :, 0:2]
            output_3D = model_agg(output_2D, output_3D_z, split) ##

        output_3D = output_3D * scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, output_3D.size(1),opt.out_joints, opt.out_channels)
        output_3D_single = output_3D[:,opt.pad].unsqueeze(1) 

        out_target_xy = out_target[:,:,:,0:2]
        out_target_z = out_target[:, :, :, -1].unsqueeze(-1)
        output_3D_xy_single = output_3D_single[:,:,:,0:2]
        output_3D_z_single = output_3D_single[:,:,:,-1].unsqueeze(-1)

        if split == 'train':
            pred_out = output_3D
        elif split == 'test':
            auc_sum.append(compute_AUC(out_target_single.view(-1, 17, 3).cpu().numpy(), output_3D_single.view(-1, 17, 3).cpu().numpy())) ##
            pck_sum.append(compute_PCK(out_target_single.view(-1, 17, 3).cpu().numpy(), output_3D_single.view(-1, 17, 3).cpu().numpy()))
            pred_out = output_3D_single

        input_2D = input_2D.permute(0, 2, 3, 1, 4).view(N, -1, opt.n_joints ,2)

        if opt.refine:
            pred_uv = input_2D
            uvd = torch.cat((pred_uv[:, opt.pad, :, :].unsqueeze(1), output_3D_single[:, :, :, 2].unsqueeze(-1)), -1)
            xyz = get_uvd2xyz(uvd, gt_3D_single, batch_cam)
            xyz[:, :, 0, :] = 0
            post_out = model_refine(output_3D_single, xyz)
            loss = mpjpe_cal(post_out, out_target_single)
        else:
            loss = mpjpe_cal(pred_out[:,:,:,-1].unsqueeze(-1), out_target[:,:,:,-1].unsqueeze(-1))  ##
            loss_b = b_mpjpe(pred_out, out_target) ##Better for multi
            loss = loss + 5e-7 * loss_b
                
        loss_all['loss'].update(loss.detach().cpu().numpy() * N, N)

        if split == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if opt.refine:
                post_out[:,:,14,:] = 0
                joint_error = mpjpe_cal(post_out, out_target_single).item()
            else:
                pred_out[:,:,14,:] = 0
                joint_error = mpjpe_cal(pred_out, out_target).item()

            error_sum.update(joint_error*N, N)

        elif split == 'test':

            pred_out[:, :, 14, :] = 0
            #action_error_sum = test_calculation(pred_out, out_target, action, action_error_sum, opt.dataset, subject)
            joint_error_test = mpjpe_cal(pred_out, out_target).item()
            joint_error_test_xy = mpjpe_cal(output_3D_xy_single, out_target_xy).item()
            joint_error_test_z = mpjpe_cal(output_3D_z_single, out_target_z).item()
            out = pred_out

            if opt.train == 0:
                for seq_cnt in range(len(seq)):
                    seq_name = seq[seq_cnt]
                    if seq_name in data_inference:
                        data_inference[seq_name] = np.concatenate(
                            (data_inference[seq_name], out[seq_cnt].permute(2, 1, 0).cpu().numpy()), axis=2)
                    else:
                        data_inference[seq_name] = out[seq_cnt].permute(2, 1, 0).cpu().numpy()

            error_sum_test.update(joint_error_test * N, N)
            error_sum_test_xy.update(joint_error_test_xy * N, N)
            error_sum_test_z.update(joint_error_test_z * N, N)


    if split == 'train':
        return loss_all['loss'].avg, error_sum.avg
    elif split == 'test':
        if opt.refine:
            p1, p2 = print_error(opt.dataset, action_error_sum_post_out, opt.train)
        else:
            #p1, p2 = print_error(opt.dataset, action_error_sum, opt.train)
            if opt.train == 0:
                for seq_name in data_inference.keys():
                    data_inference[seq_name] = data_inference[seq_name][:, :, None, :]
                mat_path = os.path.join(opt.checkpoint, 'inference_data_81_3dhp.mat')
                scio.savemat(mat_path, data_inference)

        print(np.mean(auc_sum), np.mean(pck_sum))
        return error_sum_test.avg, error_sum_test_xy.avg, error_sum_test_z.avg


def input_augmentation(input_2D, model_pre, model_trans, model_agg, joints_left, joints_right):
    N, _, T, J, C = input_2D.shape 

    input_2D_flip = input_2D[:, 1].view(N, T, J, C, 1).permute(0, 3, 1, 2, 4)   
    input_2D_non_flip = input_2D[:, 0].view(N, T, J, C, 1).permute(0, 3, 1, 2, 4) 

    output_3D_z_non_flip = model_trans(input_2D_non_flip, is_3dhp=True)[:,:,:,-1].unsqueeze(-1)
    output_3D_xy_non_flip = model_pre(input_2D_non_flip, is_3dhp=True)[:,:,:,0:2]
    output_3D_z_flip         = model_trans(input_2D_flip, is_3dhp=True)[:,:,:,-1].unsqueeze(-1)
    output_3D_xy_flip         = model_pre(input_2D_flip, is_3dhp=True)[:,:,:,0:2]
    output_3D_non_flip = model_agg(output_3D_xy_non_flip, output_3D_z_non_flip) ##
    output_3D_flip = model_agg(output_3D_xy_flip, output_3D_z_flip) ##
    output_3D_flip[:,:,:, 0] *= -1
    output_3D_flip[:, :, joints_left + joints_right] = output_3D_flip[:, :, joints_right + joints_left]
    output_3D = (output_3D_non_flip + output_3D_flip) / 2
    input_2D = input_2D_non_flip

    return input_2D, output_3D


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    #opt.manualSeed = 1

    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.cuda.manual_seed_all(opt.manualSeed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if opt.train == 1:
        logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
                            filename=os.path.join(opt.checkpoint, 'train.log'), level=logging.INFO)
            
    root_path = opt.root_path
    dataset_path = root_path + 'data_3d_' + opt.dataset + '.npz'

    #dataset = Human36mDataset(dataset_path, opt)
    actions = define_actions(opt.actions)

    if opt.train:
        #train_data = Fusion(opt=opt, train=True, root_path=root_path)
        train_data = Fusion(opt=opt, train=True, root_path=root_path, MAE=opt.MAE)
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=opt.batchSize//opt.stride, ##
                                                       shuffle=True, num_workers=int(opt.workers), pin_memory=True)
    if opt.test:
        #test_data = Fusion(opt=opt, train=False,root_path =root_path)
        test_data = Fusion(opt=opt, train=False, root_path=root_path, MAE=opt.MAE)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batchSize//opt.stride,
                                                      shuffle=False, num_workers=int(opt.workers), pin_memory=True)

    opt.out_joints = 17

    pre_path = opt.previous_dir_scale
    pre_dict_pre = torch.load(pre_path, map_location='cuda:0')
    no_refine_path = opt.previous_dir_depth
    pre_dict = torch.load(no_refine_path, map_location='cuda:0')

    model = {}
    model['pre'] = nn.DataParallel(Model_scale(opt,2,pre_dict_pre['module.regress_head.weight'].shape[0])).cuda()
    model['trans'] = nn.DataParallel(Model_depth(opt,2,pre_dict['module.regress_head.weight'].shape[0])).cuda()
    model['agg'] = nn.DataParallel(Model_agg(opt)).cuda()
    model['refine'] = nn.DataParallel(refine(opt)).cuda()

    model_params = 0
    for parameter in model['trans'].parameters():
        model_params += parameter.numel()
    for parameter in model['pre'].parameters():
        model_params += parameter.numel()
    for parameter in model['agg'].parameters():
        model_params += parameter.numel()
    print('INFO: Trainable parameter count:', model_params)


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
        pre_dict_agg = torch.load(agg_path)
        for name, key in model_dict_agg.items():
            model_dict_agg[name] = pre_dict_agg[name]
        model['agg'].load_state_dict(model_dict_agg)

    refine_dict = model['refine'].state_dict()
    if opt.refine_reload == 1:
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
        if opt.train == 1:
            loss, mpjpe = train(opt, actions, train_dataloader, model, optimizer_all, epoch)
        if opt.test == 1:
            p1, p1_xy, p1_z = val(opt, actions, test_dataloader, model)
            data_threshold = p1

            if opt.train and data_threshold < opt.previous_best_threshold:
                opt.previous_name = save_model(opt.previous_name, opt.checkpoint, epoch, p1, model['pre'], 'agg_scale')
                opt.previous_name = save_model(opt.previous_name, opt.checkpoint, epoch, data_threshold, model['trans'], '3dhp_agg_depth')
                opt.previous_name = save_model(opt.previous_name, opt.checkpoint, epoch, data_threshold, model['agg'], '3dhp_agg')
                if opt.refine:
                    opt.previous_refine_name = save_model(opt.previous_refine_name, opt.checkpoint, epoch,
                                                              data_threshold, model['refine'], 'refine_agg')
                opt.previous_best_threshold = data_threshold

            if opt.train == 0:
                print('p1: %.2f, p1_xy: %.2f, p1_z: %.2f' % (p1, p1_xy, p1_z))
                break
            else:
                logging.info('epoch: %d, lr: %.7f, loss: %.4f, MPJPE: %.2f, p1: %.2f, p1_xy: %.2f, p1_z: %.2f' % (epoch, lr, loss, mpjpe, p1, p1_xy, p1_z))
                print('e: %d, lr: %.7f, loss: %.4f, M: %.2f, p1: %.2f, p1_xy: %.2f, p1_z: %.2f' % (epoch, lr, loss, mpjpe, p1, p1_xy, p1_z))

        if epoch % opt.large_decay_epoch == 0: 
            for param_group in optimizer_all.param_groups:
                param_group['lr'] *= opt.lr_decay_large
                lr *= opt.lr_decay_large
        else:
            for param_group in optimizer_all.param_groups:
                param_group['lr'] *= opt.lr_decay
                lr *= opt.lr_decay








