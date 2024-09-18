
import argparse
import os
import os.path as osp
import scipy.io as sio
import numpy as np
import torch


def gen_camera(pose3d, camera, gt3d=None, direction='camera_to_image', rootIdx=0): #pose3d: [b, f, 17, 3], camera: [b, 4]
    cam = [] 
    gen_3d = [] 
    for i in range(camera.size(0)):
        cam.append({'fx':camera[i, 0].item(), 'fy':camera[i, 1].item(), 'cx':camera[i, 2].item(), 'cy':camera[i, 3].item()})
    for i in range(pose3d.size(0)):
        for j in range(pose3d.size(1)):          
            if direction == 'camera_to_image':
                box = _infer_box(pose3d[i, j].cpu().numpy(), cam[i], rootIdx)  #3D: [17, 3], cam:[4]
                gen_3d.append(camera_to_image_frame(pose3d[i, j].cpu().numpy(), box, cam[i], rootIdx))
            else:
                box = _infer_box(gt3d[i, j].cpu().numpy(), cam[i], rootIdx)  #3D: [17, 3], cam:[4]
                gen_3d.append(image_to_camera_frame(pose3d[i, j].cpu().numpy(), box, cam[i], rootIdx, gt3d[i, j].cpu().numpy())) #root_depth: [rootIdx, 2] for 3D_gt
    gen_3d = torch.tensor(np.array(gen_3d)).view(pose3d.size(0), -1, pose3d.size(2), pose3d.size(3)).cuda()
    return gen_3d


def gen_camera_fetch(pose3d, camera, rootIdx=0): #pose3d: [x, 17, 3], camera: dict for one camera
    cam = {'fx':camera['intrinsic'][0].item(), 'fy':camera['intrinsic'][1].item(), 'cx':camera['intrinsic'][2].item(), 'cy':camera['intrinsic'][3].item()}
    positions_3d = []
    for i in range(len(pose3d)):
        box = _infer_box(pose3d[i], cam, rootIdx)
        #print(cam['fx'], box)
        positions_3d.append(camera_to_image_frame(pose3d[i], box, cam, rootIdx))
    return np.array(positions_3d)


def camera_to_image_frame(pose3d, box, camera, rootIdx):
    rectangle_3d_size = 2000.0
    ratio = 1 #(box[2] - box[0] + 1) / rectangle_3d_size
    pose3d_image_frame = np.zeros_like(pose3d)
    pose3d_image_frame[:, :2] = _weak_project(
        pose3d.copy(), camera['fx'], camera['fy'], camera['cx'], camera['cy'])
    pose3d_depth = ratio * (pose3d[:, 2] - pose3d[rootIdx, 2])
    pose3d_image_frame[:, 2] = pose3d_depth
    return pose3d_image_frame


def image_to_camera_frame(pose3d_image_frame, box, camera, rootIdx, root):
    rectangle_3d_size = 2000.0
    ratio = 1 #(box[2] - box[0] + 1) / rectangle_3d_size
    root_depth = root[rootIdx, 2].item()

    pose3d_image_frame[:, 2] = pose3d_image_frame[:, 2] / ratio + root_depth  #/ ratio 
    root_scale = root[:, 2] 
    root_scale[1:] = root_scale[1:] + root_depth  
    #print(root_depth, pose3d_image_frame[:, 2], root_scale)

    cx, cy, fx, fy = camera['cx'], camera['cy'], camera['fx'], camera['fy']
    #cx, cy = 0, 0
    pose3d_image_frame[:, 0] = (pose3d_image_frame[:, 0] - cx) / fx
    pose3d_image_frame[:, 1] = (pose3d_image_frame[:, 1] - cy) / fy
    pose3d_image_frame[:, 0] *= pose3d_image_frame[:, 2]
    pose3d_image_frame[:, 1] *= pose3d_image_frame[:, 2]
    return pose3d_image_frame


def _infer_box(pose3d, camera, rootIdx):  # 3D: [17, 3], cam:[4]
    root_joint = pose3d[rootIdx, :]
    tl_joint = root_joint.copy()
    tl_joint[:2] -= 1000.0
    br_joint = root_joint.copy()
    br_joint[:2] += 1000.0
    tl_joint = np.reshape(tl_joint, (1, 3))
    br_joint = np.reshape(br_joint, (1, 3))

    tl2d = _weak_project(tl_joint, camera['fx'], camera['fy'], camera['cx'],
                         camera['cy']).flatten()

    br2d = _weak_project(br_joint, camera['fx'], camera['fy'], camera['cx'],
                         camera['cy']).flatten()
    return np.array([tl2d[0], tl2d[1], br2d[0], br2d[1]])


def _weak_project(pose3d, fx, fy, cx, cy):
       
    for i in range(len(pose3d)):
        if pose3d[i, 2] == 0:
            print(i, pose3d[i])

    #cx, cy = 0, 0
    pose2d = pose3d[:, :2]  / pose3d[:, 2:3] ##
    pose2d[:, 0] *= fx
    pose2d[:, 1] *= fy
    pose2d[:, 0] += cx
    pose2d[:, 1] += cy
    return pose2d


def _retrieve_camera(cameras, subject, cameraidx):
    R, T, f, c, k, p, name = cameras[(subject, cameraidx + 1)]
    camera = {}
    camera['R'] = R
    camera['T'] = T
    camera['fx'] = f[0]
    camera['fy'] = f[1]
    camera['cx'] = c[0]
    camera['cy'] = c[1]
    camera['k'] = k
    camera['p'] = p
    camera['name'] = name
    return camera


