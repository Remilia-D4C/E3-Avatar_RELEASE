import torch
import torch.nn as nn
from torch.nn import functional as F
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix, matrix_to_quaternion, quaternion_to_matrix, axis_angle_to_matrix, matrix_to_axis_angle
from pytorch3d.ops import knn_points
from utils.transforms import eval_sh, RGB2SH, get_fov, get_view_matrix, get_proj_matrix
from utils.smpl_x import smpl_x
from smplx.lbs import batch_rigid_transform
#from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from diff_gaussian_rasterization_depth import GaussianRasterizationSettings, GaussianRasterizer
from pytorch3d.structures import Meshes
from config import cfg
import copy
from plyfile import PlyData, PlyElement
import pickle
import json
import numpy as np
from utils.flame import flame
import math
import os
from errno import EEXIST
from os import makedirs, path
def cat_params_to_optimizer(params_new, optimizer):
    optimizable_params = {}
    for group in optimizer.param_groups:
        if group['name'] not in params_new:
            continue
        param_new = params_new[group['name']]
        stored_state = optimizer.state.get(group['params'][0], None)
        if stored_state is not None:
            stored_state['exp_avg'] = torch.cat((stored_state['exp_avg'], torch.zeros_like(param_new)))
            stored_state['exp_avg_sq'] = torch.cat((stored_state['exp_avg_sq'], torch.zeros_like(param_new)))

            del optimizer.state[group['params'][0]]
            group['params'][0] = nn.Parameter(torch.cat((group['params'][0], param_new)).requires_grad_(True))
            optimizer.state[group['params'][0]] = stored_state

            optimizable_params[group['name']] = group['params'][0]
        else:
            group['params'][0] = nn.Parameter(torch.cat((group['params'][0], param_new)).requires_grad_(True))
            optimizable_params[group['name']] = group['params'][0]
    return optimizable_params

def prune_params_from_optimizer(param_names, is_valid, optimizer):
    optimizable_params = {}
    for group in optimizer.param_groups:
        if group['name'] not in param_names:
            continue
        stored_state = optimizer.state.get(group['params'][0], None)
        if stored_state is not None:
            stored_state['exp_avg'] = stored_state['exp_avg'][is_valid]
            stored_state['exp_avg_sq'] = stored_state['exp_avg_sq'][is_valid]

            del optimizer.state[group['params'][0]]
            group['params'][0] = nn.Parameter((group['params'][0][is_valid].requires_grad_(True)))
            optimizer.state[group['params'][0]] = stored_state

            optimizable_params[group['name']] = group['params'][0]
        else:
            group['params'][0] = nn.Parameter(group['params'][0][is_valid].requires_grad_(True))
            optimizable_params[group['name']] = group['params'][0]
    return optimizable_params

def replace_param_from_optimizer(param, name, optimizer):
    optimizable_params = {}
    for group in optimizer.param_groups:
        if group['name'] == name:
            stored_state = optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state['exp_avg'] = torch.zeros_like(param)
                stored_state['exp_avg_sq'] = torch.zeros_like(param)
                del optimizer.state[group['params'][0]]

            group['params'][0] = nn.Parameter(param.requires_grad_(True))
            if stored_state is not None:
                optimizer.state[group['params'][0]] = stored_state
            optimizable_params[group['name']] = group['params'][0]
    return optimizable_params


class HumanGaussian(nn.Module):
    def __init__(self):
        super(HumanGaussian, self).__init__()

        self.smplx_layer = copy.deepcopy(smpl_x.layer[cfg.smplx_gender]).cuda()
        self.shape_param = nn.Parameter(smpl_x.shape_param.float().cuda())
        self.joint_offset = nn.Parameter(smpl_x.joint_offset.float().cuda())

        self.mean_3d = nn.Parameter(torch.empty(0).cuda())
        self.scale = nn.Parameter(torch.empty(0).cuda())
        self.rotation = nn.Parameter(torch.empty(0).cuda())
        self.opacity = nn.Parameter(torch.empty(0).cuda())
        self.rgb = nn.Parameter(torch.empty(0).cuda())

        d_query = 63     
        d_context = 122           
        d_hidden = 128

        self.query_proj = nn.Linear(d_query, d_hidden)
        self.context_proj = nn.Linear(d_context, d_hidden)
        self.attn = nn.MultiheadAttention(embed_dim=d_hidden, num_heads=4, batch_first=True)
        self.out_proj = nn.Sequential(
            nn.Linear(d_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 8)  
        )
        self.pts_freq = 10 
        self.pts_embedder = Embedder(self.pts_freq)  
        self.cnt = 0

        
    def init(self):
        # upsample mesh and other assets
        xyz, mesh_neutral_pose_wo_sample, _ , _ = self.get_neutral_pose_human(jaw_zero_pose=False, use_id_info=False)
        skinning_weight = self.smplx_layer.lbs_weights.float()

        pri_lgm = self.load_ply(f"/mnt/newssd/LHM-main/exps/meshs/LHM-training-from-scratch/LHM-MINI/test_36.ply")
        self.register_buffer('point_num', torch.LongTensor([self.pri_num]).cuda())
        
        self.register_buffer('skinning_weight', skinning_weight)

        self.register_buffer('radius_max', torch.zeros((self.pri_num)).float().cuda())
        self.register_buffer('xyz_grad_accum', torch.zeros((self.pri_num,1)).float().cuda())
        self.register_buffer('track_cnt', torch.zeros((self.pri_num,1)).float().cuda())

        self.body_grad = torch.tensor(0.0058, device='cuda')
        self.hand_grad = torch.tensor(0.0011, device='cuda')
        self.face_grad = torch.tensor(0.0021, device='cuda')

        self.flag = 1

        self.register_buffer('cam_dist_trans', torch.zeros((3)).float().cuda())
        self.register_buffer('cam_dist_radius', torch.zeros((1)).float().cuda())

        smplx_param , pripose_mean3d , joint_pri_pose , trans = self.get_smplx(use_id_info=True)
        _ , tpose , _= self.get_zero_pose_human(return_mesh=True)
        pripose_mean3d = tpose
        mean_3d = self.adjust_pri(pri_lgm['mean_3d'], pripose_mean3d)
        lgm_idxs = knn_points(mean_3d[None,:,:], pripose_mean3d[None,:,:], K=1, return_nn=True).idx[0,:,0]
        lgm_idxs = self.lr_idx_to_hr_idx(lgm_idxs)

        mesh_neutral_pose, _, _, transform_mat_neutral_pose = self.get_neutral_pose_human(jaw_zero_pose=True, use_id_info=True)
        joint_zero_pose = self.get_zero_pose_human()
        transform_mat_joint = self.get_transform_mat_joint(transform_mat_neutral_pose, joint_zero_pose, smplx_param)
        transform_mat_vertex = self.get_transform_mat_vertex(transform_mat_joint, lgm_idxs ,int(self.point_num))
        transform_mat_vertex  = torch.inverse(transform_mat_vertex)
        transformed_mean_3d = self.lbs(
            mean_3d,  
            transform_mat_vertex, 
            torch.zeros_like(mean_3d),
            int(self.point_num)
        )
        self.mean_3d = nn.Parameter(transformed_mean_3d)
        self.iter_count = 400
        self.register_buffer('trans', trans)
        self.register_buffer('pripose_mean3d', pripose_mean3d)


    def init_from_point_num(self, point_num):
        # upsample mesh and other assets
        xyz, mesh_neutral_pose_wo_sample , _ ,  _ = self.get_neutral_pose_human(jaw_zero_pose=False, use_id_info=False)
        skinning_weight = self.smplx_layer.lbs_weights.float()

        # smplx_param , pripose_mean3d , joint_pri_pose , trans = self.get_smplx(use_id_info=True)
        # self.register_buffer('trans', trans)

        self.mean_3d = nn.Parameter(torch.zeros((point_num,3)).float().cuda())
        self.scale = nn.Parameter(torch.zeros((point_num,3)).float().cuda())
        self.rotation = nn.Parameter(torch.zeros((point_num,6)).float().cuda())
        # self.rgb = nn.Parameter(torch.zeros((point_num,3)).float().cuda())
        self.feature_dc = nn.Parameter(torch.zeros((point_num, 1, 3)).float().cuda())
        self.feature_rest = nn.Parameter(torch.zeros((point_num, (cfg.max_sh_degree + 1) ** 2 - 1, 3)).float().cuda())
        self.opacity = nn.Parameter(torch.zeros((point_num,1)).float().cuda())

        self.register_buffer('skinning_weight', skinning_weight)

        #gaussian adjust
        self.register_buffer('radius_max', torch.zeros((point_num)).float().cuda())
        self.register_buffer('xyz_grad_accum', torch.zeros((point_num,1)).float().cuda())
        self.register_buffer('track_cnt', torch.zeros((point_num,1)).float().cuda())
        self.body_grad = torch.tensor(0.0002, device='cuda')
        self.hand_grad = torch.tensor(0.0011, device='cuda')
        self.face_grad = torch.tensor(0.0021, device='cuda')
        self.register_buffer('cam_dist_trans', torch.zeros((3)).float().cuda())
        self.register_buffer('cam_dist_radius', torch.zeros((1)).float().cuda())
        self.register_buffer('point_num', torch.LongTensor([point_num]).cuda())
        self.flag = 1

    def get_optimizable_params(self, epoch):
        stage1_params = [
            {'params': [self.shape_param], 'name': 'shape_param_human', 'lr': cfg.lr},
            {'params': [self.joint_offset], 'name': 'joint_offset_human', 'lr': cfg.lr},

            {'params': [self.mean_3d], 'name': 'mean_3d', 'lr': 0.000016},
            {'params': [self.scale], 'name': 'scale', 'lr': cfg.scale_lr},
            {'params': [self.rotation], 'name': 'rotation', 'lr':  cfg.rotation_lr},
            {'params': [self.opacity], 'name': 'opacity', 'lr': cfg.opacity_lr},
            {'params': [self.feature_dc], 'name': 'feature_dc_scene', 'lr': cfg.feature_lr},
            {'params': [self.feature_rest], 'name': 'feature_rest_scene', 'lr': cfg.feature_lr / 20.0},
            # {'params': [self.rgb], 'name': 'rgb', 'lr': 0.0015},
        ]

        stage2_params = [
            {'params': [self.shape_param], 'name': 'shape_param_human', 'lr': cfg.lr},
            {'params': [self.joint_offset], 'name': 'joint_offset_human', 'lr': cfg.lr},

            {'params': [self.mean_3d], 'name': 'mean_3d', 'lr': 0.0000016},
            {'params': [self.scale], 'name': 'scale', 'lr': cfg.scale_lr},
            {'params': [self.rotation], 'name': 'rotation', 'lr':  cfg.rotation_lr},
            {'params': [self.opacity], 'name': 'opacity', 'lr': cfg.opacity_lr},
            {'params': [self.feature_dc], 'name': 'feature_dc_scene', 'lr': cfg.feature_lr},
            {'params': [self.feature_rest], 'name': 'feature_rest_scene', 'lr': cfg.feature_lr / 20.0},

            {'params': self.query_proj.parameters(), 'name':'query_proj' , 'lr': 1e-4},
            {'params': self.context_proj.parameters(),'name':'context_proj' ,'lr': 1e-4},
            {'params': self.out_proj.parameters(), 'name':'out_proj','lr': 1e-4},
            {'params': self.attn.parameters(), 'name':'attn','lr': 1e-4},
        ]
        if epoch == 0:
            print("---------------------first stage-------------------")
            return stage1_params
        else:
            print("---------------------second stage-------------------")
            return stage2_params
        
    def track_stats(self, mean_2d_grad, do_update):
        mean_2d_grad = mean_2d_grad[0:int(self.point_num)]
        self.xyz_grad_accum[do_update,:] += torch.norm(mean_2d_grad[do_update,:2], dim=1, keepdim=True)
        self.track_cnt[do_update,:] += 1

    def densify_and_prune(self, screen_size_max, optimizer,cur_itr):
        grad_track = torch.nan_to_num(self.xyz_grad_accum / self.track_cnt)
        self.clone_points(grad_track, optimizer)
        self.split_points(grad_track, optimizer)
        do_prune = (torch.sigmoid(self.opacity) < cfg.opacity_min)[:,0]
        if screen_size_max and (cur_itr & 10==0):
            big_points_vs = self.radius_max > screen_size_max
            do_prune = torch.logical_or(do_prune, big_points_vs)
        self.prune_points(do_prune, optimizer)
        self.idx,_,_,_ = self.get_mask()
        torch.cuda.empty_cache()
        # self.iter_count+=100

    # def clone_points(self, grad, optimizer, lambda_pos_grad=0.0):
    #     padded_grad = torch.zeros((int(self.point_num)), dtype=torch.float32, device='cuda')
    #     padded_grad[:grad.shape[0]] = grad[:, 0]

    #     _, hand_mask, face_mask, body_mask = self.get_mask()

    #     exp_scale = torch.exp(self.scale)
    #     avg_scale_hand = exp_scale[hand_mask].mean() if hand_mask.any() else torch.tensor(0.0, device='cuda')
    #     avg_scale_face = exp_scale[face_mask].mean() if face_mask.any() else torch.tensor(0.0, device='cuda')
    #     avg_scale_body = exp_scale[body_mask].mean() if body_mask.any() else torch.tensor(0.0, device='cuda')

    #     global_min = torch.min(torch.stack([avg_scale_hand, avg_scale_face, avg_scale_body]))
    #     global_max = torch.max(torch.stack([avg_scale_hand, avg_scale_face, avg_scale_body]))

    #     def compute_size_thr(avg_scale, min_scale, max_scale, thr_min=0.002, thr_max=0.05):
    #         norm = (avg_scale - min_scale) / (max_scale - min_scale + 1e-6)
    #         return thr_min + (thr_max - thr_min) * norm

    #     size_thr_body = compute_size_thr(avg_scale_body, global_min, global_max)
    #     size_thr_hand = compute_size_thr(avg_scale_hand, global_min, global_max)
    #     size_thr_face = compute_size_thr(avg_scale_face, global_min, global_max)

    #     region_cfgs = [
    #         {
    #             "name": "body", "mask": body_mask,
    #             "grad_thr": cfg.body_grad_thr, "prev_mean": self.body_grad,
    #             "alpha": cfg.body_alpha, "size_thr": size_thr_body
    #         },
    #         {
    #             "name": "hand", "mask": hand_mask,
    #             "grad_thr": cfg.hand_grad_thr, "prev_mean": self.hand_grad,
    #             "alpha": cfg.hand_alpha, "size_thr": size_thr_hand
    #         },
    #         {
    #             "name": "face", "mask": face_mask,
    #             "grad_thr": cfg.face_grad_thr, "prev_mean": self.face_grad,
    #             "alpha": cfg.face_alpha, "size_thr": size_thr_face
    #         }
    #     ]

    #     all_mean, all_rgb, all_opacity, all_scale, all_rot = [], [], [], [], []

    #     for region in region_cfgs:
    #         name = region["name"]
    #         mask = region["mask"]
    #         alpha = region["alpha"]
    #         if mask.sum() == 0:
    #             continue

    #         grad_thr = region["grad_thr"]
    #         prev_mean = region["prev_mean"]
    #         size_thr = region["size_thr"]

    #         region_grad = padded_grad[mask]
    #         region_scale = torch.exp(self.scale)[mask]

    #         cur_mean = region_grad.mean()

    #         grad_thr_cur = ((cfg.densify_end_itr - self.iter_count) / cfg.densify_end_itr) * prev_mean + (self.iter_count / cfg.densify_end_itr) * (cur_mean) + alpha 
    #         grad_thr_cur = torch.clamp(grad_thr_cur, min=0.1 * grad_thr, max=10.0 * grad_thr)

    #         region_grad_mask = region_grad >= grad_thr_cur
    #         region_size_mask = torch.max(region_scale, dim=1).values < size_thr  
    #         selected_mask = region_grad_mask & region_size_mask

    #         if selected_mask.sum() == 0:
    #             continue

    #         final_mask = torch.zeros_like(padded_grad, dtype=torch.bool)
    #         final_mask[mask] = selected_mask

    #         all_mean.append(self.mean_3d[final_mask])
    #         all_rgb.append(self.rgb[final_mask])
    #         all_opacity.append(self.opacity[final_mask])
    #         all_scale.append(self.scale[final_mask])
    #         all_rot.append(self.rotation[final_mask])

    #     if all_mean:
    #         mean_all = torch.cat(all_mean, dim=0)
    #         rgb_all = torch.cat(all_rgb, dim=0)
    #         opacity_all = torch.cat(all_opacity, dim=0)
    #         scale_all = torch.cat(all_scale, dim=0)
    #         rot_all = torch.cat(all_rot, dim=0)

    #         self.densify(mean_all, rgb_all, opacity_all, scale_all, rot_all, optimizer)

    def densify(self, mean_3d, feature_dc,feature_rest, opacity, scale, rotation, optimizer):
        optimizable_params_new = {'mean_3d': mean_3d, 'feature_dc_scene': feature_dc, 'feature_rest_scene': feature_rest, 'opacity': opacity, 'scale': scale, 'rotation': rotation}

        optimizable_params = cat_params_to_optimizer(optimizable_params_new, optimizer)
        self.point_num[:] = optimizable_params['mean_3d'].shape[0]
        self.mean_3d = optimizable_params['mean_3d']
        self.feature_dc = optimizable_params['feature_dc_scene']
        self.feature_rest = optimizable_params['feature_rest_scene']
        self.opacity = optimizable_params['opacity']
        self.scale = optimizable_params['scale']
        self.rotation = optimizable_params['rotation']

        self.radius_max = torch.zeros((int(self.point_num))).float().cuda()
        self.xyz_grad_accum = torch.zeros((int(self.point_num),1)).float().cuda()
        self.track_cnt = torch.zeros((int(self.point_num),1)).float().cuda()

    def clone_points(self, grad, optimizer, lambda_pos_grad=0.0):
        padded_grad = torch.zeros((int(self.point_num)), dtype=torch.float32, device='cuda')
        padded_grad[:grad.shape[0]] = grad[:, 0]

        _, hand_mask, face_mask, body_mask = self.get_mask()
        print(f"[区域点数量统计 - Clone]")
        print(f"Hand 点数: {hand_mask.sum().item()}")
        print(f"Face 点数: {face_mask.sum().item()}")
        print(f"Body 点数: {body_mask.sum().item()}")

        # === 计算每个区域的平均 scale (exp) ===
        exp_scale = torch.exp(self.scale)
        avg_scale_hand = exp_scale[hand_mask].mean() if hand_mask.any() else torch.tensor(0.0, device='cuda')
        avg_scale_face = exp_scale[face_mask].mean() if face_mask.any() else torch.tensor(0.0, device='cuda')
        avg_scale_body = exp_scale[body_mask].mean() if body_mask.any() else torch.tensor(0.0, device='cuda')

        global_min = torch.min(torch.stack([avg_scale_hand, avg_scale_face, avg_scale_body]))
        global_max = torch.max(torch.stack([avg_scale_hand, avg_scale_face, avg_scale_body]))

        # === 映射函数 ===
        def compute_size_thr(avg_scale, min_scale, max_scale, thr_min=0.002, thr_max=0.05):
            norm = (avg_scale - min_scale) / (max_scale - min_scale + 1e-6)
            return thr_min + (thr_max - thr_min) * norm

        # === 阈值动态调整 ===
        size_thr_body = compute_size_thr(avg_scale_body, global_min, global_max)
        size_thr_hand = compute_size_thr(avg_scale_hand, global_min, global_max)
        size_thr_face = compute_size_thr(avg_scale_face, global_min, global_max)
        print(f"[动态 Size 阈值] hand: {size_thr_hand:.5f}, face: {size_thr_face:.5f}, body: {size_thr_body:.5f}")

        region_cfgs = [
            {
                "name": "body", "mask": body_mask,
                "grad_thr": cfg.body_grad_thr, "prev_mean": self.body_grad,
                "alpha": cfg.body_alpha, "size_thr": size_thr_body
            },
            {
                "name": "hand", "mask": hand_mask,
                "grad_thr": cfg.hand_grad_thr, "prev_mean": self.hand_grad,
                "alpha": cfg.hand_alpha, "size_thr": size_thr_hand
            },
            {
                "name": "face", "mask": face_mask,
                "grad_thr": cfg.face_grad_thr, "prev_mean": self.face_grad,
                "alpha": cfg.face_alpha, "size_thr": size_thr_face
            }
        ]

        all_mean, all_f_dc,all_f_rest, all_opacity, all_scale, all_rot = [], [], [], [], [],[]

        for region in region_cfgs:
            name = region["name"]
            mask = region["mask"]
            alpha = region["alpha"]
            if mask.sum() == 0:
                continue

            grad_thr = region["grad_thr"]
            prev_mean = region["prev_mean"]
            size_thr = region["size_thr"]

            region_grad = padded_grad[mask]
            region_scale = torch.exp(self.scale)[mask]

            cur_mean = region_grad.mean()

            grad_thr_cur = grad_thr * (1 + alpha * (prev_mean - cur_mean))
            # grad_thr_cur = (prev_mean*0.8 + cur_mean*0.2)/2
            grad_thr_cur = torch.clamp(grad_thr_cur, min=0.1 * grad_thr, max=10.0 * grad_thr)

            region_grad_mask = region_grad >= grad_thr_cur
            region_size_mask = torch.max(region_scale, dim=1).values < size_thr  # 动态 threshold 应用
            selected_mask = region_grad_mask & region_size_mask

            print("------------prev_mean - cur_mean-----------",prev_mean - cur_mean,prev_mean,cur_mean)
            print(f"[{name.upper()}] grad_thr_cur: {grad_thr_cur:.5f}, 选中点数: {selected_mask.sum().item()}")

            if selected_mask.sum() == 0:
                continue

            final_mask = torch.zeros_like(padded_grad, dtype=torch.bool)
            final_mask[mask] = selected_mask

            all_mean.append(self.mean_3d[final_mask])
            all_f_dc.append(self.feature_dc[final_mask])
            all_f_rest.append(self.feature_rest[final_mask])
            all_opacity.append(self.opacity[final_mask])
            all_scale.append(self.scale[final_mask])
            all_rot.append(self.rotation[final_mask])

        if all_mean:
            mean_all = torch.cat(all_mean, dim=0)
            f_dc_all = torch.cat(all_f_dc, dim=0)
            f_rest_all = torch.cat(all_f_rest, dim=0)
            opacity_all = torch.cat(all_opacity, dim=0)
            scale_all = torch.cat(all_scale, dim=0)
            rot_all = torch.cat(all_rot, dim=0)

            self.densify(mean_all, f_dc_all,f_rest_all, opacity_all, scale_all, rot_all, optimizer)

    def split_points(self, grad, optimizer, split_factor=2, lambda_pos_grad=0.3):
        padded_grad = torch.zeros((int(self.point_num)), dtype=torch.float32, device='cuda')
        padded_grad[:grad.shape[0]] = grad[:, 0]

        # === 获取 mask ===
        _, hand_mask, face_mask, body_mask = self.get_mask()
        print(f"[区域点数量统计]")
        print(f"Hand 点数: {hand_mask.sum().item()}")
        print(f"Face 点数: {face_mask.sum().item()}")
        print(f"Body 点数: {body_mask.sum().item()}")

        # === 计算每个区域的平均 scale（exp） ===
        avg_scale_hand = torch.exp(self.scale[hand_mask]).mean()
        avg_scale_face = torch.exp(self.scale[face_mask]).mean()
        avg_scale_body = torch.exp(self.scale[body_mask]).mean()

        # === 归一化映射函数：动态 size_thr ===
        thr_min, thr_max = 0.002, 0.05
        def compute_size_thr(avg_scale, global_min, global_max):
            norm = (avg_scale - global_min) / (global_max - global_min + 1e-6)
            return thr_min + (thr_max - thr_min) * norm

        global_min = min(avg_scale_hand, avg_scale_face, avg_scale_body)
        global_max = max(avg_scale_hand, avg_scale_face, avg_scale_body)

        size_thr_hand = compute_size_thr(avg_scale_hand, global_min, global_max)
        size_thr_face = compute_size_thr(avg_scale_face, global_min, global_max)
        size_thr_body = compute_size_thr(avg_scale_body, global_min, global_max)

        print(f"[动态 Size 阈值] hand: {size_thr_hand:.5f}, face: {size_thr_face:.5f}, body: {size_thr_body:.5f}")

        # === 区域配置 ===
        region_cfgs = [
            {
                "name": "body", "mask": body_mask,
                "grad_thr": cfg.body_grad_thr, "prev_mean": self.body_grad,
                "split": split_factor, "alpha": cfg.body_alpha, "beta": 1.0,
                "size_thr": size_thr_body
            },
            {
                "name": "hand", "mask": hand_mask,
                "grad_thr": cfg.hand_grad_thr, "prev_mean": self.hand_grad,
                "split": split_factor, "alpha": cfg.hand_alpha, "beta": 1.0,
                "size_thr": size_thr_hand
            },
            {
                "name": "face", "mask": face_mask,
                "grad_thr": cfg.face_grad_thr, "prev_mean": self.face_grad,
                "split": split_factor, "alpha": cfg.face_alpha, "beta": 1.0,
                "size_thr": size_thr_face
            }
        ]

        all_mean, all_f_dc,all_f_rest, all_opacity, all_scale, all_rot = [], [], [], [], [],[]
        do_prune_mask = torch.zeros_like(padded_grad, dtype=torch.bool)

        for region in region_cfgs:
            name = region["name"]
            mask = region["mask"]
            alpha = region["alpha"]
            beta = region["beta"]
            grad_thr = region["grad_thr"]
            prev_mean = region["prev_mean"]
            split = region["split"]
            size_thr = region["size_thr"]

            if mask.sum() == 0:
                continue

            region_grad = padded_grad[mask]
            region_size = torch.exp(self.scale)[mask]
            cur_mean = region_grad.mean()

            # === 动态阈值计算 ===
            # grad_thr_cur = (prev_mean*0.8 + cur_mean*0.2)
            grad_thr_cur = grad_thr * (1 + alpha * (prev_mean - cur_mean))
            grad_thr_cur = torch.clamp(grad_thr_cur, min=0.1 * grad_thr, max=10.0 * grad_thr)

            region_grad_mask = region_grad >= grad_thr_cur
            region_size_mask = torch.max(region_size, dim=1).values > size_thr
            selected_mask = region_grad_mask & region_size_mask
            print("------------prev_mean - cur_mean-----------",prev_mean - cur_mean)
            print(f"[{name.upper()}] grad_thr_cur: {grad_thr_cur:.5f}, 选中点数: {selected_mask.sum().item()}")

            if selected_mask.sum() == 0:
                continue

            final_mask = torch.zeros_like(padded_grad, dtype=torch.bool)
            final_mask[mask] = selected_mask

            # === 分裂点 ===
            std = torch.exp(self.scale)[final_mask].repeat(split, 1)
            mean = torch.zeros_like(std)
            samples = torch.normal(mean=mean, std=std)

            num_points_to_split = final_mask.sum().item()
            rotation = torch.eye(3, device='cuda').float()[None, ...].repeat(num_points_to_split * split, 1, 1)

            mean_new = samples + self.mean_3d[final_mask, :].repeat(split, 1)
            scale_new = self.scale[final_mask].repeat(split, 1) - math.log(split)
            rot_new = self.rotation[final_mask].repeat(split, 1)
            feature_dc_new = self.feature_dc[final_mask].repeat(split,1,1)
            feature_rest_new = self.feature_rest[final_mask].repeat(split,1,1)
            opacity_new = self.opacity[final_mask].repeat(split, 1)

            all_mean.append(mean_new)
            all_f_dc.append(feature_dc_new)
            all_f_rest.append(feature_rest_new)
            all_opacity.append(opacity_new)
            all_scale.append(scale_new)
            all_rot.append(rot_new)

            do_prune_mask |= final_mask

            # === 更新区域平均梯度记录 ===
            if name == "body":
                self.body_grad = cur_mean
            elif name == "hand":
                self.hand_grad = cur_mean
            elif name == "face":
                self.face_grad = cur_mean

        # === 聚合新点 densify + prune ===
        if all_mean:
            mean_all = torch.cat(all_mean, dim=0)
            f_dc_all = torch.cat(all_f_dc, dim=0)
            f_rest_all = torch.cat(all_f_rest, dim=0)
            opacity_all = torch.cat(all_opacity, dim=0)
            scale_all = torch.cat(all_scale, dim=0)
            rot_all = torch.cat(all_rot, dim=0)

            self.densify(mean_all, f_dc_all,f_rest_all, opacity_all, scale_all, rot_all, optimizer)

            prune_mask = torch.cat([
                do_prune_mask,
                torch.zeros(mean_all.shape[0], dtype=torch.bool, device='cuda')
            ])
            self.prune_points(prune_mask, optimizer)

    # def split_points(self, grad, optimizer, split_factor=2, lambda_pos_grad=0.3):
    #     padded_grad = torch.zeros((int(self.point_num)), dtype=torch.float32, device='cuda')
    #     padded_grad[:grad.shape[0]] = grad[:, 0]

    #     _, hand_mask, face_mask, body_mask = self.get_mask()
 
    #     avg_scale_hand = torch.exp(self.scale[hand_mask]).mean()
    #     avg_scale_face = torch.exp(self.scale[face_mask]).mean()
    #     avg_scale_body = torch.exp(self.scale[body_mask]).mean()

    #     thr_min, thr_max = 0.002, 0.05
    #     def compute_size_thr(avg_scale, global_min, global_max):
    #         norm = (avg_scale - global_min) / (global_max - global_min + 1e-6)
    #         return thr_min + (thr_max - thr_min) * norm

    #     global_min = min(avg_scale_hand, avg_scale_face, avg_scale_body)
    #     global_max = max(avg_scale_hand, avg_scale_face, avg_scale_body)

    #     size_thr_hand = compute_size_thr(avg_scale_hand, global_min, global_max)
    #     size_thr_face = compute_size_thr(avg_scale_face, global_min, global_max)
    #     size_thr_body = compute_size_thr(avg_scale_body, global_min, global_max)

    #     region_cfgs = [
    #         {
    #             "name": "body", "mask": body_mask,
    #             "grad_thr": cfg.body_grad_thr, "prev_mean": self.body_grad,
    #             "split": split_factor, "alpha": cfg.body_alpha, "beta": 1.0,
    #             "size_thr": size_thr_body
    #         },
    #         {
    #             "name": "hand", "mask": hand_mask,
    #             "grad_thr": cfg.hand_grad_thr, "prev_mean": self.hand_grad,
    #             "split": split_factor, "alpha": cfg.hand_alpha, "beta": 1.0,
    #             "size_thr": size_thr_hand
    #         },
    #         {
    #             "name": "face", "mask": face_mask,
    #             "grad_thr": cfg.face_grad_thr, "prev_mean": self.face_grad,
    #             "split": split_factor, "alpha": cfg.face_alpha, "beta": 1.0,
    #             "size_thr": size_thr_face
    #         }
    #     ]

    #     all_mean, all_rgb, all_opacity, all_scale, all_rot = [], [], [], [], []
    #     do_prune_mask = torch.zeros_like(padded_grad, dtype=torch.bool)

    #     for region in region_cfgs:
    #         name = region["name"]
    #         mask = region["mask"]
    #         alpha = region["alpha"]
    #         beta = region["beta"]
    #         grad_thr = region["grad_thr"]
    #         prev_mean = region["prev_mean"]
    #         split = region["split"]
    #         size_thr = region["size_thr"]

    #         if mask.sum() == 0:
    #             continue

    #         region_grad = padded_grad[mask]
    #         region_size = torch.exp(self.scale)[mask]
    #         cur_mean = region_grad.mean()
    #         grad_thr_cur = ((cfg.densify_end_itr - self.iter_count) / cfg.densify_end_itr) * prev_mean + (self.iter_count / cfg.densify_end_itr) * (cur_mean) + alpha
    #         grad_thr_cur = torch.clamp(grad_thr_cur, min=0.1 * grad_thr, max=10.0 * grad_thr)
    #         region_grad_mask = region_grad >= grad_thr_cur
    #         region_size_mask = torch.max(region_size, dim=1).values > size_thr
    #         selected_mask = region_grad_mask & region_size_mask

    #         if selected_mask.sum() == 0:
    #             continue

    #         final_mask = torch.zeros_like(padded_grad, dtype=torch.bool)
    #         final_mask[mask] = selected_mask

    #         std = torch.exp(self.scale)[final_mask].repeat(split, 1)
    #         mean = torch.zeros_like(std)
    #         samples = torch.normal(mean=mean, std=std)

    #         num_points_to_split = final_mask.sum().item()
    #         rotation = torch.eye(3, device='cuda').float()[None, ...].repeat(num_points_to_split * split, 1, 1)

    #         mean_new = samples + self.mean_3d[final_mask, :].repeat(split, 1)
    #         scale_new = self.scale[final_mask].repeat(split, 1) - math.log(split)
    #         rot_new = self.rotation[final_mask].repeat(split, 1)
    #         rgb_new = self.rgb[final_mask].repeat(split, 1)
    #         opacity_new = self.opacity[final_mask].repeat(split, 1)

    #         all_mean.append(mean_new)
    #         all_rgb.append(rgb_new)
    #         all_opacity.append(opacity_new)
    #         all_scale.append(scale_new)
    #         all_rot.append(rot_new)

    #         do_prune_mask |= final_mask

    #         if name == "body":
    #             self.body_grad = cur_mean
    #         elif name == "hand":
    #             self.hand_grad = cur_mean
    #         elif name == "face":
    #             self.face_grad = cur_mean

    #     if all_mean:
    #         mean_all = torch.cat(all_mean, dim=0)
    #         rgb_all = torch.cat(all_rgb, dim=0)
    #         opacity_all = torch.cat(all_opacity, dim=0)
    #         scale_all = torch.cat(all_scale, dim=0)
    #         rot_all = torch.cat(all_rot, dim=0)

    #         self.densify(mean_all, rgb_all, opacity_all, scale_all, rot_all, optimizer)

    #         prune_mask = torch.cat([
    #             do_prune_mask,
    #             torch.zeros(mean_all.shape[0], dtype=torch.bool, device='cuda')
    #         ])
    #         self.prune_points(prune_mask, optimizer)

    def prune_points(self, do_prune, optimizer):
        param_names = ['mean_3d', 'feature_dc_scene', 'feature_rest_scene', 'opacity', 'scale', 'rotation']
        is_valid = ~do_prune
        optimizable_params = prune_params_from_optimizer(param_names, is_valid, optimizer)
        
        self.point_num[:] = optimizable_params['mean_3d'].shape[0]
        self.mean_3d = optimizable_params['mean_3d']
        self.feature_dc = optimizable_params['feature_dc_scene']
        self.feature_rest = optimizable_params['feature_rest_scene']
        self.opacity = optimizable_params['opacity']
        self.scale = optimizable_params['scale']
        self.rotation = optimizable_params['rotation']

        self.radius_max = self.radius_max[is_valid]
        self.xyz_grad_accum = self.xyz_grad_accum[is_valid]
        self.track_cnt = self.track_cnt[is_valid]

    def reset_opacity(self, optimizer):
        opacity_new = torch.minimum(torch.sigmoid(self.opacity), torch.ones_like(self.opacity)*0.01)
        opacity_new = torch.log(opacity_new / (1 - opacity_new)) # inverse of sigmoid
        optimizable_params = replace_param_from_optimizer(opacity_new, 'opacity', optimizer)
        self.opacity = optimizable_params['opacity']

    def get_neutral_pose_human(self, jaw_zero_pose, use_id_info):
        zero_pose = torch.zeros((1,3)).float().cuda()
        neutral_body_pose = smpl_x.neutral_body_pose.view(1,-1).cuda() # 大 pose
        zero_hand_pose = torch.zeros((1,len(smpl_x.joint_part['lhand'])*3)).float().cuda()
        zero_expr = torch.zeros((1,smpl_x.expr_param_dim)).float().cuda()
        if jaw_zero_pose:
            jaw_pose = torch.zeros((1,3)).float().cuda()
        else:
            jaw_pose = smpl_x.neutral_jaw_pose.view(1,3).cuda() # open mouth
        if use_id_info:
            shape_param = self.shape_param[None,:]
            face_offset = smpl_x.face_offset[None,:,:].float().cuda()
            joint_offset = smpl_x.get_joint_offset(self.joint_offset[None,:,:])
            #joint_offset = None
        else:
            shape_param = torch.zeros((1,smpl_x.shape_param_dim)).float().cuda()
            face_offset = None
            joint_offset = None
        output = self.smplx_layer(global_orient=zero_pose, body_pose=neutral_body_pose, left_hand_pose=zero_hand_pose, right_hand_pose=zero_hand_pose, jaw_pose=jaw_pose, leye_pose=zero_pose, reye_pose=zero_pose, expression=zero_expr, betas=shape_param, face_offset=face_offset, joint_offset=joint_offset)
        
        mesh_neutral_pose = output.vertices[0] # 大 pose human
        mesh_neutral_pose_upsampled = smpl_x.upsample_mesh(mesh_neutral_pose) # 大 pose human
        joint_neutral_pose = output.joints[0][:smpl_x.joint_num,:] # 大 pose human

        # compute transformation matrix for making 大 pose to zero pose
        neutral_body_pose = neutral_body_pose.view(len(smpl_x.joint_part['body'])-1,3)
        zero_hand_pose = zero_hand_pose.view(len(smpl_x.joint_part['lhand']),3)
        neutral_body_pose_inv = matrix_to_axis_angle(torch.inverse(axis_angle_to_matrix(neutral_body_pose)))
        jaw_pose_inv = matrix_to_axis_angle(torch.inverse(axis_angle_to_matrix(jaw_pose)))
        pose = torch.cat((zero_pose, neutral_body_pose_inv, jaw_pose_inv, zero_pose, zero_pose, zero_hand_pose, zero_hand_pose)) 
        pose = axis_angle_to_matrix(pose)
        _, transform_mat_neutral_pose = batch_rigid_transform(pose[None,:,:,:], joint_neutral_pose[None,:,:], self.smplx_layer.parents)
        transform_mat_neutral_pose = transform_mat_neutral_pose[0]
        
        return mesh_neutral_pose_upsampled, mesh_neutral_pose, joint_neutral_pose, transform_mat_neutral_pose

    def get_smplx(self, use_id_info=False):
        smplx_param_path = f"../data/XHumans/data/{cfg.subject_id}/mesh-f00002_smplx.pkl"
        flame_path = f"../data/XHumans/data/{cfg.subject_id}/000002.json"

        with open(smplx_param_path, 'rb') as f:
            smplx_raw = pickle.load(f, encoding='latin1')

        with open(flame_path) as f:
            flame_param = json.load(f)
            if flame_param['is_valid']:
                expr = np.array(flame_param['expr'], dtype=np.float32)
            else:
                expr = np.zeros((flame.expr_param_dim), dtype=np.float32)

        smplx_param = {
            'root_pose': torch.from_numpy(smplx_raw['global_orient']).float().cuda().view(1, 3),
            'body_pose': torch.from_numpy(smplx_raw['body_pose'].reshape(-1, 3)).float().cuda(),
            'jaw_pose': torch.from_numpy(smplx_raw['jaw_pose']).float().cuda().view(1, 3),
            'leye_pose': torch.from_numpy(smplx_raw['leye_pose']).float().cuda().view(1, 3),
            'reye_pose': torch.from_numpy(smplx_raw['reye_pose']).float().cuda().view(1, 3),
            'lhand_pose': torch.from_numpy(smplx_raw['left_hand_pose'].reshape(-1, 3)).float().cuda(),
            'rhand_pose': torch.from_numpy(smplx_raw['right_hand_pose'].reshape(-1, 3)).float().cuda(),
            'trans': torch.from_numpy(smplx_raw['transl']).float().cuda().view(1, 3),
            'expr': torch.from_numpy(expr).float().cuda().view(1, -1)
        }

        root_pose = smplx_param['root_pose']
        body_pose = smplx_param['body_pose'].view(1, -1)
        jaw_pose = smplx_param['jaw_pose']
        leye_pose = smplx_param['leye_pose']
        reye_pose = smplx_param['reye_pose']
        lhand_pose = smplx_param['lhand_pose'].view(1, -1)
        rhand_pose = smplx_param['rhand_pose'].view(1, -1)
        expr = smplx_param['expr']
        trans = smplx_param['trans']

        if use_id_info:
            shape_param = self.shape_param[None, :]
            face_offset = smpl_x.face_offset[None, :, :].float().cuda()
            joint_offset = smpl_x.get_joint_offset(self.joint_offset[None, :, :])
        else:
            shape_param = torch.zeros((1, smpl_x.shape_param_dim)).float().cuda()
            face_offset = None
            joint_offset = None

        output = self.smplx_layer(
            global_orient=root_pose,
            body_pose=body_pose,
            left_hand_pose=lhand_pose,
            right_hand_pose=rhand_pose,
            jaw_pose=jaw_pose,
            leye_pose=leye_pose,
            reye_pose=reye_pose,
            expression=expr,
            betas=shape_param,
            face_offset=face_offset,
            joint_offset=joint_offset
        )

        mesh_pri_mean = output.vertices[0]
        joint_pri_pose = output.joints[0][:smpl_x.joint_num, :]

        return smplx_param, mesh_pri_mean, joint_pri_pose, trans
    
    def get_tpose(self, use_id_info=False):
        smplx_param_path = f"../data/XHumans/data/{cfg.subject_id}/mesh-f00002_smplx.pkl"
        flame_path = f"../data/XHumans/data/{cfg.subject_id}/000002.json"

        with open(smplx_param_path, 'rb') as f:
            smplx_raw = pickle.load(f, encoding='latin1')

        with open(flame_path) as f:
            flame_param = json.load(f)
            if flame_param['is_valid']:
                expr = np.array(flame_param['expr'], dtype=np.float32)
            else:
                expr = np.zeros((flame.expr_param_dim), dtype=np.float32)

        # 正常的 smplx 参数（依然返回，方便调试）
        smplx_param = {
            'root_pose': torch.from_numpy(smplx_raw['global_orient']).float().cuda().view(1, 3),
            'body_pose': torch.from_numpy(smplx_raw['body_pose'].reshape(-1, 3)).float().cuda(),
            'jaw_pose': torch.from_numpy(smplx_raw['jaw_pose']).float().cuda().view(1, 3),
            'leye_pose': torch.from_numpy(smplx_raw['leye_pose']).float().cuda().view(1, 3),
            'reye_pose': torch.from_numpy(smplx_raw['reye_pose']).float().cuda().view(1, 3),
            'lhand_pose': torch.from_numpy(smplx_raw['left_hand_pose'].reshape(-1, 3)).float().cuda(),
            'rhand_pose': torch.from_numpy(smplx_raw['right_hand_pose'].reshape(-1, 3)).float().cuda(),
            'trans': torch.from_numpy(smplx_raw['transl']).float().cuda().view(1, 3),
            'expr': torch.from_numpy(expr).float().cuda().view(1, -1)
        }

        # 形状参数/偏移
        if use_id_info:
            shape_param = self.shape_param[None, :]
            face_offset = smpl_x.face_offset[None, :, :].float().cuda()
            joint_offset = smpl_x.get_joint_offset(self.joint_offset[None, :, :])
        else:
            shape_param = torch.zeros((1, smpl_x.shape_param_dim)).float().cuda()
            face_offset = None
            joint_offset = None

        # 固定 T-pose: 所有姿态置零
        zero_pose = torch.zeros((1, 3)).float().cuda()
        zero_body = torch.zeros((1, smpl_x.body_pose_dim)).float().cuda()
        zero_hand = torch.zeros((1, smpl_x.hand_pose_dim)).float().cuda()
        zero_expr = torch.zeros((1, smpl_x.expr_param_dim)).float().cuda()

        output_tpose = self.smplx_layer(
            global_orient=zero_pose,
            body_pose=zero_body,
            left_hand_pose=zero_hand,
            right_hand_pose=zero_hand,
            jaw_pose=zero_pose,
            leye_pose=zero_pose,
            reye_pose=zero_pose,
            expression=zero_expr,
            betas=shape_param,
            face_offset=face_offset,
            joint_offset=joint_offset
        )

        mesh_pri_mean = output_tpose.vertices[0]       # T-pose 网格
        joint_pri_pose = output_tpose.joints[0][:smpl_x.joint_num, :]
        trans = smplx_param['trans']

        return smplx_param, mesh_pri_mean, joint_pri_pose, trans


    def get_zero_pose_human(self, return_mesh=False):
        zero_pose = torch.zeros((1,3)).float().cuda()
        zero_body_pose = torch.zeros((1,(len(smpl_x.joint_part['body'])-1)*3)).float().cuda()
        zero_hand_pose = torch.zeros((1,len(smpl_x.joint_part['lhand'])*3)).float().cuda()
        zero_expr = torch.zeros((1,smpl_x.expr_param_dim)).float().cuda()
        shape_param = self.shape_param[None,:]
        face_offset = smpl_x.face_offset[None,:,:].float().cuda()
        joint_offset = smpl_x.get_joint_offset(self.joint_offset[None,:,:])
        #joint_offset = None 
        output = self.smplx_layer(global_orient=zero_pose, body_pose=zero_body_pose, left_hand_pose=zero_hand_pose, right_hand_pose=zero_hand_pose, jaw_pose=zero_pose, leye_pose=zero_pose, reye_pose=zero_pose, expression=zero_expr, betas=shape_param, face_offset=face_offset, joint_offset=joint_offset)
        
        joint_zero_pose = output.joints[0][:smpl_x.joint_num,:] # zero pose human
        if not return_mesh:
            return joint_zero_pose
        else: 
            mesh_zero_pose = output.vertices[0] # zero pose human
            mesh_zero_pose_upsampled = smpl_x.upsample_mesh(mesh_zero_pose) # zero pose human
            return mesh_zero_pose_upsampled, mesh_zero_pose, joint_zero_pose

    def get_transform_mat_joint(self, transform_mat_neutral_pose, joint_zero_pose, smplx_param):
        transform_mat_joint_1 = transform_mat_neutral_pose

        root_pose = smplx_param['root_pose'].view(1,3)
        body_pose = smplx_param['body_pose'].view(len(smpl_x.joint_part['body'])-1,3)
        jaw_pose = smplx_param['jaw_pose'].view(1,3)
        leye_pose = smplx_param['leye_pose'].view(1,3)
        reye_pose = smplx_param['reye_pose'].view(1,3)
        lhand_pose = smplx_param['lhand_pose'].view(len(smpl_x.joint_part['lhand']),3)
        rhand_pose = smplx_param['rhand_pose'].view(len(smpl_x.joint_part['rhand']),3)
        trans = smplx_param['trans'].view(1,3)

        pose = torch.cat((root_pose, body_pose, jaw_pose, leye_pose, reye_pose, lhand_pose, rhand_pose)) 
        pose = axis_angle_to_matrix(pose)
        _, transform_mat_joint_2 = batch_rigid_transform(pose[None,:,:,:], joint_zero_pose[None,:,:], self.smplx_layer.parents)
        transform_mat_joint_2 = transform_mat_joint_2[0]
        
        transform_mat_joint = torch.bmm(transform_mat_joint_2, transform_mat_joint_1)
        return transform_mat_joint
    
    def get_transform_mat_vertex(self, transform_mat_joint, nn_vertex_idxs,point_num):
        skinning_weight = self.skinning_weight[nn_vertex_idxs,:]
        transform_mat_vertex = torch.matmul(skinning_weight, transform_mat_joint.view(smpl_x.joint_num,16)).view(point_num,4,4)
        return transform_mat_vertex

    def lbs(self, xyz, transform_mat_vertex, trans,point_num):
        xyz = torch.cat((xyz, torch.ones_like(xyz[:,:1])),1) # 大 pose. xyz1
        xyz = torch.bmm(transform_mat_vertex, xyz[:,:,None]).view(point_num,4)[:,:3]
        xyz = xyz + trans
        return xyz
    
    def lr_idx_to_hr_idx(self, idx):
        return idx

    def load_ply(self, path, compatible=True):
        plydata = PlyData.read(path)

        required_fields = ["x", "y", "z", "opacity", "f_dc_0", "f_dc_1", "f_dc_2", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
        for field in required_fields:
            if field not in plydata.elements[0].data.dtype.names:
                print(f"[ERROR] Missing field: {field}")
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        
        self.pri_num = xyz.shape[0]
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        shs = np.zeros((xyz.shape[0], 3))
        shs[:, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        shs[:, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
        shs[:, 2] = np.asarray(plydata.elements[0]["f_dc_2"])

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot_")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
          
        gaussians = np.concatenate([xyz, opacities, scales, rots, shs], axis=1)
        gaussians = torch.from_numpy(gaussians).float().cuda()
        gaussian_feat=gaussians
        if compatible:
            gaussians[..., 3:4] = gaussians[..., 3:4]
            gaussians[..., 4:7] = gaussians[..., 4:7]
            gaussians[..., 11:] = 0.28209479177387814 * gaussians[..., 11:] + 0.5
        mean_3d = gaussians[:,0:3].contiguous().float()
        opacity = gaussians[:,3:4].contiguous().float()
        # scale = gaussians[:,4:7].contiguous().float()
        scale = torch.full((xyz.shape[0], 3), -5.5).cuda()
        rotation = gaussians[:,7:11].contiguous().float()
        rgb = gaussians[:,11:].contiguous().float() # [N, 3]
        rotation = matrix_to_rotation_6d(quaternion_to_matrix(rotation))
        self.scale = nn.Parameter(scale)
        self.rotation = nn.Parameter(rotation)
        self.opacity = nn.Parameter(opacity)
        self.rgb = nn.Parameter(rgb)
        
        pri_lgm={
            'mean_3d': mean_3d,
            'scale': scale,
            'rotation': rotation,
            'opacity':opacity,
            'rgb': rgb
        }
        return pri_lgm
    
    def adjust_pri(self,pri_lgm,mesh_neutral_pose):
        points = pri_lgm  
        mean_min = torch.stack([
            torch.sort(points[:, i])[0][:10].mean() for i in range(3)
        ])  
        mean_max = torch.stack([
            torch.sort(points[:, i])[0][-10:].mean() for i in range(3)
        ])  
        mesh_min = torch.stack([
            torch.sort(mesh_neutral_pose[:, i])[0][:10].mean() for i in range(3)
        ])  
        mesh_max = torch.stack([
            torch.sort(mesh_neutral_pose[:, i])[0][-10:].mean() for i in range(3)
        ]) 
        mean_range = mean_max - mean_min  
        mesh_range = mesh_max - mesh_min  
        mean_range = torch.clamp(mean_range, min=1e-6)
        scale_factor = mesh_range / mean_range
        scaled_mean = pri_lgm * scale_factor 
        
        mean_center = (mean_min + mean_max) / 2  
        mesh_center = (mesh_min + mesh_max) / 2    
        trans = mesh_center - mean_center
        adjusted_mean = scaled_mean + trans
        return adjusted_mean
    
    def example_init(self, mean_3d_refined):
        self.mean_3d_refined = mean_3d_refined.unsqueeze(0)
        self.uv_vertices_shape_embeded = self.pts_embedder(self.mean_3d_refined)

    def decode_with_attention(self, smplx_param, point_feature):
        body_pose = smplx_param['body_pose'].view(1, 63)
        jaw_pose = smplx_param['jaw_pose'].view(1, 3)
        leye_pose = smplx_param['leye_pose'].view(1, 3)
        reye_pose = smplx_param['reye_pose'].view(1, 3)
        expr = smplx_param['expr'].view(1, 50)

        smplx_feat = torch.cat([body_pose, jaw_pose, leye_pose, reye_pose, expr], dim=1)  # [1, 122]
        smplx_feat = smplx_feat.unsqueeze(1)  

        Q = self.query_proj(point_feature)         
        K = self.context_proj(smplx_feat)          
        V = K                                      
        attn_out, _ = self.attn(Q, K, V)           

        out = self.out_proj(attn_out)              
        mean_3d_deforms = out[..., :3]      
        scale_coef = torch.exp(out[..., 3:4])     

        rot_delta_0 = out[..., 4:8]
        rot_delta_r = torch.exp(rot_delta_0[..., 0]).unsqueeze(-1)  
        rot_delta_v = rot_delta_0[..., 1:]  
        rot_delta = torch.cat((rot_delta_r, rot_delta_v), dim=-1)  
        return mean_3d_deforms, scale_coef ,rot_delta

    def get_mask(self):
        mesh_neutral_pose, mesh_neutral_pose_wo_upsample, _, transform_mat_neutral_pose = self.get_neutral_pose_human(
            jaw_zero_pose=True, use_id_info=True
        )

        lgm_idxs = knn_points(
            self.mean_3d[None, :, :],
            mesh_neutral_pose_wo_upsample[None, :, :],
            K=1,
            return_nn=True
        ).idx[0, :, 0]

        ori_idxs = self.lr_idx_to_hr_idx(lgm_idxs)

        rhand_idx = torch.tensor(smpl_x.rhand_vertex_idx, device=lgm_idxs.device)
        lhand_idx = torch.tensor(smpl_x.lhand_vertex_idx, device=lgm_idxs.device)
        hand_vertex_idx = torch.cat([rhand_idx, lhand_idx], dim=0)
        hand_mask = torch.isin(lgm_idxs, hand_vertex_idx)

        face_idx = torch.tensor(smpl_x.face_vertex_idx, device=lgm_idxs.device)
        face_mask = torch.isin(lgm_idxs, face_idx)

        body_mask = ~(hand_mask | face_mask)  

        return ori_idxs, hand_mask, face_mask, body_mask

    def quatProduct_batch(self,q1, q2):
        r1 = q1[:,0] # [B]
        r2 = q2[:,0]
        v1 = torch.stack((q1[:,1], q1[:,2], q1[:,3]), dim=-1) #[B,3]
        v2 = torch.stack((q2[:,1], q2[:,2], q2[:,3]), dim=-1)

        r = r1 * r2 - torch.sum(v1*v2, dim=1) # [B]
        v = r1.unsqueeze(1) * v2 + r2.unsqueeze(1) * v1 + torch.cross(v1, v2) #[B,3]
        q = torch.stack((r, v[:,0], v[:,1], v[:,2]), dim=1)

        return q
    
    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(self.feature_dc.shape[1] * self.feature_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        for i in range(self.feature_rest.shape[1] * self.feature_rest.shape[2]):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(self.scale.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self.rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l
    
    def save_ply(self, xyz,path):
        def mkdir_p(folder_path):
            # Creates a directory. equivalent to using mkdir -p on the command line
            try:
                makedirs(folder_path)
            except OSError as exc: # Python >2.5
                if exc.errno == EEXIST and os.path.isdir(folder_path):
                    pass
                else:
                    raise
        mkdir_p(os.path.dirname(path))

        # xyz = self.mean_3d.detach().cpu().numpy()
        xyz = xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = (
            self.feature_dc.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        f_rest = (
            self.feature_rest.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        opacities = self.opacity.detach().cpu().numpy()
        scale = self.scale.detach().cpu().numpy()
        rotation = self.rotation.detach().cpu().numpy()

        dtype_full = [
            (attribute, "f4") for attribute in self.construct_list_of_attributes()
        ]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1
        )
        print("xyz:", xyz.shape[1])
        print("normals:", normals.shape[1])
        print("f_dc:", f_dc.shape[1])
        print("f_rest:", f_rest.shape[1])
        print("opacities:", opacities.shape[1])
        print("scale:", scale.shape[1])
        print("rotation:", rotation.shape[1])
        print("Total cols:", xyz.shape[1] + normals.shape[1] + f_dc.shape[1] + f_rest.shape[1] + opacities.shape[1] + scale.shape[1] + rotation.shape[1])
        print("Num attributes in dtype:", len(dtype_full))

        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def forward(self, smplx_param, epoch,cam_param=None,is_world_coord=True):
        mesh_neutral_pose, _, _, transform_mat_neutral_pose = self.get_neutral_pose_human(jaw_zero_pose=True, use_id_info=True)
        joint_zero_pose = self.get_zero_pose_human()

        rotation = matrix_to_quaternion(torch.eye(3).float().cuda()[None, :, :].repeat(int(self.point_num), 1, 1))
        # rotation = matrix_to_quaternion(rotation_6d_to_matrix(self.rotation)) 
        if self.flag:
            self.idx, hand_mask,face_mask, body_mask = self.get_mask()
            self.flag = 0
        transform_mat_joint = self.get_transform_mat_joint(transform_mat_neutral_pose, joint_zero_pose, smplx_param)
        transform_mat_vertex = self.get_transform_mat_vertex(transform_mat_joint, self.idx ,int(self.point_num))
        
        transformed_mean_3d = self.lbs(
            self.mean_3d,  
            transform_mat_vertex, 
            smplx_param['trans'],
            int(self.point_num)
        )  
        
        scale = (torch.exp(self.scale))
        # rgb = self.rgb

        sh = torch.cat((self.feature_dc, self.feature_rest), 1)
        active_sh_degree = 0
        sh = sh.permute(0, 2, 1)

        cam_pos = torch.matmul(torch.inverse(cam_param['R']), -cam_param['t'].view(3, 1)).view(1, 3)
        mean_3d = self.mean_3d
        view_dir = F.normalize(mean_3d - cam_pos, p=2, dim=1)
        rgb = eval_sh(active_sh_degree, sh, view_dir)
        rgb = torch.clamp_min(rgb + 0.5, 0.0)

        
        opacity = torch.sigmoid(self.opacity)
        path = '/mnt/newssd/test1/ExAvatar_RELEASE_our/avatar/data/Custom/data/cxk/gaussian_ply/0000.ply'
        self.save_ply(transformed_mean_3d,path)
        print('save ply to ', path)
        if epoch >= 1:
            self.example_init(transformed_mean_3d)
            mean_3d_deforms, scale_coef,rot = self.decode_with_attention(smplx_param, self.uv_vertices_shape_embeded)
            transformed_mean_3d = transformed_mean_3d + mean_3d_deforms.squeeze(0)
            # scale = (torch.exp(self.scale * scale_coef.squeeze(0)))
            # rotation = self.quatProduct_batch(matrix_to_quaternion(rotation_6d_to_matrix(self.rotation)) , rot.squeeze(0))
        with torch.no_grad():
            ori_mean_3d = self.mean_3d+smplx_param['trans']
        ori_rotation = rotation.detach()
        ori_opacity = opacity.detach()
        ori_scale = scale.detach()
        ori_rgb = rgb.detach()

        assets = {
                'mean_3d': transformed_mean_3d, 
                'opacity': opacity, 
                'scale': scale, 
                'rotation': rotation, 
                'rgb': rgb
                }
        ori_assets ={
                'mean_3d': ori_mean_3d, 
                'opacity': ori_opacity, 
                'scale': ori_scale, 
                'rotation': ori_rotation, 
                'rgb': ori_rgb
        }
        if epoch >=1 :
            offsets = {
                    'mean_offset': mean_3d_deforms.squeeze(0),}
        else:
            offsets = {
                    'mean_offset':None,                    
                    }

        return assets, offsets, ori_assets ,mesh_neutral_pose , int(self.point_num)

class GaussianRenderer(nn.Module):
    def __init__(self):
        super(GaussianRenderer, self).__init__()
    
    def forward(self, gaussian_assets, img_shape, cam_param,body_num, bg=None):
        # assets for the rendering
        mean_3d = gaussian_assets['mean_3d']
        opacity = gaussian_assets['opacity']
        scale = gaussian_assets['scale']
        rotation = gaussian_assets['rotation']
        rgb = gaussian_assets['rgb']

        # create rasterizer
        fov = get_fov(cam_param['focal'], cam_param['princpt'], img_shape)
        view_matrix = get_view_matrix(cam_param['R'], cam_param['t']).permute(1,0)
        proj_matrix = get_proj_matrix(cam_param['focal'], cam_param['princpt'], img_shape, 0.01, 100, 1.0).permute(1,0)
        full_proj_matrix = torch.mm(view_matrix, proj_matrix)
        cam_pos = view_matrix.inverse()[3,:3]
        if bg is None:
            bg = torch.ones((3)).float().cuda()
        raster_settings = GaussianRasterizationSettings(
            image_height=img_shape[0],
            image_width=img_shape[1],
            tanfovx=float(torch.tan(fov[0]/2)),
            tanfovy=float(torch.tan(fov[1]/2)),
            bg=bg, 
            scale_modifier=1.0,
            viewmatrix=view_matrix, 
            projmatrix=full_proj_matrix,
            sh_degree=0, # dummy sh degree. as rgb values are already computed, rasterizer does not use this one
            campos=cam_pos,
            prefiltered=False,
            debug=False
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        
        # prepare Gaussian position in the image space for the gradient tracking
        point_num = mean_3d.shape[0]
        mean_2d = torch.zeros((point_num,3)).float().cuda()
        mean_2d.requires_grad = True
        mean_2d.retain_grad()
        
        # rasterize visible Gaussians to image and obtain their radius (on screen). 
        render_img, radius, render_depthmap, render_mask = rasterizer(
            means3D=mean_3d,
            means2D=mean_2d,
            shs=None,
            colors_precomp=rgb,
            opacities=opacity,
            scales=scale,
            rotations=rotation,
            cov3D_precomp=None)

        return {'img': render_img,
                'depthmap': render_depthmap,
                'mask': render_mask,
                'mean_2d': mean_2d,
                'is_vis': radius[0:body_num]> 0,
                'radius': radius[0:body_num]}

class SMPLXParamDict(nn.Module):
    def __init__(self):
        super(SMPLXParamDict, self).__init__()

    # initialize SMPL-X parameters of all frames
    # used to train models from scratch
    def init(self, smplx_params):
        _smplx_params = {}
        for capture_id in smplx_params.keys():
            _smplx_params[capture_id] = nn.ParameterDict({})
            for frame_idx in smplx_params[capture_id].keys():
                _smplx_params[capture_id][str(frame_idx)] = nn.ParameterDict({})
                for param_name in ['root_pose', 'body_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'lhand_pose', 'rhand_pose', 'expr', 'trans']:
                    if 'pose' in param_name:
                        _smplx_params[capture_id][str(frame_idx)][param_name] = nn.Parameter(matrix_to_rotation_6d(axis_angle_to_matrix(smplx_params[capture_id][frame_idx][param_name].cuda())))
                    else:
                        _smplx_params[capture_id][str(frame_idx)][param_name] = nn.Parameter(smplx_params[capture_id][frame_idx][param_name].cuda())
        self.smplx_params = nn.ParameterDict(_smplx_params)

    def get_optimizable_params(self):
        optimizable_params = []
        for capture_id in self.smplx_params.keys():
            for frame_idx in self.smplx_params[capture_id].keys():
                for param_name in self.smplx_params[capture_id][frame_idx].keys():
                    optimizable_params.append({'params': [self.smplx_params[capture_id][frame_idx][param_name]], 'name': 'smplx_' + param_name + '_' + capture_id + '_' + frame_idx, 'lr': cfg.smplx_param_lr})
        return optimizable_params

    def forward(self, capture_ids, frame_idxs):
        out = []
        for capture_id, frame_idx in zip(capture_ids, frame_idxs):
            capture_id = str(capture_id)
            frame_idx = str(int(frame_idx))
            smplx_param = {}

            for param_name in self.smplx_params[capture_id][frame_idx].keys():
                if 'pose' in param_name:
                    smplx_param[param_name] = matrix_to_axis_angle(rotation_6d_to_matrix(self.smplx_params[capture_id][frame_idx][param_name]))
                else:
                    smplx_param[param_name] = self.smplx_params[capture_id][frame_idx][param_name]
            out.append(smplx_param)
        return out

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, hidden_layers=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fcs = nn.ModuleList(
            [nn.Linear(input_dim, hidden_dim)] + [nn.Linear(hidden_dim, hidden_dim) for i in range(hidden_layers - 1)]
        )
        self.output_linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, input):
        # input: B,V,d
        batch_size, N_v, input_dim = input.shape
        input_ori = input.reshape(batch_size * N_v, -1)
        h = input_ori
        for i, l in enumerate(self.fcs):
            h = self.fcs[i](h)
            h = F.relu(h)
        output = self.output_linear(h)
        output = output.reshape(batch_size, N_v, -1)

        return output


class Embedder(nn.Module):
    def __init__(self, N_freqs, input_dims=3, include_input=True) -> None:
        super().__init__()
        self.log_sampling = True
        self.periodic_fns = [torch.sin, torch.cos]
        self.max_freq = N_freqs - 1
        self.N_freqs = N_freqs
        self.include_input = include_input
        self.input_dims = input_dims
        embed_fns = []
        if self.include_input:
            embed_fns.append(lambda x: x)

        if self.log_sampling:
            freq_bands = 2. ** torch.linspace(0.,
                                              self.max_freq, steps=self.N_freqs)
        else:
            freq_bands = torch.linspace(
                2. ** 0., 2. ** self.max_freq, steps=self.N_freqs)

        for freq in freq_bands:
            for p_fn in self.periodic_fns:
                embed_fns.append(lambda x, p_fn=p_fn,
                                        freq=freq: p_fn(x * freq))
        self.embed_fns = embed_fns
        self.dim_embeded = self.input_dims * len(self.embed_fns)

    def forward(self, inputs, alpha=10.):
        output = torch.cat([fn(inputs) for fn in self.embed_fns], 2)
        start = 0
        # print(alpha)
        # if self.include_input:
        #     start = 1
        # for i in range(output.shape[1]//2):
        #     output[:, (2*i+start)*self.input_dims:(2*(i+1)+start)*self.input_dims] *= (1-math.cos(math.pi*(max(min(alpha-i, 1.), 0.))))*.5
        return output