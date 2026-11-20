import argparse
import os
import os.path as osp
import torch
import numpy as np
import json
import cv2
from glob import glob
from tqdm import tqdm
from config import cfg
from base import Tester
from utils.smpl_x import smpl_x
from utils.vis import render_mesh

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject_id', type=str, dest='subject_id')
    parser.add_argument('--test_epoch', type=str, dest='test_epoch')
    parser.add_argument('--motion_path', type=str, dest='motion_path')
    args = parser.parse_args()

    assert args.subject_id, "Please set subject ID"
    assert args.test_epoch, 'Test epoch is required.'
    assert args.motion_path, 'Motion path for the animation is required.'
    return args

def rgb_to_featuredc(rgb_values):
    """
    将RGB值转换为feature_dc格式（球谐函数系数）
    
    Args:
        rgb_values: RGB值，形状为 [N, 3] 或 [N, 1, 3]
    
    Returns:
        feature_dc: 转换后的球谐函数系数，形状为 [N, 1, 3]
    """
    # 确保输入形状正确
    if len(rgb_values.shape) == 2 and rgb_values.shape[1] == 3:
        # [N, 3] -> [N, 1, 3]
        rgb_values = rgb_values.reshape(-1, 1, 3)
    elif len(rgb_values.shape) == 3 and rgb_values.shape[1] == 1 and rgb_values.shape[2] == 3:
        # 已经是 [N, 1, 3] 格式
        pass
    else:
        raise ValueError(f"不支持的RGB形状: {rgb_values.shape}")
    
    # 使用RGB2SH转换函数
    C0 = 0.28209479177387814  # 球谐函数常数
    feature_dc = (rgb_values - 0.5) / C0
    
    return feature_dc

def main():
    # argument parse and create log
    args = parse_args()
    cfg.set_args(args.subject_id)

    tester = Tester(args.test_epoch)

    # load ID information
    root_path = osp.join('..', 'data', cfg.dataset, 'data', cfg.subject_id)
    with open(osp.join(root_path, 'smplx_optimized', 'shape_param.json')) as f:
        shape_param = torch.FloatTensor(json.load(f))
    with open(osp.join(root_path, 'smplx_optimized', 'face_offset.json')) as f:
        face_offset = torch.FloatTensor(json.load(f))
    with open(osp.join(root_path, 'smplx_optimized', 'joint_offset.json')) as f:
        joint_offset = torch.FloatTensor(json.load(f))
    with open(osp.join(root_path, 'smplx_optimized', 'locator_offset.json')) as f:
        locator_offset = torch.FloatTensor(json.load(f))
    smpl_x.set_id_info(shape_param, face_offset, joint_offset, locator_offset)

    tester.smplx_params = None
    tester._make_model()
    
    motion_name = args.motion_path.split('/')[-1]
    frame_idx_list = sorted([int(x.split('/')[-1][:-5]) for x in glob(osp.join(args.motion_path, 'smplx_optimized', 'smplx_params_smoothed', '*.json'))])
    render_shape = cv2.imread(osp.join(args.motion_path, 'frames', str(frame_idx_list[0]) + '.png')).shape[:2]
    video_out = cv2.VideoWriter(motion_name + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (render_shape[1]*2, render_shape[0]))
    for frame_idx in frame_idx_list:
        with open(osp.join(args.motion_path, 'cam_params', str(frame_idx) + '.json')) as f:
        # with open(osp.join('cam_params', str(frame_idx) + '.json')) as f:
            cam_param = {k: torch.FloatTensor(v).cuda() for k,v in json.load(f).items()}
        with open(osp.join(args.motion_path, 'smplx_optimized', 'smplx_params_smoothed', str(frame_idx) + '.json')) as f:
            smplx_param = {k: torch.FloatTensor(v).cuda().view(-1) for k,v in json.load(f).items()}

        # forward
        with torch.no_grad():
            human_asset, human_offset,human_asset_ori, mesh_neutral_pose ,_= tester.model.module.human_gaussian(smplx_param,6, cam_param)
            human_render = tester.model.module.gaussian_renderer(human_asset, render_shape, cam_param,0)
        # # 保存当前帧的高斯点为ply
        # out_dir = osp.join(args.motion_path, 'gaussian_ply')
        # os.makedirs(out_dir, exist_ok=True)
        # ply_path = osp.join(out_dir, f'{frame_idx:04d}.ply')
        # # save_ply_rgb(ply_path)
        # mean_3d = human_asset['mean_3d'].cpu().numpy()  # [N, 3]
        # scale = human_asset['scale'].cpu().numpy()      # [N, 3]
        # rgb = human_asset['rgb'].detach().cpu().numpy()          # [N, 3]
        # feature_dc = rgb_to_featuredc(rgb)
        # max_sh_degree = 0
        # feature_rest_dim = (max_sh_degree + 1)
        # feature_rest = np.zeros((rgb.shape[0], feature_rest_dim, 3))
        # # feature_dc = human_asset['feature_dc'].detach().cpu().numpy()  # [N, 1, 3]
        # # feature_rest = human_asset['feature_rest'].detach().cpu().numpy()  # [N, M, 3] 其中M取决于球谐函数阶数
        # alpha = human_asset['opacity'].detach().cpu().numpy()    # [N, 1] or [N]

        # N = mean_3d.shape[0]

        # feature_rest_shape = feature_rest.shape
        # feature_rest_dim = feature_rest_shape[1] * feature_rest_shape[2]  # 展平后的维度
        # with open(ply_path, 'w') as f:
        #     f.write("ply\n")
        #     f.write("format ascii 1.0\n")
        #     f.write(f"element vertex {N}\n")
        #     f.write("property float x\n")
        #     f.write("property float y\n")
        #     f.write("property float z\n")
        #     f.write("property float scale_x\n")
        #     f.write("property float scale_y\n")
        #     f.write("property float scale_z\n")
        #     f.write("property float f_dc_0\n") 
        #     f.write("property float f_dc_1\n") 
        #     f.write("property float f_dc_2\n") 

        #     for i in range(feature_rest_dim):
        #         f.write(f"property float feature_rest_{i}\n") 

        #     f.write("property float alpha\n")
        #     f.write("end_header\n")

        #     for i in range(N): 
        #         x, y, z = mean_3d[i] 
        #         sx, sy, sz = scale[i] 
                
        #         # 获取feature_dc值（3个通道）
        #         fdc_r, fdc_g, fdc_b = feature_dc[i][0]  # [1, 3] -> [3]
                
        #         # 展平feature_rest
        #         frest_flat = feature_rest[i].flatten()  # [M, 3] -> [M*3]
                
        #         a = alpha[i] if alpha.ndim == 1 else alpha[i][0] 
                
        #         # 写入数据
        #         f.write(f"{x} {y} {z} {sx} {sy} {sz} ")
        #         f.write(f"{fdc_r} {fdc_g} {fdc_b} ")
                
        #         # 写入展平的feature_rest
        #         for val in frest_flat:
        #             f.write(f"{val} ")
                
        #         f.write(f"{a}\n")

        face_offset = smpl_x.face_offset.cuda()[None]
        joint_offset = tester.model.module.human_gaussian.joint_offset[None]

        img = cv2.imread(osp.join(args.motion_path, 'frames', str(frame_idx) + '.png'))
        render_ori = (human_render['img'].cpu().numpy().transpose(1,2,0)[:,:,::-1]*255).copy().astype(np.uint8)
        font_size = 1.5
        thick = 3
        cv2.putText(img, 'image', (int(1/3*img.shape[1]), int(0.05*img.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, font_size, [51,51,255], thick, 2) 

        out = np.concatenate((img,render_ori),1).astype(np.uint8)
        out = cv2.putText(out, str(frame_idx), (int(out.shape[1]*0.05), int(out.shape[0]*0.05)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2, 2)
        video_out.write(out)
    
    video_out.release()
    
if __name__ == "__main__":
    main()
