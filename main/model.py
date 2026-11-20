import torch
import torch.nn as nn
from torch.nn import functional as F

from nets.module import HumanGaussian, SMPLXParamDict, GaussianRenderer
from nets.loss import RGBLoss, SSIM, LPIPS, LaplacianReg, JointOffsetSymmetricReg, HandMeanReg,HandRGBReg
from utils.flame import flame
from utils.smpl_x import smpl_x
import copy
from config import cfg

class Model(nn.Module):
    def __init__(self, human_gaussian, smplx_param_dict,epoch):
        super(Model, self).__init__()
        self.human_gaussian = human_gaussian
        self.smplx_param_dict = smplx_param_dict
        self.gaussian_renderer = GaussianRenderer()
        self.optimizable_params = self.human_gaussian.get_optimizable_params(epoch) # for X-Humans dataset, we do not optimize smplx paraeters as it gives better results

        self.smplx_layer = copy.deepcopy(smpl_x.layer[cfg.smplx_gender])
        self.rgb_loss = RGBLoss()
        self.ssim = SSIM()
        self.lpips = LPIPS()
        self.lap_reg = LaplacianReg(smpl_x.vertex_num_upsampled, smpl_x.face_upsampled)
        self.joint_offset_sym_reg = JointOffsetSymmetricReg()
        self.hand_mean_reg = HandMeanReg()
        self.hand_rgb_reg = HandRGBReg()
        self.eval_modules = [self.lpips]
    
    def forward(self, data, mode, epoch):
        batch_size, _, img_height, img_width = data['img'].shape
        if mode == 'train':
            bg = torch.rand(3).float().cuda()
        else:
            bg = torch.ones((3)).float().cuda()

        # get assets for the rendering and render
        human_assets, human_assets_ori, human_offsets, smplx_outputs = {}, {}, {}, []
        human_renders, human_renders_ori = {}, {}
        face_renders, face_renders_refined = [], []
        for i in range(batch_size):
            smplx_param = self.smplx_param_dict([data['capture_id'][i]], [data['frame_idx'][i]])[0]
            human_asset, human_offset ,human_asset_ori , mesh_neutral_pose, point_num= self.human_gaussian(smplx_param,epoch,{k: v[i] for k,v in data['cam_param'].items()})

            # clamp scale in early of the training as garbace large scales from randomly initialized networks take HUGE GPU memory
            key_list = ['mean_3d', 'scale', 'rotation', 'rgb','opacity']
            if (mode == 'train') and cfg.is_warmup:
                human_asset['scale_wo_clamp'] = human_asset['scale'].clone()
                human_asset['scale'] = torch.clamp(human_asset['scale'], max=0.001)
                key_list += ['scale_wo_clamp']

            # gather assets
            for key in key_list:
                if key not in human_assets:
                    human_assets[key] = [human_asset[key]]
                else:
                    human_assets[key].append(human_asset[key])

            for key in ['mean_3d', 'scale', 'rotation', 'rgb','opacity']:
                if key not in human_assets_ori:
                    human_assets_ori[key] = [human_asset_ori[key]]
                else:
                    human_assets_ori[key].append(human_asset_ori[key])

            for key in ['mean_offset']:
                if key not in human_offsets:
                    human_offsets[key] = [human_offset[key]]
                else:
                    human_offsets[key].append(human_offset[key])

            # human render
            human_render = self.gaussian_renderer(human_asset, (img_height, img_width), {k: v[i] for k,v in data['cam_param'].items()},point_num,bg)
            for key in ['img', 'mask','mean_2d','is_vis','radius']:
                if key not in human_renders:
                    human_renders[key] = [human_render[key]]
                else:
                    human_renders[key].append(human_render[key])

        # aggregate assets and renders
        human_assets = {k: torch.stack(v) for k,v in human_assets.items()}
        human_renders = {k: (torch.stack(v) if k != 'mean_2d' else v) for k,v in human_renders.items()}
        if epoch >=1:
            human_offsets = {k: torch.stack(v) for k,v in human_offsets.items()}
        if mode == 'train':
            human_render_ori = self.gaussian_renderer(human_asset_ori, (img_height, img_width), {k: v[i] for k,v in data['cam_param'].items()}, 0,bg=None)
            for key in ['img']:
                if key not in human_renders_ori:
                    human_renders_ori[key] = [human_render_ori[key]]
                else:
                    human_renders_ori[key].append(human_render_ori[key])
            human_renders_ori = {k: torch.stack(v) for k,v in human_renders_ori.items()}
        
        if mode == 'train':
            stats = {'mean_2d': human_renders['mean_2d'], 'is_vis': human_renders['is_vis'].detach(), 'radius': human_renders['radius'].detach()}

            # loss functions
            loss = {}
            loss['rgb'] = self.rgb_loss(human_renders['img'], data['img'], bbox=data['bbox'], mask=data['mask'], bg=bg[None]) * cfg.rgb_loss_weight
            loss['ssim'] = (1 - self.ssim(human_renders['img'], data['img'], bbox=data['bbox'], mask=data['mask'], bg=bg[None])) * cfg.ssim_loss_weight
            loss['lpips'] = self.lpips(human_renders['img'], data['img'], bbox=data['bbox'], mask=data['mask'], bg=bg[None]) * cfg.lpips_weight
            
            if epoch>=1:
                loss['mean_3d'] = (human_offsets['mean_offset'] ** 2)

            weight = torch.ones((smpl_x.joint_num,3)).float().cuda()
            weight[smpl_x.joint_part['lhand'],:] = 10
            weight[smpl_x.joint_part['rhand'],:] = 10
            loss['joint_offset_reg'] = (self.human_gaussian.joint_offset - smpl_x.joint_offset.cuda()) ** 2 * weight
            loss['joint_offset_sym_reg'] = self.joint_offset_sym_reg(self.human_gaussian.joint_offset)
            
            out = {}
            out['human_img'] = human_renders_ori['img']
            return loss , stats ,out
        else:
            out = {}
            out['human_img'] = human_renders['img']
            out['human_asset'] = human_asset
            return out
    
    def adjust_gaussians(self, track, gaussian, cur_itr, optimizer,epoch):
        batch_size = track['mean_2d_grad'].shape[0]
        for i in range(batch_size):
            gaussian.radius_max[track['is_vis'][i]] = torch.maximum(gaussian.radius_max[track['is_vis'][i]], track['radius'][i][track['is_vis'][i]])
            gaussian.track_stats(track['mean_2d_grad'][i], track['is_vis'][i])

        if (cur_itr > cfg.densify_start_itr) and (cur_itr % cfg.densify_interval == 0):
            size_threshold = 20 if cur_itr > cfg.opacity_reset_interval else None
            gaussian.densify_and_prune(size_threshold, optimizer,cur_itr)
        if (cur_itr > 0) and (cur_itr % cfg.opacity_reset_interval == 0):
            gaussian.reset_opacity(optimizer)



def get_model(smplx_params,epoch,point_num):
    human_gaussian = HumanGaussian()
    with torch.no_grad():
        if point_num is None:
            human_gaussian.init()
        elif point_num is not None:
            human_gaussian.init_from_point_num(point_num)
    if smplx_params is not None:
        smplx_param_dict = SMPLXParamDict()
        with torch.no_grad():
            smplx_param_dict.init(smplx_params)
    else:
        smplx_param_dict = None

    model = Model(human_gaussian, smplx_param_dict,epoch)
    return model

