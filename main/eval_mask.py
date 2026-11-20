import cv2
import torch
import os.path as osp
from glob import glob
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, dest='output_path')
    parser.add_argument('--subject_id', type=str, dest='subject_id')
    parser.add_argument('--include_bkg', dest='include_bkg', action='store_true')
    parser.add_argument('--use_hand_mask', dest='use_hand_mask', action='store_true')
    parser.add_argument('--use_head_mask', dest='use_head_mask', action='store_true')
    args = parser.parse_args()
    assert args.output_path, "Please set output_path."
    assert args.subject_id, "Please set subject_id."
    return args

# get path
args = parse_args()
output_path = args.output_path
subject_id = args.subject_id
include_bkg = args.include_bkg
use_head_mask = args.use_head_mask
use_hand_mask = args.use_hand_mask

results = {'psnr': [], 'ssim': [], 'lpips': []}
psnr = PeakSignalNoiseRatio(data_range=1).cuda()
ssim = StructuralSimilarityIndexMeasure(data_range=1).cuda()
lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex").cuda()

with open(osp.join('..', 'data', 'Custom', 'data', subject_id, 'frame_list_train.txt')) as f:
    lines = f.readlines()
for line in lines:
    frame_idx = int(line)
    print(frame_idx)
    mask = "masks"
    if use_head_mask:
        mask = "head_masks"
    if use_hand_mask:
        mask = "hand_masks"

    out_path = osp.join(output_path, str(frame_idx) + '_human.png')
    out = cv2.imread(out_path)[:,:,::-1]/255.
    out = torch.FloatTensor(out).permute(2,0,1)[None,:,:,:].cuda()
    
    # gt image
    gt_path = osp.join('..', 'data', 'Custom', 'data', subject_id, 'frames', '%d.png' % frame_idx)
    gt = cv2.imread(gt_path)[:,:,::-1]/255.
    gt = torch.FloatTensor(gt).permute(2,0,1)[None,:,:,:].cuda()
    
    # gt mask
    mask_path = osp.join('..', 'data', 'Custom', 'data', subject_id, mask, '%d.png' % frame_idx)
    mask = cv2.imread(mask_path)
    # mask = 1 - mask/255. # 0: bkg, 1: human
    mask=mask/255
    mask = torch.FloatTensor(mask).permute(2,0,1)[None,:,:,:].cuda()
    out = out * mask + 1 * (1 - mask)
    gt = gt * mask + 1 * (1 - mask)

    results['psnr'].append(psnr(out, gt))
    results['ssim'].append(ssim(out, gt))
    results['lpips'].append(lpips(out*2-1, gt*2-1)) # normalize to [-1,1]

print('output path: ' + output_path)
print('subject_id: ' + subject_id)
print('use_hand_mask: ' + str(use_hand_mask))
print('use_head_mask: ' + str(use_head_mask))
print({k: torch.FloatTensor(v).mean() for k,v in results.items()})

