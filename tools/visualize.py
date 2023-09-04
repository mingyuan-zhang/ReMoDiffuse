import argparse
import os
import os.path as osp
import mmcv
import numpy as np
import torch
from mogen.models import build_architecture
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel
from mogen.utils.plot_utils import (
    recover_from_ric,
    plot_3d_motion,
    t2m_kinematic_chain 
)
from scipy.ndimage import gaussian_filter


def motion_temporal_filter(motion, sigma=1):
    motion = motion.reshape(motion.shape[0], -1)
    for i in range(motion.shape[1]):
        motion[:, i] = gaussian_filter(motion[:, i], sigma=sigma, mode="nearest")
    return motion.reshape(motion.shape[0], -1, 3)


def plot_t2m(data, result_path, npy_path, caption):
    joint = recover_from_ric(torch.from_numpy(data).float(), 22).numpy()
    joint = motion_temporal_filter(joint, sigma=2.5)
    plot_3d_motion(result_path, t2m_kinematic_chain, joint, title=caption, fps=20)
    if npy_path is not None:
        np.save(npy_path, joint)


def parse_args():
    parser = argparse.ArgumentParser(description='mogen evaluation')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--text', help='motion description')
    parser.add_argument('--motion_length', type=int, help='expected motion length')
    parser.add_argument('--out', help='output animation file')
    parser.add_argument('--pose_npy', help='output pose sequence file', default=None)
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda'],
        default='cuda',
        help='device used for testing')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    assert args.motion_length >= 16 and args.motion_length <= 196
    
    # build the model and load checkpoint
    model = build_architecture(cfg.model)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    if args.device == 'cpu':
        model = model.cpu()
    else:
        model = MMDataParallel(model, device_ids=[0])
    model.eval()
    
    dataset_name = cfg.data.test.dataset_name
    assert dataset_name == "human_ml3d"
    mean_path = "data/datasets/human_ml3d/mean.npy"
    std_path = "data/datasets/human_ml3d/std.npy"
    mean = np.load(mean_path)
    std = np.load(std_path)
    
    device = args.device
    text = args.text
    motion_length = args.motion_length
    motion = torch.zeros(1, motion_length, 263).to(device)
    motion_mask = torch.ones(1, motion_length).to(device)
    motion_length = torch.Tensor([motion_length]).long().to(device)
    model = model.to(device)
    
    input = {
        'motion': motion,
        'motion_mask': motion_mask,
        'motion_length': motion_length,
        'motion_metas': [{'text': text}],
    }

    all_pred_motion = []
    with torch.no_grad():
        input['inference_kwargs'] = {}
        output_list = []
        output = model(**input)[0]['pred_motion']
        pred_motion = output.cpu().detach().numpy()
        pred_motion = pred_motion * std + mean

    plot_t2m(pred_motion, args.out, args.pose_npy, text)


if __name__ == '__main__':
    main()