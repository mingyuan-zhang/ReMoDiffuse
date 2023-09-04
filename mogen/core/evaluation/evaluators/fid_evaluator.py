import numpy as np
import torch

from ..get_model import get_motion_model
from .base_evaluator import BaseEvaluator
from ..utils import (
    calculate_activation_statistics,
    calculate_frechet_distance)


class FIDEvaluator(BaseEvaluator):
    
    def __init__(self,
                 data_len=0,
                 motion_encoder_name=None,
                 motion_encoder_path=None,
                 batch_size=None,
                 drop_last=False,
                 replication_times=1,
                 replication_reduction='statistics',
                 **kwargs):
        super().__init__(
            replication_times=replication_times,
            replication_reduction=replication_reduction,
            batch_size=batch_size,
            drop_last=drop_last,
            eval_begin_idx=0,
            eval_end_idx=data_len
        )
        self.append_indexes = None
        self.motion_encoder = get_motion_model(motion_encoder_name, motion_encoder_path)
        self.model_list = [self.motion_encoder]
        
    def single_evaluate(self, results):
        results = self.prepare_results(results)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pred_motion = results['pred_motion']
        
        pred_motion_length = results['pred_motion_length']
        pred_motion_mask = results['pred_motion_mask']
        motion = results['motion']
        motion_length = results['motion_length']
        motion_mask = results['motion_mask']
        self.motion_encoder.to(device)
        self.motion_encoder.eval()
        with torch.no_grad():
            pred_motion_emb = self.motion_encode(pred_motion, pred_motion_length, pred_motion_mask, device).cpu().detach().numpy()
            gt_motion_emb = self.motion_encode(motion, motion_length, motion_mask, device).cpu().detach().numpy()
        gt_mu, gt_cov = calculate_activation_statistics(gt_motion_emb)
        pred_mu, pred_cov = calculate_activation_statistics(pred_motion_emb)
        fid = calculate_frechet_distance(gt_mu, gt_cov, pred_mu, pred_cov)
        return fid
        
    def parse_values(self, values):
        metrics = {}
        metrics['FID (mean)'] = values[0]
        metrics['FID (conf)'] = values[1]
        return metrics
