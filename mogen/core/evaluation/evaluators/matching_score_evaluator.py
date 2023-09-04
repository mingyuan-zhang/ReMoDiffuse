import numpy as np
import torch

from ..get_model import get_motion_model, get_text_model
from .base_evaluator import BaseEvaluator
from ..utils import calculate_top_k, euclidean_distance_matrix


class MatchingScoreEvaluator(BaseEvaluator):
    
    def __init__(self,
                 data_len=0,
                 text_encoder_name=None,
                 text_encoder_path=None,
                 motion_encoder_name=None,
                 motion_encoder_path=None,
                 top_k=3,
                 batch_size=32,
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
        self.text_encoder = get_text_model(text_encoder_name, text_encoder_path)
        self.motion_encoder = get_motion_model(motion_encoder_name, motion_encoder_path)
        self.top_k = top_k
        self.model_list = [self.text_encoder, self.motion_encoder]
        
    def single_evaluate(self, results):
        results = self.prepare_results(results)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        motion = results['motion']
        pred_motion = results['pred_motion']
        pred_motion_length = results['pred_motion_length']
        pred_motion_mask = results['pred_motion_mask']
        text = results['text']
        token = results['token']
        self.text_encoder.to(device)
        self.motion_encoder.to(device)
        self.text_encoder.eval()
        self.motion_encoder.eval()
        with torch.no_grad():
            word_emb = self.text_encode(text, token, device=device).cpu().detach().numpy()
            motion_emb = self.motion_encode(pred_motion, pred_motion_length, pred_motion_mask, device).cpu().detach().numpy()
            dist_mat = euclidean_distance_matrix(word_emb, motion_emb)
            matching_score = dist_mat.trace()
            all_size = word_emb.shape[0]
        return matching_score, all_size
    
    def concat_batch_metrics(self, batch_metrics):
        matching_score_sum = 0
        all_size = 0
        for batch_matching_score, batch_all_size in batch_metrics:
            matching_score_sum += batch_matching_score
            all_size += batch_all_size
        matching_score = matching_score_sum / all_size
        return matching_score
        
    def parse_values(self, values):
        metrics = {}
        metrics['Matching Score (mean)'] = values[0]
        metrics['Matching Score (conf)'] = values[1]
        return metrics
