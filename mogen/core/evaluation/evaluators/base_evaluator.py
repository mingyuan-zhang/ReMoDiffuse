import torch
import numpy as np
from ..utils import get_metric_statistics


class BaseEvaluator(object):
    
    def __init__(self,
                 batch_size=None,
                 drop_last=False,
                 replication_times=1,
                 replication_reduction='statistics',
                 eval_begin_idx=None,
                 eval_end_idx=None):
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.replication_times = replication_times
        self.replication_reduction = replication_reduction
        assert replication_reduction in ['statistics', 'mean', 'concat']
        self.eval_begin_idx = eval_begin_idx
        self.eval_end_idx = eval_end_idx
    
    def evaluate(self, results):
        total_len = len(results)
        partial_len = total_len // self.replication_times
        all_metrics = []
        for replication_idx in range(self.replication_times):
            partial_results = results[
                replication_idx * partial_len: (replication_idx + 1) * partial_len]
            if self.batch_size is not None:
                batch_metrics = []
                for batch_start in range(self.eval_begin_idx, self.eval_end_idx, self.batch_size):
                    batch_results = partial_results[batch_start: batch_start + self.batch_size]
                    if len(batch_results) < self.batch_size and self.drop_last:
                        continue
                    batch_metrics.append(self.single_evaluate(batch_results))
                all_metrics.append(self.concat_batch_metrics(batch_metrics))
            else:
                batch_results = partial_results[self.eval_begin_idx: self.eval_end_idx]
                all_metrics.append(self.single_evaluate(batch_results))
        all_metrics = np.stack(all_metrics, axis=0)
        if self.replication_reduction == 'statistics':
            values = get_metric_statistics(all_metrics, self.replication_times)
        elif self.replication_reduction == 'mean':
            values = np.mean(all_metrics, axis=0)
        elif self.replication_reduction == 'concat':
            values = all_metrics
        return self.parse_values(values)
    
    def prepare_results(self, results):
        text = []
        pred_motion = []
        pred_motion_length = []
        pred_motion_mask = []
        motion = []
        motion_length = []
        motion_mask = []
        token = []
        # count the maximum motion length
        T = max([result['motion'].shape[0] for result in results])
        for result in results:
            cur_motion = result['motion']
            if cur_motion.shape[0] < T:
                padding_values = torch.zeros((T - cur_motion.shape[0], cur_motion.shape[1]))
                padding_values = padding_values.type_as(pred_motion)
                cur_motion = torch.cat([cur_motion, padding_values], dim=0)
            motion.append(cur_motion)
            cur_pred_motion = result['pred_motion']
            if cur_pred_motion.shape[0] < T:
                padding_values = torch.zeros((T - cur_pred_motion.shape[0], cur_pred_motion.shape[1]))
                padding_values = padding_values.type_as(cur_pred_motion)
                cur_pred_motion = torch.cat([cur_pred_motion, padding_values], dim=0)
            pred_motion.append(cur_pred_motion)
            cur_motion_mask = result['motion_mask']
            if cur_motion_mask.shape[0] < T:
                padding_values = torch.zeros((T - cur_motion_mask.shape[0]))
                padding_values = padding_values.type_as(cur_motion_mask)
                cur_motion_mask= torch.cat([cur_motion_mask, padding_values], dim=0)
            motion_mask.append(cur_motion_mask)
            cur_pred_motion_mask = result['pred_motion_mask']
            if cur_pred_motion_mask.shape[0] < T:
                padding_values = torch.zeros((T - cur_pred_motion_mask.shape[0]))
                padding_values = padding_values.type_as(cur_pred_motion_mask)
                cur_pred_motion_mask= torch.cat([cur_pred_motion_mask, padding_values], dim=0)
            pred_motion_mask.append(cur_pred_motion_mask)
            motion_length.append(result['motion_length'].item())
            pred_motion_length.append(result['pred_motion_length'].item())
            if 'text' in result.keys():  
                text.append(result['text'])
            if 'token' in result.keys():
                token.append(result['token'])
            
        motion = torch.stack(motion, dim=0)
        pred_motion = torch.stack(pred_motion, dim=0)
        motion_mask = torch.stack(motion_mask, dim=0)
        pred_motion_mask = torch.stack(pred_motion_mask, dim=0)
        motion_length = torch.Tensor(motion_length).to(motion.device).long()
        pred_motion_length = torch.Tensor(pred_motion_length).to(motion.device).long()
        output = {
            'pred_motion': pred_motion,
            'pred_motion_mask': pred_motion_mask,
            'pred_motion_length': pred_motion_length,
            'motion': motion,
            'motion_mask': motion_mask,
            'motion_length': motion_length,
            'text': text,
            'token': token
        }
        return output

    def to_device(self, device):
        for model in self.model_list:
            model.to(device)

    def motion_encode(self, motion, motion_length, motion_mask, device):
        N = motion.shape[0]
        motion_emb = []
        batch_size = 32
        cur_idx = 0
        with torch.no_grad():
            while cur_idx < N:
                cur_motion = motion[cur_idx: cur_idx + batch_size].to(device)
                cur_motion_length = motion_length[cur_idx: cur_idx + batch_size].to(device)
                cur_motion_mask = motion_mask[cur_idx: cur_idx + batch_size].to(device)
                cur_motion_emb = self.motion_encoder(cur_motion, cur_motion_length, cur_motion_mask)
                motion_emb.append(cur_motion_emb)
                cur_idx += batch_size
        motion_emb = torch.cat(motion_emb, dim=0)
        return motion_emb

    def text_encode(self, text, token, device):
        N = len(text)
        text_emb = []
        batch_size = 32
        cur_idx = 0
        with torch.no_grad():
            while cur_idx < N:
                cur_text = text[cur_idx: cur_idx + batch_size]
                cur_token = token[cur_idx: cur_idx + batch_size]
                cur_text_emb = self.text_encoder(cur_text, cur_token, device)
                text_emb.append(cur_text_emb)
                cur_idx += batch_size
        text_emb = torch.cat(text_emb, dim=0)
        return text_emb