import os
import copy
from typing import Optional, Union

import numpy as np
from torch.utils.data import Dataset

from .pipelines import Compose
from .builder import DATASETS
from mogen.core.evaluation import build_evaluator


@DATASETS.register_module()
class BaseMotionDataset(Dataset):
    """Base motion dataset.
    Args:
        data_prefix (str): the prefix of data path.
        pipeline (list): a list of dict, where each element represents
            a operation defined in `mogen.datasets.pipelines`.
        ann_file (str | None, optional): the annotation file. When ann_file is
            str, the subclass is expected to read from the ann_file. When
            ann_file is None, the subclass is expected to read according
            to data_prefix.
        test_mode (bool): in train mode or test mode. Default: None.
        dataset_name (str | None, optional): the name of dataset. It is used
            to identify the type of evaluation metric. Default: None.
    """

    def __init__(self,
                 data_prefix: str,
                 pipeline: list,
                 dataset_name: Optional[Union[str, None]] = None,
                 fixed_length: Optional[Union[int, None]] = None,
                 ann_file: Optional[Union[str, None]] = None,
                 motion_dir: Optional[Union[str, None]] = None,
                 eval_cfg: Optional[Union[dict, None]] = None,
                 test_mode: Optional[bool] = False):
        super(BaseMotionDataset, self).__init__()

        self.data_prefix = data_prefix
        self.pipeline = Compose(pipeline)
        self.dataset_name = dataset_name
        self.fixed_length = fixed_length
        self.ann_file = os.path.join(data_prefix, 'datasets', dataset_name, ann_file)
        self.motion_dir = os.path.join(data_prefix, 'datasets', dataset_name, motion_dir)
        self.eval_cfg = copy.deepcopy(eval_cfg)
        self.test_mode = test_mode

        self.load_annotations()
        if self.test_mode:
            self.prepare_evaluation()

    def load_anno(self, name):
        motion_path = os.path.join(self.motion_dir, name + '.npy')
        motion_data = np.load(motion_path)
        return {'motion': motion_data}
        

    def load_annotations(self):
        """Load annotations from ``ann_file`` to ``data_infos``"""
        self.data_infos = []
        for line in open(self.ann_file, 'r').readlines():
            line = line.strip()
            self.data_infos.append(self.load_anno(line))
            

    def prepare_data(self, idx: int):
        """"Prepare raw data for the f'{idx'}-th data."""
        results = copy.deepcopy(self.data_infos[idx])
        results['dataset_name'] = self.dataset_name
        results['sample_idx'] = idx
        return self.pipeline(results)

    def __len__(self):
        """Return the length of current dataset."""
        if self.test_mode:
            return len(self.eval_indexes)
        elif self.fixed_length is not None:
            return self.fixed_length
        return len(self.data_infos)

    def __getitem__(self, idx: int):
        """Prepare data for the ``idx``-th data.
        As for video dataset, we can first parse raw data for each frame. Then
        we combine annotations from all frames. This interface is used to
        simplify the logic of video dataset and other special datasets.
        """
        if self.test_mode:
            idx = self.eval_indexes[idx]
        elif self.fixed_length is not None:
            idx = idx % len(self.data_infos)
        return self.prepare_data(idx)

    def prepare_evaluation(self):
        self.evaluators = []
        self.eval_indexes = []
        for _ in range(self.eval_cfg['replication_times']):
            eval_indexes = np.arange(len(self.data_infos)) 
            if self.eval_cfg.get('shuffle_indexes', False):
                np.random.shuffle(eval_indexes)
            self.eval_indexes.append(eval_indexes)
        for metric in self.eval_cfg['metrics']:
            evaluator, self.eval_indexes = build_evaluator(
                metric, self.eval_cfg, len(self.data_infos), self.eval_indexes)
            self.evaluators.append(evaluator)
        
        self.eval_indexes = np.concatenate(self.eval_indexes)
            
    def evaluate(self, results, work_dir, logger=None):
        metrics = {}
        device = results[0]['motion'].device
        for evaluator in self.evaluators:
            evaluator.to_device(device)
            metrics.update(evaluator.evaluate(results))
        if logger is not None:
            logger.info(metrics)
        return metrics
