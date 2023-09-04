import copy
import numpy as np
from mmcv.utils import Registry
from .evaluators.precision_evaluator import PrecisionEvaluator
from .evaluators.matching_score_evaluator import MatchingScoreEvaluator
from .evaluators.fid_evaluator import FIDEvaluator
from .evaluators.diversity_evaluator import DiversityEvaluator
from .evaluators.multimodality_evaluator import MultiModalityEvaluator

EVALUATORS = Registry('evaluators')

EVALUATORS.register_module(name='R Precision', module=PrecisionEvaluator)
EVALUATORS.register_module(name='Matching Score', module=MatchingScoreEvaluator)
EVALUATORS.register_module(name='FID', module=FIDEvaluator)
EVALUATORS.register_module(name='Diversity', module=DiversityEvaluator)
EVALUATORS.register_module(name='MultiModality', module=MultiModalityEvaluator)


def build_evaluator(metric, eval_cfg, data_len, eval_indexes):
    cfg = copy.deepcopy(eval_cfg)
    cfg.update(metric)
    cfg.pop('metrics')
    cfg['data_len'] = data_len
    cfg['eval_indexes'] = eval_indexes
    evaluator = EVALUATORS.build(cfg)
    if evaluator.append_indexes is not None:
        for i in range(eval_cfg['replication_times']):
            eval_indexes[i] = np.concatenate((eval_indexes[i], evaluator.append_indexes[i]), axis=0)
    return evaluator, eval_indexes
