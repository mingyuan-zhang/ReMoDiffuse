from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import torch
import torch.distributed as dist
from mmcv.runner import BaseModule


def to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    return x


class BaseArchitecture(BaseModule):
    """Base class for mogen architecture."""

    def __init__(self, init_cfg=None):
        super(BaseArchitecture, self).__init__(init_cfg)

    def forward_train(self, **kwargs):
        pass

    def forward_test(self, **kwargs):
        pass

    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.
        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.
        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def train_step(self, data, optimizer):
        """The iteration step during training.
        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.
        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.
        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.
                - ``loss`` is a tensor for back propagation, which can be a
                  weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                  logger.
                - ``num_samples`` indicates the batch size (when the model is
                  DDP, it means the batch size on each GPU), which is used for
                  averaging the logs.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['motion']))

        return outputs

    def val_step(self, data, optimizer=None):
        """The iteration step during validation.
        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['motion']))

        return outputs

    def forward(self, **kwargs):
        if self.training:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)
        
    def split_results(self, results):
        B = results['motion'].shape[0]
        output = []
        for i in range(B):
            batch_output = dict()
            batch_output['motion'] = to_cpu(results['motion'][i])
            batch_output['pred_motion'] = to_cpu(results['pred_motion'][i])
            batch_output['motion_length'] = to_cpu(results['motion_length'][i])
            batch_output['motion_mask'] = to_cpu(results['motion_mask'][i])
            if 'pred_motion_length' in results.keys():
                batch_output['pred_motion_length'] = to_cpu(results['pred_motion_length'][i])
            else:
                batch_output['pred_motion_length'] = to_cpu(results['motion_length'][i])
            if 'pred_motion_mask' in results:
                batch_output['pred_motion_mask'] = to_cpu(results['pred_motion_mask'][i])
            else:
                batch_output['pred_motion_mask'] = to_cpu(results['motion_mask'][i])
            if 'motion_metas' in results.keys():
                motion_metas = results['motion_metas'][i]
                if 'text' in motion_metas.keys():
                    batch_output['text'] = motion_metas['text']
                if 'token' in motion_metas.keys():
                    batch_output['token'] = motion_metas['token']
            output.append(batch_output)
        return output
