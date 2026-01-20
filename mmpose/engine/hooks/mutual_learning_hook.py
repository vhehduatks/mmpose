# Copyright (c) OpenMMLab. All rights reserved.
"""
Progressive Warmup Hook for Dual Backbone Mutual Learning

This hook updates the current_epoch in the head module before each epoch,
enabling progressive warmup for mutual learning loss.

Usage in config:
    custom_hooks = [
        dict(type='MutualLearningWarmupHook')
    ]
"""
from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class MutualLearningWarmupHook(Hook):
    """Progressive warmup hook for dual backbone mutual learning.

    This hook updates the `current_epoch` attribute in the head module
    before each training epoch. The head uses this to compute the
    progressive warmup weight for mutual learning loss.

    The warmup schedule is controlled by head parameters:
    - mutual_warmup_epochs: epochs with no mutual learning (weight=0)
    - mutual_rampup_epochs: epochs to linearly increase weight (0->1)

    Example:
        >>> # In config file
        >>> custom_hooks = [
        ...     dict(type='MutualLearningWarmupHook')
        ... ]
        >>> model = dict(
        ...     head=dict(
        ...         type='CustomxRegoposeBaselinel1_multi_backbone_v2',
        ...         mutual_warmup_epochs=5,
        ...         mutual_rampup_epochs=10,
        ...     )
        ... )

    Schedule example (warmup=5, rampup=10):
        Epoch 0-4:  weight = 0.0  (no mutual learning)
        Epoch 5:    weight = 0.0
        Epoch 6:    weight = 0.1
        Epoch 10:   weight = 0.5
        Epoch 15+:  weight = 1.0  (full mutual learning)
    """

    priority = 'NORMAL'

    def before_train_epoch(self, runner) -> None:
        """Update current_epoch in the head before each training epoch.

        Args:
            runner: The runner of the training process.
        """
        epoch = runner.epoch
        model = runner.model

        # Handle DDP wrapper
        if hasattr(model, 'module'):
            model = model.module

        # Update head's current_epoch
        head = model.head
        if hasattr(head, 'current_epoch'):
            head.current_epoch = epoch

            # Log the mutual learning weight
            if hasattr(head, 'get_mutual_loss_weight'):
                weight = head.get_mutual_loss_weight()
                runner.logger.info(
                    f'[MutualLearningWarmupHook] Epoch {epoch}: '
                    f'mutual_learning_weight = {weight:.3f}'
                )
