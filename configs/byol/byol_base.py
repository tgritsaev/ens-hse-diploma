import os
import numpy as np
from dataclasses import dataclass, field
from typing import List


@dataclass
class BYOLBaseConfig:
    # Seed
    seed: int = None
    
    # logging
    verbose: bool = True
    use_wandb: bool = True

    # Checkpoint and directories params
    work_dir: str = './experiments/'
    save_path: str = None
    ckpt: str = './checkpoints/byol-1000ep-1.pt'
    save_logits: bool = True

    # Model, dataset and augmentation params
    dataset: str = 'cifar100'
    data_path: str = './datasets/'
    model: str = 'ImageNetResNet50'
    transform_name: str = 'imageNet'
    interpolation: str = 'bilinear'
    use_test: bool = True
    use_mixup: bool = False
    num_workers: int = 4

    # Training params
    epochs: int = 1
    start_epoch: int = 1
    batch_size: int = 256

    # Optimizer params
    optimizer: str = 'sgd'
    lr: float = 0.05
    wd: float = 5e-4
    momentum: float = 0.9
    scheduler: str = 'cosine'
    nesterov: bool = False

    # Optimization utils params
    loss_scaler: str = None
    amp_enable: bool = False
    clip_grad: bool = False

    # FGE/SSE params
    num_fge: int = 1
    fge_epochs: int = None
    fge_lr: float = None
    fge_warmup_epochs: int = None
    star_fge: bool = False
    reset_optim_state: bool = True

    # Logging params
    valid_freq: int = 4
    wandb_log_rate: int = 30
    wandb_project: str = 'EnsForTransfer'
    wandb_group: str = 'BYOL_Test'
    exp_name: str = ''
    
    # RTD regularization    
    which_model_is_regularizer: str = "first"
    fmap_betas: List[float] = field(default_factory=lambda: [0., 0., 0., 0., 0.])
    fmap_loss_type: str = "RTDLoss"
    
    # Diversity regularization using weights
    wo_beta: float = 0.
    # use_weights_orthogonalizer_first_n_epochs: int = 100
    
    # Diversity regularization using cross-entropy
    prev_ce_beta: float = 0.
    only_negative: bool = False
    
    # Freeze chosen layers
    freeze_layers: List[str] = field(default_factory=lambda: [])
        
    # Save all intermediate models
    save_all_models: bool = False

    def __post_init__(self):
        if self.seed is None:
            self.seed = np.random.randint(1000000)
    
        print("wandb_group:", self.wandb_group, flush=True)
        self.work_dir = os.path.join(
            self.work_dir, self.dataset, self.wandb_group,
            self.model.lower() + '_seed=' + str(self.seed) + '/'
        )

        self.save_path = \
            f'{self.work_dir}model={self.model}_dataset_={self.dataset}_sched={self.scheduler}_' \
            f'lr={self.lr}_wd={self.wd}_epoch={self.epochs}.pt'
