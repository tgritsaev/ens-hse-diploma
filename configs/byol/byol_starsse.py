from dataclasses import dataclass, field
from typing import List
from configs.byol.byol_base import BYOLBaseConfig


@dataclass
class Params(BYOLBaseConfig):
    dataset: str = 'cifar100'
    scheduler: str = 'cosine+warmup_cosine'
    epochs: int = 25
    lr: float = 0.05
    wd: float = 1e-4
    wandb_group: str = 'BYOL_StarSSE'

    num_fge: int = 5
    fge_epochs: int = 12
    fge_lr: float = 0.1
    fge_warmup_epochs: int = 0
    star_fge: bool = True
    
    which_model_is_regularizer: str = "first"
    fmap_betas: List[float] = field(default_factory=lambda: [0., 0., 0., 0., 0.])
    
    wo_beta: float = 0.
    # use_weights_orthogonalizer_first_n_epochs: int = 100
    
    prev_ce_beta: float = 0.
    only_negative: bool = False
