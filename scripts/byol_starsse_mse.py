import sys
import numpy as np

sys.path.append('.')
from main import main
from configs.byol.byol_starsse import Params


PT_CKPTS = [
    './checkpoints/byol-1000ep-1.pt',
    './checkpoints/byol-1000ep-2.pt',
]
NUM_RUNS = 3

seeds = np.random.choice(100000, size=[len(PT_CKPTS), NUM_RUNS], replace=False)
for i, ckpt in enumerate(PT_CKPTS):
    for j in range(NUM_RUNS):
        beta = 10
        x_lr = 4
        config = Params(
            seed=int(seeds[i, j]),
            ckpt=ckpt,
            wandb_group=f"BYOL_StarSSE-MSE-beta={beta}",
        )
        
        config.data_path = "/opt/software/datasets/cifar"
        
        config.fge_lr = x_lr * config.lr
        config.fge_epochs = int(0.5 * config.epochs)
        
        config.fmap_loss_type = "MSELoss"
        config.fmap_betas = [0, 0, 0, 0, beta]
        
        main(config)