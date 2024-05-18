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
        x_lr = 4
        config = Params(
            seed=int(seeds[i, j]),
            ckpt=ckpt,
            wandb_group=f"BYOL_FreezeStarSSE_before_layer3_xLR={x_lr}",
        )
        
        config.data_path = "/opt/software/datasets/cifar"
        
        config.fge_lr = x_lr * config.lr
        config.fge_epochs = int(0.5 * config.epochs)
        
        # Define frozen layers
        config.freeze_layers = [
            "conv1",
            "bn1",
            "relu",
            "maxpool",
            "layer1",
            "layer2",
            # "layer3",
            # "layer4",
        ]
        
        main(config)