import os
import torch
import wandb
from timm.loss import SoftTargetCrossEntropy

import argparser
import models
import utils
from training import train
from utils import set_random_seed, get_scheduler, load_model, build_optimizer, build_scaler, \
    FeatureMapLoss, WeightsOrthogonalizer, PrevCELoss
from data.get_datasets import DS_TO_METRIC_AND_N_CLASSES
from data.transforms_and_loaders import get_dataloaders
from data.swin_transforms import get_mixup_fn


def main(args=None):
    if args is None:
        args = argparser.parse_args()
        Params = argparser.get_config(args)
        args = Params()

    set_random_seed(args.seed)

    # check if working dir exists
    if not os.path.isdir(args.work_dir):
        os.makedirs(args.work_dir, exist_ok=True)

    if args.verbose:
        print('Working dir:', args.work_dir)

    utils.save_config(args)
    # check if gpu training is available
    args.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    if args.verbose:
        print('Using device', args.device)

    # Prepare dataloaders
    loaders = get_dataloaders(args, use_test=args.use_test)
    args.batches_per_epoch = len(loaders['train'])
    metrics_and_n_classes = DS_TO_METRIC_AND_N_CLASSES[args.dataset.lower()]
    args.metric_function = getattr(utils, metrics_and_n_classes[0])
    args.num_classes = metrics_and_n_classes[1]

    if args.verbose:
        print(args.dataset, 'loaded, split and transformed')

    # Initialize model, optimizer (and loss scaler for Swin)
    model = load_model(config=args).to(args.device)
    # finally, we decide not to freeze layers during the first training cycle
    # if len(args.freeze_layers) > 0:
    #     for layer in args.freeze_layers:
    #         print(f"{layer} is frozen.")
    #         for p in getattr(model, layer).parameters():
    #             p.requires_grad = False
    
    optimizer = build_optimizer(config=args, model=model, simmim=getattr(args, 'simmim', False))
    loss_scaler = build_scaler(config=args)
    if args.verbose:
        print('Model and optimizer initialized')

    # Initialize criterion, mixup function and regularization
    criterion = SoftTargetCrossEntropy() if args.use_mixup else torch.nn.CrossEntropyLoss()
    weights_orthogonalizer = WeightsOrthogonalizer(**vars(args))
    fmap_criterion = FeatureMapLoss(model, **vars(args))
    prev_ce_criterion = PrevCELoss(**vars(args))
    mixup_fn = get_mixup_fn(config=args) if args.use_mixup else None

    # Setup regularization if required
    reg_type = getattr(args, 'reg_type', None)
    regularization = None if reg_type is None else getattr(models.reg, reg_type + 'Regularization')(args.num_points)

    if args.verbose:
        print(f'Using {optimizer}')
        if getattr(args, 'ckpt', None) is not None:
            print(f'Loaded pre-trained model {args.model} from {args.ckpt}')
        else:
            print(f'Initialized random model {args.model}')

    scheduler = None
    if hasattr(args, 'scheduler') and args.scheduler is not None:
        scheduler = get_scheduler(optimizer, args)

    if args.verbose:
        scheduler_name = args.scheduler if scheduler else 'constant'
        print(f'{scheduler_name} scheduler is initialized')

    if args.use_wandb:
        utils.init_wandb(args)
        wandb.watch(model)

    train(
        model=model, 
        optimizer=optimizer, 
        criterion=criterion, 
        weights_orthogonalizer=weights_orthogonalizer,
        fmap_criterion=fmap_criterion, 
        prev_ce_criterion=prev_ce_criterion,
        regularization=regularization, 
        scheduler=scheduler,
        train_loader=loaders['train'], 
        valid_loader=loaders['test'], 
        args=args, 
        mixup_fn=mixup_fn, 
        loss_scaler=loss_scaler
    )

    if args.use_wandb:
        wandb_summary = dict(wandb.run.summary)
        wandb.finish()
        return wandb_summary


if __name__ == "__main__":
    main()
