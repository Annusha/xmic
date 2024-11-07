import argparse
import torch

import os
import sys
sys.path.append('./Dassl.pytorch/')
sys.path.append('../C1-Action-Recognition/')
sys.path.append('../epic-kitchens-100-hand-object-bboxes')

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

# custom
# import datasets.oxford_pets
# import datasets.oxford_flowers
# import datasets.fgvc_aircraft
# import datasets.dtd
# import datasets.eurosat
# import datasets.stanford_cars
# import datasets.food101
# import datasets.sun397
# import datasets.caltech101
# import datasets.ucf101
# import datasets.imagenet
# import datasets.imagenet_sketch
# import datasets.imagenetv2
# import datasets.imagenet_a
# import datasets.imagenet_r


import datasets.epic_kitchen
import datasets.epic_kitchen_segments
# import dassl.data.data_manager_ego4d
import datasets.epic_kitchen_segments_all_label_types
import datasets.egtea
import datasets.egoclip_features


import trainers.coop
import trainers.cocoop
import trainers.zsclip
import trainers.temporal_maple
import trainers.independentVL
import trainers.vpt
import trainers.clip_ft
import trainers.temporal_clip_ft
import trainers.vis_temporal_clip_ft
import trainers.temporal_coop
import trainers.zslavila_segments
import trainers.ego_coop
import trainers.ego_detic_coop
import trainers.clip_ft_contrastive
import trainers.temporal_cocoop
import trainers.lavila_ft
import trainers.temporal_decomp_cocoop
import trainers.temporal_decomp_cocoop_wo_ctx
import trainers.temporal_decomp_addition
import trainers.zsblip_segments
import trainers.temporal_decomp_cocoop_wo_text_ctx_precompute
import trainers.decCC_lavila_clip_blip_wo_text_ctx_precompute
import trainers.lavilazs_segments2
import trainers.decCC_lavila_clip_blip_wo_text_ctx_precompute_w_narrations
import trainers.decCC_lavila_clip_blip_wo_text_ctx_precompute_w_narrations2
import trainers.help_hand_lavila_clip_wo_text_ctx_precompute_w_narrations
import trainers.maple
import trainers.temporal_maple_with_ctx
import trainers.xmic

# from sacred import Experiment

import neptune
import warnings
import numpy as np
import random
# from neptune.new.integrations.sacred import NeptuneObserver

from utils.dist_utils import init_distributed_mode, is_main_process, get_rank, get_world_size
from utils.extend_config import extend_cfg, reset_cfg

import torch.multiprocessing as mp
from torch.distributed import init_process_group

# ex = Experiment('train', save_git_info=False)

# LOCAL_RANK = int(os.environ['LOCAL_RANK'])
# WORLD_SIZE = int(os.environ['WORLD_SIZE'])
# WORLD_RANK = int(os.environ['RANK'])

def print_args(args, cfg):
    print("***************", flush=True)
    print("** Arguments **", flush=True)
    print("***************", flush=True)
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]), flush=True)
    print("************", flush=True)
    print("** Config **", flush=True)
    print(f"** {cfg.NAME} **", flush=True)
    print("************", flush=True)
    print(cfg)



def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    # if args.dataset_config_file:
    #     cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def ddp_setup():
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    # os.environ["MASTER_ADDR"] = "d38dc1e7d770"
    # os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if rank == 0:
            builtin_print(*args, **kwargs)

    __builtin__.print = print



def main(args):
    # init_distributed_mode(args=args)
    rank=int(os.environ["LOCAL_RANK"])
    print(f'Setup {rank}', flush=True)
    ddp_setup()
    print(f'Setup finish {rank}', flush=True)

    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

        seed = cfg.SEED + rank
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

    setup_logger(cfg.OUTPUT_DIR)

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    # torch.multiprocessing.set_start_method('spawn')

    trainer = build_trainer(cfg)

    if args.neptune and is_main_process():
        print('RANK', rank, get_world_size())

        api_token = 'your_api'
        trainer.init_neptune(
            project='prompt-learning/baselines',
            api_token=api_token,
            with_id=args.with_neptune_id,
            mode=args.neptune_mode,
            tags=[args.slurm_id, args.slurm_job_name, args.tag])
        trainer.neptune_log_cfg(cfg)
        trainer.neptune_log_args(args)


    if args.eval_only:
        epochs2load = [int(i) for i in args.load_epoch.split(',')]
        for epoch in epochs2load:
            trainer.load_model(args.model_dir, epoch=epoch)
            trainer.test(split=args.eval_split)
        return

    if not args.no_train:
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--eval_split", type=str, default="val", help="evaluation split")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument("--resume",type=str,default="",help="checkpoint directory (from which the training resumes)",)
    parser.add_argument("--seed", type=int, default=1, help="only positive value enables a fixed seed")
    parser.add_argument("--source-domains", type=str, nargs="+", help="source domains for DA/DG")
    parser.add_argument("--target-domains", type=str, nargs="+", help="target domains for DA/DG")
    parser.add_argument("--transforms", type=str, nargs="+", help="data augmentation methods")
    parser.add_argument('-c', "--config-file", type=str, default="", help="path to config file")
    parser.add_argument("--dataset-config-file",type=str,default="",help="path to config file for dataset setup",)
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument("--no-train", action="store_true", help="do not call trainer.train()")
    parser.add_argument("--model-dir", type=str,default="", help="load model from this directory for eval-only mode",)
    parser.add_argument("--load-epoch", type=str, default='1', help="load model weights at this epoch for evaluation")

    parser.add_argument('-n', '--neptune', action='store_true',
                        help='Whether to observe (neptune)')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--resume_chkp', default='')
    parser.add_argument('--world_size', default=-1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', action="store_true")
    parser.add_argument('--force', action="store_true")
    parser.add_argument('--with_neptune_id', default=None, type=str)
    parser.add_argument('--neptune_mode', default='offline', type=str)
    parser.add_argument('--slurm_id', default='0', type=str)
    parser.add_argument('--init_method', default='', type=str)
    parser.add_argument('--slurm_job_name', default='0', type=str)
    parser.add_argument('--tag', default='', type=str)

    parser.add_argument('--multiprocessing_distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training') #python -m torch.distributed.launch --nproc_per_node=
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')

    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, help="modify config options using the command-line",)

    args = parser.parse_args()

    print('Init distributed mode', flush=True)
    # if args.dist_url == "env://" and args.world_size == -1:
    #     args.world_size = int(os.environ["WORLD_SIZE"])
    #     print('Set world size: ', args.world_size, flush=True)

    # args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    # print('Distributed', args.distributed, args.world_size, args.multiprocessing_distributed)
    # ngpus_per_node = args.world_size
    # if args.multiprocessing_distributed:
    #     # Since we have ngpus_per_node processes per node, the total world_size
    #     # needs to be adjusted accordingly
    #     # args.world_size = ngpus_per_node * args.world_size
    #     # Use torch.multiprocessing.spawn to launch distributed processes: the
    #     # main_worker process function
    #     print('world size', args.world_size, flush=True)
    #     mp.spawn(main, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    # else:
    #     # Simply call main_worker function
    #     run_worker(args.gpu, ngpus_per_node, config)

    main(args)
