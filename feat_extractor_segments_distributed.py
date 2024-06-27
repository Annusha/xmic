import os, argparse
import numpy as np
import torch
import sys
from tqdm import tqdm
import pickle
from pathlib import Path

from lavila import models
from lavila.tokenizer import SimpleTokenizer
from lavila.utils import inflate_positional_embeds
from collections import OrderedDict

import sys
sys.path.append('./Dassl.pytorch/')

# sys.path.append(os.path.abspath(".."))

from datasets.epic_kitchen_segments import EpicKitchenSegments
from datasets.epic_kitchen_segments_blackout_OC_v2 import EpicKitchenSegmentsSpecialOCv2

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.data.transforms import build_transform
from dassl.data import DatasetWrapper, DatasetSegmentWrapper
# from dassl.data import DatasetWrapper, DatasetSegmentWrapper, DatasetSegmentWrapperSpecialOCv2

import clip
from utils.extend_config import extend_cfg, reset_cfg

from utils.dist_utils import init_distributed_mode, is_main_process, get_rank, get_world_size
import torch.multiprocessing as mp
from torch.distributed import init_process_group

# import pdb; pdb.set_trace()


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
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
    args.config_file = f'scripts/configs/{args.config_name}.yaml'
    if args.distributed:
        rank=int(os.environ["LOCAL_RANK"])
        print(f'Setup {rank}', flush=True)
        ddp_setup()
        print(f'Setup finish {rank}', flush=True)
    else:
        rank = 0

    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED + rank * 1000 + rank * 100 + rank * 10 + rank)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    ######################################
    #   Setup DataLoader
    ######################################
    dataset = eval(cfg.DATASET.NAME)(cfg)

    if args.split == "train":
        dataset_input = dataset.train_x
    elif args.split == "validation":
        dataset_input = dataset.val
    else:
        dataset_input = dataset.test

    tfm_train = build_transform(cfg, is_train=False)
    # if args.special_blackout:
    #     dataset = DatasetSegmentWrapperSpecialOCv2(cfg, dataset_input, transform=tfm_train, is_train=False, meta_data_only=True)
    # else:
    dataset = DatasetSegmentWrapper(cfg, dataset_input, transform=tfm_train, is_train=False, meta_data_only=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
        sampler=None,
        shuffle=True,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        drop_last=False,
        pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA),
    )

    ########################################
    #   Setup Network
    ########################################
    if cfg.MODEL.BACKBONE.FRAMEWORK == 'clip':
        design_details = {"trainer": 'CoOp',
                        "vision_depth": 0,
                        "language_depth": 0, "vision_ctx": 0,
                        "language_ctx": 0}

        clip_model, _ = clip.load(cfg.MODEL.BACKBONE.NAME, "cpu", jit=False, design_details=design_details)
        clip_model.cuda().eval()
        print('loaded clip', clip_model)
    if cfg.MODEL.BACKBONE.FRAMEWORK == 'lavila':
        print(f"Loading Lavila (backbone: {cfg.MODEL.BACKBONE.NAME})")
        ckpt_path = cfg.MODEL.BACKBONE.CKPT_PATH

        ckpt = torch.load(ckpt_path, map_location='cpu')

        # create model
        state_dict = OrderedDict()
        for k, v in ckpt['state_dict'].items():
            state_dict[k.replace('module.', '')] = v

        old_args = ckpt['args']
        print('=> creating model: {}'.format(old_args.model))
        clip_model = getattr(models, old_args.model)(
            text_use_cls_token=old_args.use_cls_token,
            project_embed_dim=old_args.project_embed_dim,
            gated_xattn=False if 'gated_xattn' not in old_args else old_args.gated_xattn,
            timesformer_gated_xattn=False if 'timesformer_gated_xattn' not in old_args else old_args.timesformer_gated_xattn,
            timesformer_freeze_space=False if 'timesformer_freeze_space' not in old_args else old_args.timesformer_freeze_space,
            freeze_lm_vclm=False if 'freeze_lm_vclm' not in old_args else old_args.freeze_lm_vclm,
            freeze_visual_vclm=False if 'freeze_visual_vclm' not in old_args else old_args.freeze_visual_vclm,
            num_frames=1,
            drop_path_rate=0,
        )
        clip_model.cuda().eval()

        if 'TIMESFORMER' in old_args.model or 'EGOVLP' in old_args.model:
            # inflate weight
            print('=> inflating PE in models due to different frame numbers')
            state_dict = inflate_positional_embeds(
                clip_model.state_dict(), state_dict,
                num_frames=1,
                load_temporal_fix='bilinear',
            )

        clip_model.load_state_dict(state_dict, strict=True)
        print("=> loaded resume checkpoint '{}' (epoch {}, best_metric = {})".format(cfg.MODEL.BACKBONE.CKPT_PATH, ckpt['epoch'], ckpt['best_acc1']))

    ###################################################################################################################
    # Start Feature Extractor
    feature_list = []
    label_list = []
    train_dataiter = iter(data_loader)
    # output = {}

    root_dir = os.path.join(cfg.OUTPUT_DIR, cfg.DATASET.NAME, f'{args.config_name}')
    save_dir = os.path.join(root_dir, 'segments')
    check_dir = os.path.join(root_dir, 'segments_npy')
    tmp_dir = os.path.join(root_dir, 'tmp')
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(check_dir, exist_ok=True)
    # Path(f'{save_dir}/{rank}_temp').touch(exist_ok=True)
    print('create file', save_dir)

    for train_step in tqdm(range(1, len(train_dataiter) + 1), 'extracting features'):
        batch = next(train_dataiter)
        # breakpoint()
        assert batch["label"].shape[0] == 1, "Batch size should be 1 for feature extraction!"

        narration = batch["narration_id"][0]

        if args.post_process:
            if narration not in ['P22_08_367', 'P22_107_77', 'P27_101_91', 'P35_108_585']:
                continue
            print(narration)


        save_filename = f'{args.split}_{narration}.npy'
        save_local_filename = os.path.join(check_dir, save_filename)

        # save_filename = f'{args.split}_{narration}.pickle'
        # save_local_filename = os.path.join(save_dir, save_filename)

        if os.path.exists(save_local_filename):
            continue

        # try:
        #     batch = dataset.get_item(batch['index'][0])
        # except Exception as e:
        #     print(e, save_local_filename)
        #     continue

        batch = dataset.get_item(batch['index'][0])

        features_batch = []
        output = {}
        # print('batch', batch['img'].shape, flush=True)
        for offset in range(0, batch["img"].shape[0], 128):
            start = offset
            end = min(start + 128, batch['img'].shape[0])
            data = batch["img"][start:end].cuda()
            # print('data', data.shape, flush=True)
            # if len(data.shape) == 5:
            #     b,t,c,h,w = data.shape
            #     if t == 0: continue
            if len(data.shape) == 4:
                t,c,h,w = data.shape
                if t == 0: continue
            # data = data.view(-1, c,h,w)
            with torch.no_grad():
                if cfg.MODEL.BACKBONE.FRAMEWORK == 'clip':
                    feature = clip_model.visual(data)

                if cfg.MODEL.BACKBONE.FRAMEWORK == 'lavila':
                    data = data.unsqueeze(1)
                    data = data.permute(0, 2, 1, 3, 4)
                    feature = clip_model.visual(data)
                    feature = feature @ clip_model.image_projection
            # print('feature', feature.shape, flush=True)
            # feature = feature.view(b,t,-1)
            feature = feature.cpu()
            features_batch.append(feature)

        if len(features_batch):
            features_batch = torch.cat(features_batch, dim=0)
            # output[narration] = features_batch

            # save_filename = f'{args.split}_{narration}.npy'
            # local_output = os.path.join(check_dir, save_filename)

            np.save(save_local_filename, features_batch)

            # with open(save_local_filename, 'wb') as f:
            #     pickle.dump(output, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument("--config-file", type=str, default="", help="path to config file")
    parser.add_argument("--config_name", type=str, default="", help="path to config file")
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--num-shot", type=int, default=1, help="number of shots")
    parser.add_argument("--split", type=str, choices=["train", "validation", "test"], help="which split")
    parser.add_argument("--div", type=int, default=0, help="which split")
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--seed", type=int, default=-1, help="only positive value enables a fixed seed")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument("--special_blackout", action="store_true", help="special blackout for feature extraction")
    parser.add_argument("--post_process", action="store_true", help="extract features only for some subset")

    parser.add_argument("--resume",type=str,default="",help="checkpoint directory (from which the training resumes)",)

    parser.add_argument("--source-domains", type=str, nargs="+", help="source domains for DA/DG")
    parser.add_argument("--target-domains", type=str, nargs="+", help="target domains for DA/DG")
    parser.add_argument("--transforms", type=str, nargs="+", help="data augmentation methods")

    parser.add_argument("--no-train", action="store_true", help="do not call trainer.train()")
    parser.add_argument("--model-dir", type=str,default="", help="load model from this directory for eval-only mode",)
    parser.add_argument("--load-epoch", type=int, help="load model weights at this epoch for evaluation")

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
    parser.add_argument('--slurm_job_name', default='0', type=str)
    parser.add_argument('--tag', default='', type=str)

    parser.add_argument('--multiprocessing-distributed', action='store_true',
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
    main(args)

# export OMP_NUM_THREADS=64

## export NCCL_ASYNC_ERROR_HANDLING=1; torchrun --standalone --nproc_per_node=8 --nnodes 1 feat_extractor_segments_distributed.py --config_name 06_extract_clip_vit_b16_segments_OC_v1 --split train --distributed --seed 42
## torchrun --standalone --nproc_per_node=8 --nnodes 1 feat_extractor_segments_distributed.py --config_name 06_extract_clip_vit_b16_segments_OC_v1 --split validation --distributed
## torchrun --standalone --nproc_per_node=8 --nnodes 1 feat_extractor_segments_distributed.py --config_name 02_extract_clip_vit_b16_segments_center_crops --split train --distributed
## torchrun --standalone --nproc_per_node=8 --nnodes 1 feat_extractor_segments_distributed.py --config_name 02_extract_clip_vit_b16_segments_center_crops --split validation --distributed
## torchrun --standalone --nproc_per_node=8 --nnodes 1 feat_extractor_segments_distributed.py --config_name 03_extract_clip_vit_b16_segments_HCv2 --split train --distributed
## torchrun --standalone --nproc_per_node=8 --nnodes 1 feat_extractor_segments_distributed.py --config_name 03_extract_clip_vit_b16_segments_HCv2 --split validation --distributed

## export OMP_NUM_THREADS=64; torchrun --standalone --nproc_per_node=8 --nnodes 1 feat_extractor_segments_distributed.py --config_name 07_extract_clip_vit_b16_segments_OC_v2 --split train --distributed --special_blackout --seed 45
## export OMP_NUM_THREADS=64; torchrun --standalone --nproc_per_node=8 --nnodes 1 feat_extractor_segments_distributed.py --config_name 07_extract_clip_vit_b16_segments_OC_v2 --split validation --distributed --special_blackout --seed 45

########################################################################
########## uncomment if I need to run hand crops collection ############
########################################################################
# cd /mnt/graphics_ssd/nimble/users/annakukleva/code/epic-kitchens-100-hand-object-bboxes
# python setup.py install
# cd /mnt/graphics_ssd/nimble/users/annakukleva/code/multimodal-prompt-learning

## export OMP_NUM_THREADS=64; export NCCL_ASYNC_ERROR_HANDLING=1; torchrun --standalone --nproc_per_node=8 --nnodes 1 feat_extractor_segments_distributed.py --config_name 10_extract_clip_vitb16_segments_HCv2 --split train --distributed --seed
## export OMP_NUM_THREADS=64; export NCCL_ASYNC_ERROR_HANDLING=1; torchrun --standalone --nproc_per_node=8 --nnodes 1 feat_extractor_segments_distributed.py --config_name 10_extract_clip_vitb16_segments_HCv2 --split validation --distributed --seed

## python feat_extractor_segments_distributed.py --config_name 06_extract_clip_vit_b16_segments_OC_v1_visualization --split train

## 14_extract_clip_vitb16_segments_HCv2_extended_boundaries
## python feat_extractor_segments_distributed.py --config_name 14_extract_clip_vitb16_segments_HCv2_extended_boundaries --split train
## export OMP_NUM_THREADS=64; export NCCL_ASYNC_ERROR_HANDLING=1; torchrun --standalone --nproc_per_node=8 --nnodes 1 feat_extractor_segments_distributed.py --config_name 14_extract_clip_vitb16_segments_HCv2_extended_boundaries --split train --distributed --seed
## export OMP_NUM_THREADS=64; export NCCL_ASYNC_ERROR_HANDLING=1; torchrun --standalone --nproc_per_node=8 --nnodes 1 feat_extractor_segments_distributed.py --config_name 14_extract_clip_vitb16_segments_HCv2_extended_boundaries --split validation --distributed --seed


## 21_extract_lavila_vitb16_segments_HCv2
## python feat_extractor_segments_distributed.py --config_name 21_extract_lavila_vitb16_segments_HCv2 --split train
## export OMP_NUM_THREADS=64; export NCCL_ASYNC_ERROR_HANDLING=1; torchrun --standalone --nproc_per_node=8 --nnodes 1 feat_extractor_segments_distributed.py --config_name 21_extract_lavila_vitb16_segments_HCv2 --split train --distributed --seed
## export OMP_NUM_THREADS=64; export NCCL_ASYNC_ERROR_HANDLING=1; torchrun --standalone --nproc_per_node=8 --nnodes 1 feat_extractor_segments_distributed.py --config_name 21_extract_lavila_vitb16_segments_HCv2 --split validation --distributed --seed

## 27_extract_lavila_vitb16_segments_Full_1frames
## python feat_extractor_segments_distributed.py --config_name 27_extract_lavila_vitb16_segments_Full_1frames --split train
## export OMP_NUM_THREADS=64; export NCCL_ASYNC_ERROR_HANDLING=1; torchrun --standalone --nproc_per_node=8 --nnodes 1 feat_extractor_segments_distributed.py --config_name 27_extract_lavila_vitb16_segments_Full_1frames --split train --distributed --seed
## export OMP_NUM_THREADS=64; export NCCL_ASYNC_ERROR_HANDLING=1; torchrun --standalone --nproc_per_node=8 --nnodes 1 feat_extractor_segments_distributed.py --config_name 27_extract_lavila_vitb16_segments_Full_1frames --split validation --distributed --seed 23


## 32_extract_clip_videos_vitb16_segments
## python feat_extractor_segments_distributed.py --config_name 32_extract_clip_videos_vitb16_segments --split validation
## export OMP_NUM_THREADS=64; export NCCL_ASYNC_ERROR_HANDLING=1; torchrun --standalone --nproc_per_node=8 --nnodes 1 feat_extractor_segments_distributed.py --config_name 32_extract_clip_videos_vitb16_segments --split validation --distributed --seed 23


## 33_extract_clip_videos_vitl14_segments
## python feat_extractor_segments_distributed.py --config_name 33_extract_clip_videos_vitl14_segments --split validation
## export OMP_NUM_THREADS=64; export NCCL_ASYNC_ERROR_HANDLING=1; torchrun --standalone --nproc_per_node=8 --nnodes 1 feat_extractor_segments_distributed.py --config_name 33_extract_clip_videos_vitl14_segments --split validation --distributed --seed 23
## export OMP_NUM_THREADS=64; export NCCL_ASYNC_ERROR_HANDLING=1; torchrun --standalone --nproc_per_node=8 --nnodes 1 feat_extractor_segments_distributed.py --config_name 33_extract_clip_videos_vitl14_segments --split train --distributed --seed 23


## 38_extract_clip_vitl14_segments_HCv2
## python feat_extractor_segments_distributed.py --config_name 38_extract_clip_vitl14_segments_HCv2 --split validation
## export CUDA_VISIBLE_DEVICES=1,2,3,4; export NCCL_ASYNC_ERROR_HANDLING=1; torchrun --standalone --nproc_per_node=4 --nnodes 1 feat_extractor_segments_distributed.py --config_name 38_extract_clip_vitl14_segments_HCv2 --split train --distributed --seed
## export CUDA_VISIBLE_DEVICES=1,2,3,4; export NCCL_ASYNC_ERROR_HANDLING=1; torchrun --standalone --nproc_per_node=4 --nnodes 1 feat_extractor_segments_distributed.py --config_name 38_extract_clip_vitl14_segments_HCv2 --split validation --distributed --seed


## export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

