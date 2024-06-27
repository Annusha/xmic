import os, argparse
import numpy as np
import torch
import sys
from tqdm import tqdm
import pickle
from pathlib import Path

import sys
sys.path.append('./Dassl.pytorch/')

# sys.path.append(os.path.abspath(".."))

from datasets.oxford_pets import OxfordPets
from datasets.oxford_flowers import OxfordFlowers
from datasets.fgvc_aircraft import FGVCAircraft
from datasets.dtd import DescribableTextures
from datasets.eurosat import EuroSAT
from datasets.stanford_cars import StanfordCars
from datasets.food101 import Food101
from datasets.sun397 import SUN397
from datasets.caltech101 import Caltech101
from datasets.ucf101 import UCF101
from datasets.imagenet import ImageNet
from datasets.imagenetv2 import ImageNetV2
from datasets.imagenet_sketch import ImageNetSketch
from datasets.imagenet_a import ImageNetA
from datasets.imagenet_r import ImageNetR
from datasets.epic_kitchen_segments import EpicKitchenSegments

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.data.transforms import build_transform
from dassl.data import DatasetWrapper, DatasetSegmentWrapper

import clip
from utils.extend_config import extend_cfg, reset_cfg

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


def main(args):
    args.config_file = f'scripts/configs/{args.config_name}.yaml'
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
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
    elif args.split == "val":
        dataset_input = dataset.val
    else:
        dataset_input = dataset.test

    tfm_train = build_transform(cfg, is_train=False)
    data_loader = torch.utils.data.DataLoader(
        DatasetSegmentWrapper(cfg, dataset_input, transform=tfm_train, is_train=False),
        batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
        sampler=None,
        shuffle=args.split == "train",
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        drop_last=False,
        pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA),
    )

    ########################################
    #   Setup Network
    ########################################
    design_details = {"trainer": 'CoOp',
                    "vision_depth": 0,
                    "language_depth": 0, "vision_ctx": 0,
                    "language_ctx": 0}

    clip_model, _ = clip.load("ViT-B/32", "cpu", jit=False, design_details=design_details)
    clip_model.cuda().eval()
    ###################################################################################################################
    # Start Feature Extractor
    feature_list = []
    label_list = []
    train_dataiter = iter(data_loader)
    output = {}

    save_dir = os.path.join(cfg.OUTPUT_DIR, cfg.DATASET.NAME, f'{args.config_file}')
    tmp_dir = os.path.join(save_dir, 'tmp2')
    os.makedirs(tmp_dir, exist_ok=True)
    print('should have been created:', tmp_dir)

    for train_step in tqdm(range(1, len(train_dataiter) + 1)):
        batch = next(train_dataiter)
        assert batch["img"].shape[0] == 1, "Batch size should be 1 for feature extraction!"

        tmp_file = os.path.join(tmp_dir, batch["narration_id"][0])
        if os.path.exists(tmp_file):
            continue
        else:
            pass
            # Path(tmp_file).touch(exist_ok=True)


        features_batch = []
        # print('batch', batch['img'].shape, flush=True)
        for offset in range(0, batch["img"].shape[1], 128):
            start = offset
            end = min(start + 128, batch['img'].shape[1])
            data = batch["img"][:, start:end].cuda()
            # print('data', data.shape, flush=True)
            if len(data.shape) == 5:
                b,t,c,h,w = data.shape
                if t == 0: continue
            data = data.view(-1, c,h,w)
            with torch.no_grad():
                feature = clip_model.visual(data)
            # print('feature', feature.shape, flush=True)
            # feature = feature.view(b,t,-1)
            feature = feature.cpu()
            features_batch.append(feature)
        features_batch = torch.cat(features_batch, dim=0)
        output[batch["narration_id"][0]] = features_batch
        # print(f'{batch["narration_id"][0]} {features_batch.shape}')
            # feature_list.append(feature[idx].tolist())
        # label_list.extend(batch["narration_id"])

    save_filename = f"{args.split}_{args.div}.pickle"

    with open(os.path.join(save_dir, save_filename), 'wb') as f:
        pickle.dump(output, f)

    # np.savez(
    #     os.path.join(save_dir, save_filename),
    #     feature_list=feature_list,
    #     label_list=label_list,
    # )


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
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], help="which split")
    parser.add_argument("--div", type=int, default=0, help="which split")
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--seed", type=int, default=-1, help="only positive value enables a fixed seed")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")

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


