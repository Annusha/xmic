import os

def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COOP.CSC = False  # class-specific context
    cfg.TRAINER.COOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COOP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COCOOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COCOOP.PREC = "fp16"  # fp16, fp32, amp

    # Config for MaPLe
    cfg.TRAINER.MAPLE = CN()
    cfg.TRAINER.MAPLE.N_CTX = 2  # number of context vectors
    cfg.TRAINER.MAPLE.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.MAPLE.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.MAPLE.PROMPT_DEPTH = 9 # Max 12, minimum 0, for 1 it will act as shallow MaPLe (J=1)
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new
    cfg.TRAINER.MAPLE.CTX = 'maple' # maple clip dino

    # Config for independent Vision Language prompting (independent-vlp)
    cfg.TRAINER.IVLP = CN()
    cfg.TRAINER.IVLP.N_CTX_VISION = 2  # number of context vectors at the vision branch
    cfg.TRAINER.IVLP.N_CTX_TEXT = 2  # number of context vectors at the language branch
    cfg.TRAINER.IVLP.CTX_INIT = "a photo of a"  # initialization words (only for language prompts)
    cfg.TRAINER.IVLP.PREC = "fp16"  # fp16, fp32, amp
    # If both variables below are set to 0, 0, will the config will degenerate to COOP model
    cfg.TRAINER.IVLP.PROMPT_DEPTH_VISION = 9 # Max 12, minimum 0, for 0 it will act as shallow MaPLe (J=1)
    cfg.TRAINER.IVLP.PROMPT_DEPTH_TEXT = 9  # Max 12, minimum 0, for 0 it will act as shallow MaPLe (J=1)
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

    # Config for only vision side prompting
    cfg.TRAINER.VPT = CN()
    cfg.TRAINER.VPT.N_CTX_VISION = 2  # number of context vectors at the vision branch
    cfg.TRAINER.VPT.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.VPT.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.VPT.PROMPT_DEPTH_VISION = 1  # if set to 1, will represent shallow vision prompting only
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new


    ########## NEW ##########
    cfg.DATASET.LABEL_TYPE = 'noun'
    cfg.DATASET.LABEL_SUBTYPES = 'noun,verb,action'
    cfg.DATASET.SUBSET = 'all'
    cfg.MODEL.EXTRACT_FEATURES = False
    cfg.DATALOADER.SEGMENTS = False
    cfg.DATALOADER.FRAMES_PER_SEGMENT = 4
    cfg.TEST.LT_EVAL = False
    cfg.DATALOADER.NORMALIZE = False
    cfg.MODEL.BACKBONE.CKPT_PATH = ''
    cfg.DATALOADER.USE_EXTRACTED_FEATURES = False
    cfg.DATALOADER.USE_DINO_EXTRACTED_FEATURES = False
    cfg.DATALOADER.USE_DINO_EXTRACTED_FEATURES_ONLY = False
    cfg.DATALOADER.USE_OBJECTS_FEATURES = False
    cfg.DATALOADER.FEATURES_MODEL = 'CLIP-vit-b-16'
    cfg.DATALOADER.LAVILA_FEATURES = False
    cfg.DATALOADER.FEATURES_NAME = ''
    cfg.DATALOADER.OBJECT_FEATURES_NAME = ''
    cfg.DATALOADER.FEATURES_DIM = 512
    cfg.TRAINER.COOP.WITHOUT_CLASSNAMES = False
    cfg.DATALOADER.EXTRACT_FEAT_WITH_OFFSET_SECS = 0
    cfg.DATALOADER.USE_FEAT_WITH_OFFSET_SECS = 0
    cfg.TRAIN.CHECKPOINT_EARLY_EPOCHS = 0
    cfg.TRAIN.CHECKPOINT_EVERY_LAST_EPOCH = False


    cfg.TRAINER.EGO_COOP = CN()
    cfg.TRAINER.EGO_COOP.SUBCLASSNAMES_SC = False  # specific context for subclass names
    cfg.TRAINER.EGO_COOP.WITH_SUBCLASSES = False  # use additional loss for subclasses
    cfg.TRAINER.EGO_COOP.N_SUBCLASSES = 500
    cfg.TRAINER.EGO_COOP.SUBCLASS_LOSS_WEIGHT = 1.0
    cfg.TRAINER.EGO_COOP.TEMPERATURE = -1.0
    cfg.TRAINER.EGO_COOP.WITHOUT_CLASS_LOSS = False

    cfg.TRAINER.EGO_COOP.OBJECTS = CN()
    cfg.TRAINER.EGO_COOP.OBJECTS.CONCAT_IMG_AND_OBJS = False
    cfg.TRAINER.EGO_COOP.OBJECTS.IMG_AND_OBJS_TEMP_AVG = True # do not temp average object predictions over frames, should be false when frame-level-bce = True
    cfg.TRAINER.EGO_COOP.OBJECTS.SEGMENT_LEVEL_BCE = True
    cfg.TRAINER.EGO_COOP.OBJECTS.FRAME_LEVEL_BCE = False
    cfg.TRAINER.EGO_COOP.OBJECTS.USE_BCE = False
    cfg.TRAINER.EGO_COOP.OBJECTS.BCE_THD = 0.28
    cfg.TRAINER.EGO_COOP.OBJECTS.BCE_WEIGHT = 0.1
    cfg.TRAINER.EGO_COOP.OBJECTS.FRAME_LEVEL_BCE_WEIGHT = 0.1

    cfg.TRAINER.DECOMP_COCOOP = CN()
    cfg.TRAINER.DECOMP_COCOOP.VISUAL_WITH_CTX = False
    cfg.TRAINER.DECOMP_COCOOP.VISUAL_N_CTX = 0  # will be the same as text ctx
    cfg.TRAINER.DECOMP_COCOOP.VISUAL_CTX_NEW = False     # visual ctx use its own prompts
    cfg.TRAINER.DECOMP_COCOOP.MLP_LY = 2     # how many layers in MLP (meta-net)
    cfg.TRAINER.DECOMP_COCOOP.MLP_LY_VID2VID = False    #
    cfg.TRAINER.DECOMP_COCOOP.VID2VID = False    #
    cfg.TRAINER.DECOMP_COCOOP.VID2VID_NORM = True    #
    cfg.TRAINER.DECOMP_COCOOP.MLP_LY_TXT2VID = False    #
    cfg.TRAINER.DECOMP_COCOOP.MLP_LY_TXT2TXT = False    #
    cfg.TRAINER.DECOMP_COCOOP.MLP_LY_TXT2TXT_4NARR = False    #
    cfg.TRAINER.DECOMP_COCOOP.MLP_LY_TXT2TXT_4NARR_NORM = False    #
    cfg.TRAINER.DECOMP_COCOOP.VISUAL_CTX_NORM = False     # NORM_BEFORE_ADD
    cfg.TRAINER.DECOMP_COCOOP.TEXTUAL_NORM = True     # NORM_BEFORE_ADD
    cfg.TRAINER.DECOMP_COCOOP.VISUAL_NORM = True     # NORM_BEFORE_ADD
    cfg.TRAINER.DECOMP_COCOOP.TEXTUAL_CTX_NORM = True     # NORM_BEFORE_ADD
    cfg.TRAINER.DECOMP_COCOOP.TEXTUAL_NORM2 = True     # NORM_BEFORE_ADD
    cfg.TRAINER.DECOMP_COCOOP.SKIP_TEXT_ENCODER = False
    cfg.TRAINER.DECOMP_COCOOP.FIXED_SCALE_FACTOR_VIS_CTX = 1.0
    cfg.TRAINER.DECOMP_COCOOP.SCALE_FACTOR_V2T = 1.0
    cfg.TRAINER.DECOMP_COCOOP.SCALE_FACTOR_T2V = 1.0
    cfg.TRAINER.DECOMP_COCOOP.LEARNABLE_SCALE_FACTOR_VIS_CTX = False
    cfg.TRAINER.DECOMP_COCOOP.TEXT_CONDITIONING = True
    cfg.TRAINER.DECOMP_COCOOP.VIS_CONDITIONING = False
    cfg.TRAINER.DECOMP_COCOOP.USE_LLAMA_TXT = False
    cfg.TRAINER.DECOMP_COCOOP.CLIP_ADAPTER = False
    cfg.TRAINER.DECOMP_COCOOP.CLIP_ADAPTER_RATIO = 0.2
    cfg.TRAINER.DECOMP_COCOOP.LLAMA_DIM = 5120
    # cfg.TRAINER.DECOMP_COCOOP.NARRATIONS = False
    cfg.TRAINER.DECOMP_COCOOP.CLIPLOSS_W = 1.0
    cfg.TRAINER.DECOMP_COCOOP.MLP_NARRATIONS = False # additonal manipulation on the visual side
    cfg.TRAINER.DECOMP_COCOOP.MLP_NARRATIONS2 = False # manipulations on the text side
    cfg.TRAINER.DECOMP_COCOOP.MLP_NARRATIONS2_NORM = False # manipulations on the text side
    cfg.TRAINER.DECOMP_COCOOP.WITH_RELU = False
    cfg.TRAINER.DECOMP_COCOOP.PROMPT_PREFIX = ''
    cfg.TRAINER.DECOMP_COCOOP.VIS_CTX_INIT_NORM = True


    cfg.TEST.TSNE = CN()
    cfg.TEST.TSNE.SAVE = False
    cfg.TEST.TSNE.NAME = ''
    cfg.TEST.TSNE.NOUN = ''
    cfg.TEST.TSNE.PARTICIPANT = ''



    cfg.TRAINER.BALANCED_CE = False
    cfg.TRAINER.BALANCED_CE_NORMALIZED = True
    cfg.TRAINER.CLS_STEP = 525 # iterate if there are more tahn

    cfg.TRAINER.TEMPORAL = CN()
    cfg.TRAINER.TEMPORAL.TYPE = 'avg'
    cfg.TRAINER.TEMPORAL.TYPE_CTX = 'avg'
    cfg.TRAINER.TEMPORAL.BACKBONE_TYPE = 'avg'
    cfg.TRAINER.TEMPORAL.TFM_LAYERS = 2
    cfg.TRAINER.TEMPORAL.TFM_LAYERS_TEMP = 2
    cfg.TRAINER.TEMPORAL.TFM_BOTTLE_NECK = 0.25
    cfg.TRAINER.TEMPORAL.TFM_HEADS = 8
    cfg.TRAINER.TEMPORAL.DROPOUT = 0

    cfg.DATALOADER.HAND_CROPS = False
    cfg.DATALOADER.HAND_CROPS_NAME = ''
    cfg.DATALOADER.EPIC_VIDEOS = False
    cfg.DATASET.DETECTION_ROOT = ''

    cfg.DATASET.EGTEA = CN()
    cfg.DATASET.EGTEA.ROOT = ''
    cfg.DATASET.EGTEA.ANNOTATIONS_DIR = ''
    cfg.DATASET.EGTEA.METADATA_FILE = 'test_split_all.txt'
    cfg.DATASET.EGTEA.IS_TRIMMED = True
    cfg.DATASET.EGTEA.NUM_CLIPS = 1
    cfg.DATASET.EGTEA.NUM_CROPS = 1
    cfg.DATASET.EGTEA.CLIP_LENGTH = 8
    cfg.DATASET.EGTEA.CLIP_STRIDE = 1
    cfg.DATASET.EGTEA.SPARSE_SAMPLE = False
    cfg.DATASET.EGTEA.MEAN = [108.3272985, 116.7460125, 104.09373615000001]
    cfg.DATASET.EGTEA.STD = [68.5005327, 66.6321579, 70.32316305]

    cfg.DATALOADER.EGTEA = CN()
    cfg.DATALOADER.EGTEA.FEATURES_NAME = ''
    cfg.DATALOADER.EGTEA.FEATURES_NAME_DINO = ''
    cfg.DATALOADER.EGTEA.FEATURES_NAME_DINO2 = ''

    cfg.DATALOADER.EGOCLIP = CN()
    cfg.DATALOADER.EGOCLIP.TRAIN = False
    cfg.DATALOADER.EGOCLIP.EVAL = False
    cfg.DATALOADER.EGOCLIP.WITH_NEGS = False
    cfg.DATALOADER.EGOCLIP.WITH_REPHRASED_TEXT = False
    cfg.DATALOADER.EGOCLIP.BOUNDARIES_OFFSET = 4
    cfg.DATALOADER.HIDDEN_SIZE = 0


    cfg.DATALOADER.CROPPER = CN()
    cfg.DATALOADER.CROPPER.HAND_THS = 0.1
    cfg.DATALOADER.CROPPER.OBJ_THS = 0.1
    cfg.DATALOADER.CROPPER.ONLY_INTERACTED_OBJ = True
    cfg.DATALOADER.CROPPER.WITH_HANDS = True
    cfg.DATALOADER.CROPPER.SAVE_FILE = False
    cfg.DATALOADER.FRAME_TYPE = 'preextracted_rgb'
    cfg.DATALOADER.CROPPER.TAG = ''
    cfg.DATALOADER.CROPPER.FULL_FRAME = False
    cfg.DATALOADER.CROPPER.BLACK_CROP = False
    cfg.DATALOADER.DECODE_VIDEO_AT_ONCE = False

    cfg.DATALOADER.DETIC = CN()
    cfg.DATALOADER.DETIC.CROPS = False
    cfg.DATALOADER.DETIC.SAVE_FILE = False
    cfg.DATALOADER.DETIC.ROOT = ''
    cfg.DATALOADER.DETIC.FEATURES_MODEL = ''
    cfg.DATALOADER.DETIC.FEATURES_NAME = ''
    cfg.DATALOADER.DETIC.N_OBJ_PER_FRAME = 5
    cfg.DATALOADER.DETIC.N_TOTAL_OBJECTS = 300
    cfg.DATALOADER.DETIC.USE_BCE = False
    cfg.DATALOADER.DETIC.SEGMENT_LEVEL_BCE = False
    cfg.DATALOADER.DETIC.FRAME_LEVEL_BCE = False
    cfg.DATALOADER.DETIC.IMG_AND_OBJS_TEMP_AVG = True
    cfg.DATALOADER.DETIC.FRAME_LEVEL_BCE_WEIGHT = 0.1
    cfg.DATALOADER.DETIC.SEGMENT_LEVEL_BCE_WEIGHT = 0.1
    cfg.DATALOADER.DETIC.PROMPTS = False
    cfg.DATALOADER.DETIC.SEEN_SUBSET = False
    cfg.DATALOADER.N_CLASSES_CONTRASTIVE_LR = 350

    cfg.DATALOADER.USE_DINO_FEATURES = False
    cfg.DATALOADER.FEATURES_NAME_DINO = ''
    cfg.DATALOADER.USE_DINO_FEATURES2 = False
    cfg.DATALOADER.FEATURES_NAME_DINO2 = ''
    cfg.DATALOADER.PATH_PREFIX_LLAMA = ''
    cfg.DATALOADER.DINO_DIM = 1024
    cfg.DATALOADER.BNORM_DINO = False
    cfg.DATALOADER.L2ORM_DINO = True
    cfg.DATALOADER.LOAD_ALL_FEATURES_AT_ONCE = True
    # cfg.DATALOADER.WITH_NARRATIONS = False   # for contrastive learning training



    cfg.TEST.EVERY_EPOCH = False

    cfg.DATALOADER.DETIC.N_TOTAL_OBJECTS = 190 if cfg.DATASET.SUBSET == 'seen_nouns' else 300

    ###  ego4d dataset parameters ###
    cfg.NUM_GPUS = 1
    cfg.FBLEARNER = False
    cfg.DATALOADER.PIN_MEMORY = True

    cfg.SOLVER = CN()
    cfg.SOLVER.ACCELERATOR = 'ddp'

    cfg.TRAIN.BATCH_SIZE = 32
    cfg.TRAIN.DATASET = ''
    cfg.TRAIN.TEST_BEFORE_TRAIN = False
    cfg.TRAIN.EVAL_FREQ = 100000

    cfg.TEST.BATCH_SIZE = 32
    cfg.TEST.DATASET = ''
    cfg.TEST.LOADER = 'val'
    cfg.TEST.BASE_NOVEL_EVAL = False
    cfg.TEST.EGTEA_BASE_NOVEL_EVAL = False
    cfg.TEST.VAL_SPLIT = 'val'
    cfg.TEST.TEST_SPLIT = 'test'
    cfg.TEST.EVAL_ONLY = False

    cfg.TEST.CROSS_DATASET = CN()
    cfg.TEST.CROSS_DATASET.EVAL = False
    cfg.TEST.CROSS_DATASET.BASE_NOVEL_EVAL = False
    cfg.TEST.CROSS_DATASET.EGTEA = False
    cfg.TEST.CROSS_DATASET.RETRIEVAL = False
    cfg.TEST.CROSS_DATASET.DATASET_NAME = ''
    cfg.TEST.CROSS_DATASET.SPLIT = ''

    cfg.TEST.RETRIEVAL = False

    cfg.MODEL.BACKBONE.FRAMEWORK = "clip"

    cfg.DATA = CN()
    cfg.DATA.NUM_FRAMES = 8
    cfg.DATA.SAMPLING_RATE = 8
    cfg.DATA.TARGET_FPS = 30
    cfg.DATA.PATH_TO_DATA_DIR = ''
    cfg.DATA.PATH_PREFIX = ''
    cfg.DATA.PATH_PREFIX_MCQ = ''
    cfg.DATA.PATH_PREFIX_DINO = ''
    cfg.DATA.PATH_PREFIX_DINO2 = ''
    cfg.DATA.PATH_PREFIX_LLAMA = ''
    cfg.DATA.MEAN = [0.45, 0.45, 0.45]
    cfg.DATA.STD = [0.225, 0.225, 0.225]
    cfg.DATA.TRAIN_JITTER_SCALES = [256, 320]
    cfg.DATA.TRAIN_CROP_SIZE = 224
    cfg.DATA.TEST_CROP_SIZE = 256
    cfg.DATA.CROP_SIZE = 224
    cfg.DATA.RANDOM_FLIP = True
    # Decoding backend, options include `pyav` or `torchvision`
    cfg.DATA.DECODING_BACKEND = "pyav"
    cfg.DATA.INPUT_CHANNEL_NUM = [3, 3]
    cfg.DATA.SAVE_PATH = ''
    cfg.DATA.EGO4D_CLIPS_LENGTH = 8

    cfg.OPTIM.POLICY = False

    ######### BLIP #########
    cfg.MODEL.BLIP = CN()
    cfg.MODEL.BLIP.INIT_WEIGHTS_ROOT = '/mnt/graphics_ssd/nimble/users/annakukleva/models/blip/'
    cfg.MODEL.BLIP.INIT_WEIGHTS_NAME = 'model_base.pth' # 'model_base_capfilt_large.pth'
    cfg.MODEL.BLIP.VIT = 'base' # large



def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head

    if args.distributed:
        cfg.DISTRIBUTED = True
    else:
        cfg.DISTRIBUTED = False

    cfg.NAME = os.path.splitext(os.path.basename(args.config_file))[0]
