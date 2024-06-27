 #!/bin/bash

#cd ../..

# custom config
LABEL_SUBTYPES=$1
CFG=$2

LABEL_TYPE=all

DATASET=EpicKitchenSegments #Ego4d #EpicKitchenSegments
SUBDIR="final"

TAG='_ego_v0'

export MASTER_PORT=12356

WORLD_SIZE=8
export WORLD_SIZE=${WORLD_SIZE}

export MASTER_ADDR="localhost"
echo "MASTER_ADDR="$MASTER_ADDR


export OMP_NUM_THREADS=16


TIMESTAMP=$(date +%s)
DIR=../../output/multimodal-prompt-learning/${DATASET}/${SUBDIR}/${CFG}/${LABEL_SUBTYPES}${TAG}
TAG="${LABEL_TYPE}${TAG}"
echo "Run this job and save the output to ${DIR}"



 ### DEPLOY ## DISTRIBUTED ##
#DATALOADER.TRAIN_X.BATCH_SIZE 32 \
torchrun --standalone --nproc_per_node=${WORLD_SIZE} --nnodes 1 train_dist.py \
--config-file scripts/configs/${SUBDIR}/${CFG}.yaml \
--world_size ${WORLD_SIZE} \
--output-dir ${DIR} \
--neptune --neptune_mode=offline \
--tag ${TAG} \
--multiprocessing_distributed \
--distributed \
DATASET.LABEL_TYPE ${LABEL_TYPE} \
SEED -1 \
DATALOADER.TRAIN_X.BATCH_SIZE 4 \
TAG ${TAG} \
TRAIN.BATCH_SIZE 16 \
TEST.BATCH_SIZE 32 \
NUM_GPUS 8 \
DATASET.LABEL_SUBTYPES ${LABEL_SUBTYPES} \


