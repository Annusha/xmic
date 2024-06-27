 #!/bin/bash

#cd ../..

LABEL_SUBTYPES=$1
CFG=$2

LABEL_TYPE=all

DATASET=EpicKitchenSegments #Ego4d #EpicKitchenSegments
SUBDIR="final" # $1  # conf_100_200  baselines  finetune ego4d_0 epic_0  lavila_epic_0

TAG='_epic_v0'


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



### DEPLOY ego4d scripts for epic kitchens ##
 python train.py \
 --config-file scripts/configs/${SUBDIR}/${CFG}.yaml \
 --world_size 1 \
 --output-dir ${DIR} \
 --neptune --neptune_mode=offline \
 --tag ${TAG} \
 DATASET.LABEL_TYPE ${LABEL_TYPE} \
 SEED 6549 \
 DATALOADER.TRAIN_X.BATCH_SIZE 128 \
 TRAIN.BATCH_SIZE 128 \
 TEST.BATCH_SIZE 32 \
 NUM_GPUS 1 \
 DATASET.NAME 'EpicKitchenSegmentsAllLabelTypes' \
 TEST.RETRIEVAL False \
 TEST.CROSS_DATASET.RETRIEVAL False \
 TEST.CROSS_DATASET.DATASET_NAME 'Ego4DRecognitionWrapper' \
 DATASET.LABEL_SUBTYPES ${LABEL_SUBTYPES} \

