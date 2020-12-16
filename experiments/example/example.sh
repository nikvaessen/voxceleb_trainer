#!/bin/bash
#SBATCH --partition=csedu
#SBATCH --account=nvaessen
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --gpus 1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=2GB

DISK=/scratch-csedu/other/nik/
DATA_PATH=$DISK/data/voxceleb/
SAVE_PATH=$DISK/experiments/speaker/voxceleb/example

srun python trainSpeakerNet.py \
--model ResNetSE34L \
--trainfunc angleproto \
--batch_size 400 \
--nPerSpeaker 2 \
--train_list $DATA_PATH/train_list.txt \
--train_path $DATA_PATH/voxceleb2/ \
--test_list $DATA_PATH/test_list.txt \
--test_path $DATA_PATH/voxceleb1/ \
--save_path $SAVE_PATH
