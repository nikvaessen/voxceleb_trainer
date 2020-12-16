#!/bin/bash
#SBATCH --partition=csedu
#SBATCH --account=nvaessen
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=2GB
#SBATCH -o /home/nvaessen/slurm/job_%j.out
#SBATCH -e /home/nvaessen/slurm/job_%j.err

DISK=/scratch-csedu/other/nik
DATA_PATH=$DISK/data/voxceleb
SAVE_PATH=$DISK/experiments/speaker/voxceleb/example
PROJECT_DIR=../../

echo "pwd: $(pwd)"
echo "disk: $DISK"
echo "data: $DATA_PATH"
echo "save: $SAVE_PATH"
echo "project: $PROJECT_DIR"
echo "CUDA: $CUDA_VISIBLE_DEVICES"

srun python $PROJECT_DIR/trainSpeakerNet.py \
--model ResNetSE34L \
--trainfunc angleproto \
--batch_size 400 \
--nPerSpeaker 2 \
--train_list $DATA_PATH/train_list.txt \
--train_path $DATA_PATH/voxceleb2/ \
--test_list $DATA_PATH/test_list.txt \
--test_path $DATA_PATH/voxceleb1/ \
--save_path $SAVE_PATH
