#!/usr/bin/bash
#SBATCH --job-name DV3
#SBATCH --account OPEN-30-35
#SBATCH --partition qgpu
#SBATCH --time 48:00:00
#SBATCH --gpus-per-node 8
#SBATCH --nodes 8

# Downloaded from https://huggingface.co/opensourcerelease/DeepSeek-V3-bf16

NAME='deepseekv3_685B_instruct'
MODEL_NAME='/home/ifajcik/data_scratch_new/DeepSeek-V3/bf16_weights/DeepSeek-V3-bf16'

source ./jobs/scripts/submit/fire/fire_L_smartt.sh
