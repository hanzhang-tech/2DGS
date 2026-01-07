#!/bin/bash

# ================= 配置区域 =================
CUDA_VISIBLE_DEVICES=2,3,4,5

# 1. 您的数据绝对路径 (根据您之前提供的信息已填好)
DATA_PATH="/home/zhangh/GS-CPR/ACT_Scaffold_GS/data/all_data/aachen/3D-models/aachen_v_1_1"

# 2. 实验名称 (自动加上时间戳，防止覆盖)
EXP_NAME="aachen_2dgs_train_$(date +%Y-%m-%d_%H-%M)"
OUTPUT_PATH="/data/zhangh/visual_localization/aachen"

CUDA_VISIBLE_DEVICES=2,3,4,5 python train.py \
    -s "$DATA_PATH" \
    -m "$OUTPUT_PATH" \
    --eval

echo "训练结束！结果保存在: $OUTPUT_PATH"

#    --data_device cpu\