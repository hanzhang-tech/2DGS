#!/bin/bash

# 设置 GPU ID (根据你的需求修改，比如 3)
export CUDA_VISIBLE_DEVICES=3

# 定义场景列表
# scenes=('chess' 'fire' 'heads' 'office' 'pumpkin' 'redkitchen' 'stairs')
scenes=('heads' 'office' 'pumpkin' 'redkitchen' 'stairs')

# 基础路径配置 (根据你的实际路径修改)
BASE_MODEL_DIR="/data/zhangh/visual_localization/7_scenes/2dgs"
BASE_POSE_DIR="/home/zhangh/GS-CPR/coarse_poses/ace/7Scenes_pgt"
BASE_OUTPUT_DIR="/data/zhangh/visual_localization/7_scenes/2dgs"

echo "=========================================="
echo "Starting Batch Rendering for 7-Scenes..."
echo "=========================================="

for scene in "${scenes[@]}"; do
    echo "------------------------------------------"
    echo "Processing Scene: $scene"
    
    # 拼接具体路径
    # 模型路径: .../2dgs/fire/model
    MODEL_PATH="${BASE_MODEL_DIR}/${scene}/model"
    
    # 位姿文件路径: .../poses_pgt_7scenes_fire_.txt
    # 注意：这里假设文件名格式固定为 poses_pgt_7scenes_[scene]_.txt
    POSE_FILE="${BASE_POSE_DIR}/poses_pgt_7scenes_${scene}_.txt"
    
    # 输出路径: .../2dgs/fire/render_result
    OUTPUT_PATH="${BASE_OUTPUT_DIR}/${scene}/render_result"

    echo "Model Path:  $MODEL_PATH"
    echo "Pose File:   $POSE_FILE"
    echo "Output Path: $OUTPUT_PATH"

    # 执行 Python 脚本
    # 这里的 render_7scenes.py 是你保存的主脚本文件名，如果不是这个名字请修改
    python render_7scenes.py \
        --model_path "$MODEL_PATH" \
        --pose_file "$POSE_FILE" \
        --scene_name "$scene" \
        --output_path "$OUTPUT_PATH"

    # 检查上一步是否成功
    if [ $? -eq 0 ]; then
        echo "✅ Scene [$scene] finished successfully."
    else
        echo "❌ Scene [$scene] failed!"
    fi
    
    echo "------------------------------------------"
done

echo "All tasks completed."