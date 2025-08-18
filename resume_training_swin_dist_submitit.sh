#!/bin/bash
#
# DINO 烟雾检测 —— 断点继续训练脚本 (Swin Transformer, 4×GPU)
# 用法:
#   bash resume_training.sh                 # 自动寻找最新 checkpoint
#   bash resume_training.sh --ckpt ckpt.pth # 指定 checkpoint
#

set -e

echo "==============================================="
echo "DINO 烟雾检测模型 —— 断点继续训练 (Swin Transformer)"
echo "==============================================="

###########################
# 1. 环境与资源检查
###########################

# 检查 GPU
if ! nvidia-smi &> /dev/null; then
  echo "错误: 未检测到 NVIDIA GPU 或驱动未安装"; exit 1
fi

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "检测到 $GPU_COUNT 个 GPU"

if [ "$GPU_COUNT" -lt 4 ]; then
  echo "错误: 需要至少 4 个 GPU 进行分布式训练"; exit 1
fi

# Swin 预训练模型必须存在 (仍然需要做 backbone 权重加载)
SWIN_PRETRAINED="./pretrained_models/swin_large_patch4_window12_384_22k.pth"
if [ ! -f "$SWIN_PRETRAINED" ]; then
  echo "错误: 未找到 Swin 预训练模型: $SWIN_PRETRAINED"; exit 1
fi

# 数据集路径
DATASET_PATH="./coco_smoke_merged"
if [ ! -d "$DATASET_PATH" ]; then
  echo "错误: 数据集路径不存在: $DATASET_PATH"; exit 1
fi

###########################
# 2. 解析输入参数
###########################
CKPT_MANUAL=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt)
      CKPT_MANUAL="$2"; shift 2;;
    *)
      echo "未知参数: $1"; exit 1;;
  esac
done

###########################
# 3. 找到要 resume 的 checkpoint
###########################
OUTPUT_DIR="./logs/DINO/smoke_detection_swin_distributed"

if [ -n "$CKPT_MANUAL" ]; then
  RESUME_CKPT="$CKPT_MANUAL"
else
  # 取最新的 checkpointNN.pth
  RESUME_CKPT=$(ls -t ${OUTPUT_DIR}/checkpoint*.pth 2>/dev/null | head -n 1 || true)
fi

if [ ! -f "$RESUME_CKPT" ]; then
  echo "错误: 未找到可用于继续训练的 checkpoint"
  echo "  - 若要重新训练，请使用 train.sh"
  echo "  - 若指定 checkpoint，请执行: bash resume_training.sh --ckpt /path/to/checkpoint.pth"
  exit 1
fi

echo "将从 checkpoint 继续训练: $RESUME_CKPT"

###########################
# 4. 分布式训练配置
###########################
export CUDA_VISIBLE_DEVICES=0,1,2,3
WORLD_SIZE=4
MASTER_PORT=29500

# 端口避免冲突
if netstat -tuln 2>/dev/null | grep -q ":$MASTER_PORT "; then
  MASTER_PORT=$((29500 + RANDOM % 1000))
  echo "端口冲突，使用新端口: $MASTER_PORT"
fi

###########################
# 5. 启动分布式继续训练
###########################
python -m torch.distributed.launch \
  --nproc_per_node=$WORLD_SIZE \
  --master_port=$MASTER_PORT \
  main.py \
  --output_dir "$OUTPUT_DIR" \
  --resume "$RESUME_CKPT" \
  --config_file config/DINO/DINO_4scale_swin_FASDD_DFire_dist.py \
  --coco_path "$DATASET_PATH" \
  --options backbone_dir="./pretrained_models" \
  --num_workers 32

echo ""
echo "==============================================="
echo "断点继续训练已完成！请查看:"
echo "  - 训练日志: $OUTPUT_DIR"
echo "  - 新模型:   $OUTPUT_DIR/checkpoint*.pth"
echo "==============================================="