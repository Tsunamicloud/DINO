#!/bin/bash

# DINO烟雾检测分布式训练脚本 - 使用Swin Transformer backbone
# 适配4个GPU分布式训练

echo "=========================================="
echo "DINO烟雾检测模型分布式训练 (Swin Transformer)"
echo "=========================================="

# 检查GPU可用性
if ! nvidia-smi > /dev/null 2>&1; then
    echo "错误: 未检测到NVIDIA GPU或驱动未安装"
    exit 1
fi

# 检查是否有足够的GPU
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "检测到 $GPU_COUNT 个GPU"

if [ $GPU_COUNT -lt 4 ]; then
    echo "错误: 需要至少4个GPU进行分布式训练，当前只检测到 $GPU_COUNT 个GPU"
    exit 1
fi

# 确保预训练模型存在
PRETRAINED_MODEL="./pretrained_models/checkpoint0011_4scale_swin.pth"
SWIN_PRETRAINED="./pretrained_models/swin_large_patch4_window12_384_22k.pth"

if [ ! -f "$PRETRAINED_MODEL" ]; then
    echo "错误: 预训练模型不存在: $PRETRAINED_MODEL"
    echo "请从以下链接下载:"
    echo "https://drive.google.com/drive/folders/1qD5m1NmK0kjE5hh-G17XUX751WsEG-h_"
    exit 1
fi

if [ ! -f "$SWIN_PRETRAINED" ]; then
    echo "警告: Swin预训练模型不存在: $SWIN_PRETRAINED"
    echo "请从以下链接下载:"
    echo "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth"
    echo "并保存到 ./pretrained_models/ 目录"
    exit 1
fi

# 检查数据集路径
DATASET_PATH="./coco_smoke_merged"
if [ ! -d "$DATASET_PATH" ]; then
    echo "错误: 数据集路径不存在: $DATASET_PATH"
    echo "请确保您的COCO格式烟雾数据集位于该路径"
    exit 1
fi

# 创建输出目录
OUTPUT_DIR="./logs/DINO/smoke_detection_swin_distributed"
mkdir -p "$OUTPUT_DIR"

# 分布式训练配置
export CUDA_VISIBLE_DEVICES=0,1,2,3
WORLD_SIZE=4
MASTER_PORT=29500

echo "开始分布式训练..."
echo "GPU数量: 4"
echo "配置文件: config/DINO/DINO_4scale_swin_FASDD_DFire_dist.py"
echo "数据集路径: $DATASET_PATH"
echo "输出目录: $OUTPUT_DIR"
echo "预训练模型: $PRETRAINED_MODEL"
echo "Swin预训练: $SWIN_PRETRAINED"
echo "Master端口: $MASTER_PORT"

# 检查端口是否被占用
if netstat -tuln | grep -q ":$MASTER_PORT "; then
    echo "警告: 端口 $MASTER_PORT 已被占用，尝试使用其他端口..."
    MASTER_PORT=$((29500 + $RANDOM % 1000))
    echo "使用新端口: $MASTER_PORT"
fi

# 分布式训练命令
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=$MASTER_PORT \
    main.py \
    --output_dir "$OUTPUT_DIR" \
    --config_file config/DINO/DINO_4scale_swin_FASDD_DFire_dist.py \
    --coco_path "$DATASET_PATH" \
    --pretrain_model_path "$PRETRAINED_MODEL" \
    --options backbone_dir="$SWIN_PRETRAINED" \
    --finetune_ignore label_enc.weight class_embed \
    --num_workers 32 \
    --options \
        dn_scalar=100 \
        embed_init_tgt=TRUE \
        dn_label_coef=1.0 \
        dn_bbox_coef=1.0 \
        use_ema=False \
        dn_box_noise_scale=0.4 \
        epochs=12 \
        save_checkpoint_interval=1

echo ""
echo "=========================================="
echo "分布式训练完成！"
echo "检查以下目录查看结果:"
echo "  - 训练日志: $OUTPUT_DIR"
echo "  - 模型检查点: $OUTPUT_DIR/checkpoint*.pth"
echo "  - 评估结果: $OUTPUT_DIR/eval.txt"
echo "=========================================="