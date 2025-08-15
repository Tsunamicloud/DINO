#!/bin/bash
# resume_training.sh - 智能恢复训练

OUTPUT_DIR="logs/DINO/smoke_detection"
CHECKPOINT_FILE="$OUTPUT_DIR/checkpoint.pth"

echo "=== DINO训练恢复脚本 ==="

# 检查输出目录
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "创建输出目录: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
fi

# 检查是否存在checkpoint
if [ -f "$CHECKPOINT_FILE" ]; then
    echo "✅ 发现checkpoint文件，将从断点继续训练"
    echo "📁 Checkpoint: $CHECKPOINT_FILE"
    
    # 显示训练进度
    if [ -f "$OUTPUT_DIR/log.txt" ]; then
        LAST_EPOCH=$(grep -o '"epoch": [0-9]*' "$OUTPUT_DIR/log.txt" | tail -1 | grep -o '[0-9]*')
        echo "📊 上次训练到第 $LAST_EPOCH epoch"
    fi
else
    echo "❌ 未发现checkpoint文件，将从预训练模型开始"
fi

echo "🚀 开始训练..."

python main.py \
    --output_dir "$OUTPUT_DIR" \
    --config_file config/DINO/DINO_4scale_smoke.py \
    --coco_path ./coco_smoke_merged \
    --pretrain_model_path ./pretrained_models/checkpoint0011_4scale.pth \
    --finetune_ignore label_enc.weight class_embed \
    --options \
        dn_scalar=100 \
        embed_init_tgt=TRUE \
        dn_label_coef=1.0 \
        dn_bbox_coef=1.0 \
        use_ema=False \
        dn_box_noise_scale=1.0

echo "✅ 训练完成！"