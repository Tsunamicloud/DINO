#!/bin/bash

# DINO烟雾检测训练脚本
# 确保已下载预训练模型到 ./pretrained_models/checkpoint0011_4scale.pth

echo "开始DINO烟雾检测模型训练..."

python main.py \
    --output_dir logs/DINO/smoke_detection \
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
        dn_box_noise_scale=1.0 \
        batch_size=1 \
        epochs=10 \
        lr=0.0001

echo "训练完成！检查 logs/DINO/smoke_detection 目录查看结果"