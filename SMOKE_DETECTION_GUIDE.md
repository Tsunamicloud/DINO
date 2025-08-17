# DINO烟雾检测训练指南 (Swin Transformer)

## 🎯 项目概述

本指南将帮助您使用DINO模型在单张RTX 4090 GPU上训练一个烟雾检测模型，使用Swin Transformer作为backbone。

### 🏗️ 技术栈
- **模型**: DINO (DETR with Improved DeNoising Anchor Boxes)
- **Backbone**: Swin Transformer Large (384x384, 22K预训练)
- **数据格式**: COCO格式
- **硬件**: 单张RTX 4090 (24GB显存)

## 📋 环境准备

### 1. 依赖安装

```bash
# 安装基础依赖
pip install -r requirements.txt

# 编译CUDA算子
cd models/dino/ops
python setup.py build install
# 运行测试确保编译成功
python test.py
cd ../../..
```

### 2. 下载预训练模型

创建预训练模型目录并下载必要的模型：

```bash
mkdir -p pretrained_models
cd pretrained_models

# 1. 下载DINO预训练模型 (COCO)
# 从Google Drive下载: checkpoint0011_4scale.pth
# 链接: https://drive.google.com/drive/folders/1qD5m1NmK0kjE5hh-G17XUX751WsEG-h_

# 2. 下载Swin Transformer预训练模型
wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth

cd ..
```

## 📁 数据集准备

### 1. COCO格式数据集结构

您的烟雾数据集需要组织成以下COCO格式：

```
smoke_dataset/
├── train2017/          # 训练图片
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
├── val2017/            # 验证图片
│   ├── val_001.jpg
│   ├── val_002.jpg
│   └── ...
└── annotations/        # 标注文件
    ├── instances_train2017.json
    └── instances_val2017.json
```

### 2. 标注文件格式

COCO格式的JSON标注文件需要包含以下结构：

```json
{
    "images": [
        {
            "id": 1,
            "width": 640,
            "height": 480,
            "file_name": "image_001.jpg"
        }
    ],
    "annotations": [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "bbox": [x, y, width, height],
            "area": 1200,
            "iscrowd": 0
        }
    ],
    "categories": [
        {
            "id": 1,
            "name": "smoke",
            "supercategory": "object"
        }
    ]
}
```

### 3. 数据集转换脚本

如果您的数据集不是COCO格式，可以使用以下Python脚本转换：

```python
import json
import os
from pathlib import Path

def convert_to_coco_format(image_dir, annotation_dir, output_file):
    """
    将自定义格式转换为COCO格式
    根据您的实际标注格式修改此函数
    """
    
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "smoke", "supercategory": "object"}]
    }
    
    image_id = 1
    annotation_id = 1
    
    # 遍历图片文件
    for img_file in Path(image_dir).glob("*.jpg"):
        # 添加图片信息
        # 这里需要根据您的实际情况获取图片尺寸
        coco_format["images"].append({
            "id": image_id,
            "width": 640,  # 实际宽度
            "height": 480, # 实际高度
            "file_name": img_file.name
        })
        
        # 添加对应的标注信息
        # 这里需要根据您的标注格式解析bbox
        # 假设有对应的标注文件
        ann_file = annotation_dir / f"{img_file.stem}.txt"
        if ann_file.exists():
            # 解析标注文件，获取bbox信息
            # bbox格式: [x, y, width, height]
            bbox = [100, 100, 50, 50]  # 示例，请替换为实际解析代码
            
            coco_format["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0
            })
            annotation_id += 1
        
        image_id += 1
    
    # 保存COCO格式文件
    with open(output_file, 'w') as f:
        json.dump(coco_format, f, indent=2)

# 使用示例
convert_to_coco_format(
    "your_images/train", 
    "your_annotations/train", 
    "smoke_dataset/annotations/instances_train2017.json"
)
```

## 🚀 训练步骤

### 1. 验证环境

```bash
# 检查GPU
nvidia-smi

# 验证CUDA算子编译
cd models/dino/ops
python test.py
cd ../../..
```

### 2. 开始训练

```bash
# 使用提供的训练脚本
./train_smoke_detection_swin.sh
```

### 3. 手动训练命令

如果您希望手动执行训练，可以使用以下命令：

```bash
python main.py \
    --output_dir ./logs/DINO/smoke_detection_swin \
    --config_file config/DINO/DINO_4scale_smoke_swin.py \
    --coco_path ./smoke_dataset \
    --pretrain_model_path ./pretrained_models/checkpoint0011_4scale.pth \
    --options backbone_dir="./pretrained_models/swin_large_patch4_window12_384_22k.pth" \
    --finetune_ignore label_enc.weight class_embed \
    --options \
        batch_size=1 \
        epochs=12 \
        lr=0.0001 \
        lr_drop=11 \
        save_checkpoint_interval=2
```

## ⚙️ 配置调优

### 单卡4090优化建议

1. **批量大小**: 使用 `batch_size=1`，Swin-L模型显存占用较大
2. **梯度检查点**: 已启用 `use_checkpoint=True` 节省显存
3. **学习率**: 初始学习率 `lr=0.0001`，backbone学习率 `lr_backbone=1e-05`
4. **训练周期**: 建议12个epoch，根据验证结果调整

### 内存优化

如果遇到显存不足，可以尝试：

```python
# 在配置文件中调整以下参数
batch_size = 1          # 减小批量大小
use_checkpoint = True   # 启用梯度检查点
num_queries = 300       # 减少查询数量（默认900）
hidden_dim = 256        # 保持隐藏维度不变
```

### 数据增强

可以在配置中启用强数据增强：

```bash
--options strong_aug=True
```

## 📊 监控训练

### 1. 训练日志

训练过程中的日志会保存在：
- `./logs/DINO/smoke_detection_swin/`

### 2. 检查点

模型检查点会定期保存：
- `checkpoint_{epoch:04d}.pth`：每个epoch的检查点
- `checkpoint_best.pth`：最佳模型

### 3. 评估结果

验证集上的mAP结果会记录在日志中，关注以下指标：
- `AP@0.5:0.95`：主要评估指标
- `AP@0.5`：IoU=0.5时的AP
- `AP@0.75`：IoU=0.75时的AP

## 🔧 常见问题

### 1. 显存不足

```bash
# 错误: CUDA out of memory
# 解决方案:
# 1. 减小batch_size到1
# 2. 启用gradient checkpointing
# 3. 减少num_queries
```

### 2. 训练速度慢

```bash
# 原因: Swin-L模型较大
# 优化建议:
# 1. 确保使用GPU
# 2. 启用mixed precision (如果支持)
# 3. 考虑使用更小的Swin模型
```

### 3. 精度不高

```bash
# 优化建议:
# 1. 增加训练周期到24或36个epoch
# 2. 调整学习率调度
# 3. 增加数据增强
# 4. 检查数据集质量
```

## 📈 性能期望

在烟雾检测任务上，您可以期望：

- **训练时间**: 约1-2小时/epoch（取决于数据集大小）
- **显存使用**: 约18-22GB（RTX 4090 24GB足够）
- **精度期望**: mAP@0.5 > 0.6（取决于数据质量）

## 🎉 后续步骤

训练完成后，您可以：

1. **模型评估**: 在测试集上评估模型性能
2. **模型推理**: 使用训练好的模型进行烟雾检测
3. **模型部署**: 将模型转换为ONNX等格式用于部署
4. **进一步优化**: 基于结果调整超参数或数据

## 📞 技术支持

如果遇到问题，可以：
1. 检查GitHub Issues
2. 查看DINO官方文档
3. 参考项目提供的示例和教程

---

**祝您训练顺利！🚀**
