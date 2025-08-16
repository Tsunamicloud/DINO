import json
import torch
import numpy as np
from pathlib import Path
import sys
import os

# 添加项目路径
sys.path.append('.')

from datasets import build_dataset
from util.box_ops import box_cxcywh_to_xyxy
import argparse

def create_debug_args():
    """创建调试用的参数"""
    args = argparse.Namespace()
    args.coco_path = './coco_smoke_merged'
    args.dataset_file = 'coco'
    args.masks = False
    args.fix_size = False
    args.strong_aug = False
    args.data_aug_scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    args.data_aug_max_size = 1333
    args.data_aug_scales2_resize = [400, 500, 600]
    args.data_aug_scales2_crop = [384, 600]
    args.data_aug_scale_overlap = None
    return args

def check_bbox_validity(boxes, name=""):
    """检查bbox的有效性"""
    issues = []
    
    if boxes.numel() == 0:
        return issues
        
    # 检查NaN和Inf
    if torch.isnan(boxes).any():
        issues.append(f"{name}: 包含NaN值")
    if torch.isinf(boxes).any():
        issues.append(f"{name}: 包含Inf值")
    
    # 检查范围
    if (boxes < 0).any():
        issues.append(f"{name}: 包含负值")
    if (boxes > 1.1).any():  # 允许一些数值误差
        issues.append(f"{name}: 包含超过1的值")
    
    # 检查bbox格式 (cx, cy, w, h)
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    if (w <= 0).any():
        issues.append(f"{name}: 宽度 <= 0")
    if (h <= 0).any():
        issues.append(f"{name}: 高度 <= 0")
    
    # 检查转换后的xyxy格式
    try:
        boxes_xyxy = box_cxcywh_to_xyxy(boxes)
        x1, y1, x2, y2 = boxes_xyxy[:, 0], boxes_xyxy[:, 1], boxes_xyxy[:, 2], boxes_xyxy[:, 3]
        if (x2 <= x1).any():
            issues.append(f"{name}: x2 <= x1")
        if (y2 <= y1).any():
            issues.append(f"{name}: y2 <= y1")
    except Exception as e:
        issues.append(f"{name}: 转换到xyxy时出错: {e}")
    
    return issues

def debug_single_sample(dataset, idx):
    """调试单个样本"""
    print(f"\n{'='*50}")
    print(f"调试样本 #{idx}")
    print(f"{'='*50}")
    
    try:
        img, target = dataset[idx]
        
        print(f"图像形状: {img.shape}")
        print(f"图像类型: {img.dtype}")
        print(f"图像值范围: [{img.min():.3f}, {img.max():.3f}]")
        
        print(f"\n目标信息:")
        for key, value in target.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                if key == 'boxes':
                    print(f"    bbox值范围: [{value.min():.3f}, {value.max():.3f}]")
                    issues = check_bbox_validity(value, "boxes")
                    if issues:
                        print(f"    ❌ 发现问题: {', '.join(issues)}")
                        return False
                    else:
                        print(f"    ✅ bbox格式正确")
                elif key == 'labels':
                    unique_labels = torch.unique(value)
                    print(f"    标签: {unique_labels.tolist()}")
                    if (value < 0).any() or (value >= 1).any():  # 烟雾数据集只有1个类别，标签应该是0
                        print(f"    ❌ 标签超出范围 [0, 0]: {value.tolist()}")
                        return False
            else:
                print(f"  {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ 处理样本时出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主调试函数"""
    print("=== DINO训练数据调试工具 ===\n")
    
    # 创建参数
    args = create_debug_args()
    
    # 构建数据集
    print("🔍 构建训练数据集...")
    try:
        dataset = build_dataset('train', args)
        print(f"✅ 数据集大小: {len(dataset)}")
    except Exception as e:
        print(f"❌ 构建数据集失败: {e}")
        return False
    
    # 测试前几个样本
    print(f"\n🧪 测试前10个样本...")
    failed_samples = []
    
    for i in range(min(10, len(dataset))):
        success = debug_single_sample(dataset, i)
        if not success:
            failed_samples.append(i)
    
    # 随机测试一些样本
    print(f"\n🎲 随机测试10个样本...")
    import random
    random_indices = random.sample(range(len(dataset)), min(10, len(dataset)))
    
    for idx in random_indices:
        success = debug_single_sample(dataset, idx)
        if not success:
            failed_samples.append(idx)
    
    print(f"\n{'='*50}")
    print("调试总结:")
    if failed_samples:
        print(f"❌ 失败的样本: {failed_samples}")
        print(f"❌ 总共测试了 {20} 个样本，失败了 {len(failed_samples)} 个")
        return False
    else:
        print(f"✅ 所有测试样本都正常")
        print(f"✅ 数据预处理流程正确")
        return True

if __name__ == "__main__":
    success = main()
    if not success:
        print(f"\n💡 建议:")
        print(f"1. 检查数据预处理pipeline")
        print(f"2. 检查类别标签映射")
        print(f"3. 启用CUDA_LAUNCH_BLOCKING=1进行详细调试")