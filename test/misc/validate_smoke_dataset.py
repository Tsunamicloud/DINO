import json
import torch
import numpy as np
from pathlib import Path

def validate_coco_annotations(ann_file):
    """验证COCO格式标注文件"""
    print(f"正在验证: {ann_file}")
    
    with open(ann_file, 'r') as f:
        data = json.load(f)
    
    print(f"图像数量: {len(data['images'])}")
    print(f"标注数量: {len(data['annotations'])}")
    print(f"类别数量: {len(data['categories'])}")
    
    # 检查类别ID
    category_ids = [cat['id'] for cat in data['categories']]
    print(f"类别ID范围: {min(category_ids)} - {max(category_ids)}")
    print(f"类别ID列表: {sorted(category_ids)}")
    
    # 检查标注
    invalid_boxes = []
    invalid_categories = []
    
    for i, ann in enumerate(data['annotations']):
        # 检查bbox格式 [x, y, width, height]
        bbox = ann['bbox']
        x, y, w, h = bbox
        
        # 检查bbox有效性
        if w <= 0 or h <= 0:
            invalid_boxes.append({
                'annotation_id': ann['id'],
                'bbox': bbox,
                'issue': f'宽度或高度 <= 0: w={w}, h={h}'
            })
        
        if x < 0 or y < 0:
            invalid_boxes.append({
                'annotation_id': ann['id'], 
                'bbox': bbox,
                'issue': f'坐标为负: x={x}, y={y}'
            })
        
        # 检查类别ID
        if ann['category_id'] not in category_ids:
            invalid_categories.append({
                'annotation_id': ann['id'],
                'category_id': ann['category_id'],
                'issue': '类别ID不在定义的类别中'
            })
    
    # 报告问题
    if invalid_boxes:
        print(f"\n❌ 发现 {len(invalid_boxes)} 个无效bbox:")
        for issue in invalid_boxes[:5]:  # 只显示前5个
            print(f"  - 标注ID {issue['annotation_id']}: {issue['issue']}")
        if len(invalid_boxes) > 5:
            print(f"  - ... 还有 {len(invalid_boxes) - 5} 个无效bbox")
    else:
        print("\n✅ 所有bbox格式正确")
    
    if invalid_categories:
        print(f"\n❌ 发现 {len(invalid_categories)} 个无效类别:")
        for issue in invalid_categories[:5]:
            print(f"  - 标注ID {issue['annotation_id']}: 类别ID {issue['category_id']}")
    else:
        print("\n✅ 所有类别ID正确")
    
    return len(invalid_boxes) == 0 and len(invalid_categories) == 0

def main():
    """验证烟雾数据集"""
    print("=== 烟雾数据集验证工具 ===\n")
    
    dataset_path = Path("./coco_smoke_merged")
    
    # 验证训练集
    train_ann = dataset_path / "annotations" / "instances_train2017.json"
    if train_ann.exists():
        train_valid = validate_coco_annotations(train_ann)
    else:
        print(f"❌ 训练集标注文件不存在: {train_ann}")
        train_valid = False
    
    print("\n" + "="*50 + "\n")
    
    # 验证验证集
    val_ann = dataset_path / "annotations" / "instances_val2017.json"
    if val_ann.exists():
        val_valid = validate_coco_annotations(val_ann)
    else:
        print(f"❌ 验证集标注文件不存在: {val_ann}")
        val_valid = False
    
    print("\n" + "="*50)
    print("验证总结:")
    print(f"训练集: {'✅ 通过' if train_valid else '❌ 有问题'}")
    print(f"验证集: {'✅ 通过' if val_valid else '❌ 有问题'}")
    
    if not (train_valid and val_valid):
        print("\n⚠️ 数据集存在问题，需要修复后才能训练")
        return False
    else:
        print("\n🎉 数据集验证通过，可以开始训练！")
        return True

if __name__ == "__main__":
    main()