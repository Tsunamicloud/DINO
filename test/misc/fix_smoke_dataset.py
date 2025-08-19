import json
import shutil
from pathlib import Path

def fix_coco_annotations(ann_file, output_file=None):
    """修复COCO标注文件中的无效bbox"""
    if output_file is None:
        output_file = ann_file
    
    print(f"修复文件: {ann_file}")
    
    with open(ann_file, 'r') as f:
        data = json.load(f)
    
    original_count = len(data['annotations'])
    fixed_annotations = []
    removed_annotations = []
    
    for ann in data['annotations']:
        bbox = ann['bbox']
        x, y, w, h = bbox
        annotation_id = ann['id']
        
        # 检查bbox有效性
        if w <= 0 or h <= 0:
            removed_annotations.append({
                'id': annotation_id,
                'bbox': bbox,
                'reason': f'无效尺寸: w={w}, h={h}'
            })
            print(f"🗑️  移除无效标注: ID={annotation_id}, bbox={bbox}, 原因: w={w}, h={h}")
        elif x < 0 or y < 0:
            removed_annotations.append({
                'id': annotation_id, 
                'bbox': bbox,
                'reason': f'负坐标: x={x}, y={y}'
            })
            print(f"🗑️  移除无效标注: ID={annotation_id}, bbox={bbox}, 原因: x={x}, y={y}")
        else:
            # 有效的标注，保留
            fixed_annotations.append(ann)
    
    # 更新标注列表
    data['annotations'] = fixed_annotations
    
    # 保存修复后的文件
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n📊 修复结果:")
    print(f"   原始标注数: {original_count}")
    print(f"   移除标注数: {len(removed_annotations)}")
    print(f"   保留标注数: {len(fixed_annotations)}")
    print(f"   修复文件: {output_file}")
    
    return len(removed_annotations)

def main():
    """修复烟雾数据集"""
    print("=== 烟雾数据集修复工具 ===\n")
    
    dataset_path = Path("./coco_smoke_merged")
    annotations_path = dataset_path / "annotations"
    
    # 备份原始文件
    train_ann = annotations_path / "instances_train2017.json"
    backup_file = annotations_path / "instances_train2017_backup.json"
    
    if train_ann.exists():
        if not backup_file.exists():
            shutil.copy2(train_ann, backup_file)
            print(f"💾 备份原始文件: {backup_file}")
        else:
            print(f"📁 备份文件已存在: {backup_file}")
        
        # 修复训练集
        print(f"\n🔧 修复训练集标注...")
        removed_count = fix_coco_annotations(train_ann)
        
        if removed_count == 0:
            print("✅ 训练集无需修复")
        else:
            print(f"✅ 训练集修复完成，移除了 {removed_count} 个无效标注")
        
    else:
        print(f"❌ 训练集标注文件不存在: {train_ann}")
    
    # 验证集通常不需要修复，但我们也检查一下
    val_ann = annotations_path / "instances_val2017.json"
    if val_ann.exists():
        print(f"\n🔍 检查验证集...")
        with open(val_ann, 'r') as f:
            val_data = json.load(f)
        
        invalid_count = 0
        for ann in val_data['annotations']:
            bbox = ann['bbox']
            x, y, w, h = bbox
            if w <= 0 or h <= 0 or x < 0 or y < 0:
                invalid_count += 1
        
        if invalid_count == 0:
            print("✅ 验证集无需修复")
        else:
            print(f"⚠️  验证集发现 {invalid_count} 个无效标注，也进行修复...")
            val_backup = annotations_path / "instances_val2017_backup.json"
            if not val_backup.exists():
                shutil.copy2(val_ann, val_backup)
            fix_coco_annotations(val_ann)
    
    print(f"\n🎉 数据集修复完成！")
    print(f"💡 提示：原始文件已备份，如需恢复可使用备份文件")

if __name__ == "__main__":
    main()