import json
import shutil
from pathlib import Path

def fix_category_ids(ann_file, output_file=None):
    """将COCO标注文件中的category_id从1改为0"""
    if output_file is None:
        output_file = ann_file
    
    print(f"🔧 修复文件: {ann_file}")
    
    with open(ann_file, 'r') as f:
        data = json.load(f)
    
    print(f"📊 原始数据统计:")
    print(f"   图像数量: {len(data['images'])}")
    print(f"   标注数量: {len(data['annotations'])}")
    print(f"   类别数量: {len(data['categories'])}")
    
    # 显示原始类别信息
    print(f"\n📋 原始类别信息:")
    for cat in data['categories']:
        print(f"   ID: {cat['id']}, Name: {cat['name']}")
    
    # 修复categories部分
    for cat in data['categories']:
        if cat['id'] == 1:
            print(f"✏️  修改类别ID: {cat['id']} -> 0")
            cat['id'] = 0
    
    # 修复annotations部分
    modified_count = 0
    for ann in data['annotations']:
        if ann['category_id'] == 1:
            ann['category_id'] = 0
            modified_count += 1
    
    print(f"\n📝 修改统计:")
    print(f"   修改的标注数量: {modified_count}")
    
    # 验证修改结果
    category_ids = set()
    for ann in data['annotations']:
        category_ids.add(ann['category_id'])
    
    print(f"   修改后的类别ID: {sorted(category_ids)}")
    
    # 保存修改后的文件
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"💾 保存到: {output_file}")
    return True

def main():
    """修复烟雾数据集的类别ID"""
    print("=== 烟雾数据集类别ID修复工具 ===\n")
    
    dataset_path = Path("./coco_smoke_merged")
    annotations_path = dataset_path / "annotations"
    
    # 需要修复的文件列表
    files_to_fix = [
        "instances_train2017.json",
        "instances_val2017.json"
    ]
    
    for filename in files_to_fix:
        ann_file = annotations_path / filename
        backup_file = annotations_path / f"{filename.replace('.json', '_backup_categoryid.json')}"
        
        if not ann_file.exists():
            print(f"⚠️  文件不存在，跳过: {ann_file}")
            continue
        
        print(f"\n{'='*60}")
        print(f"处理文件: {filename}")
        print(f"{'='*60}")
        
        # 备份原始文件
        if not backup_file.exists():
            shutil.copy2(ann_file, backup_file)
            print(f"💾 备份原始文件: {backup_file}")
        else:
            print(f"📁 备份文件已存在: {backup_file}")
        
        # 修复文件
        try:
            success = fix_category_ids(ann_file)
            if success:
                print(f"✅ {filename} 修复完成")
            else:
                print(f"❌ {filename} 修复失败")
        except Exception as e:
            print(f"❌ 修复 {filename} 时出错: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("🎉 类别ID修复完成！")
    print("💡 提示:")
    print("   - 原始文件已备份为 *_backup_categoryid.json")
    print("   - 现在烟雾类别的ID是0，符合DINO的要求")
    print("   - 可以重新开始训练了")

def verify_fix():
    """验证修复结果"""
    print("\n=== 验证修复结果 ===")
    
    dataset_path = Path("./coco_smoke_merged/annotations")
    
    for filename in ["instances_train2017.json", "instances_val2017.json"]:
        ann_file = dataset_path / filename
        
        if not ann_file.exists():
            continue
            
        print(f"\n🔍 验证 {filename}:")
        
        with open(ann_file, 'r') as f:
            data = json.load(f)
        
        # 检查categories
        category_ids = [cat['id'] for cat in data['categories']]
        print(f"   类别ID: {category_ids}")
        
        # 检查annotations
        ann_category_ids = set(ann['category_id'] for ann in data['annotations'])
        print(f"   标注中的类别ID: {sorted(ann_category_ids)}")
        
        # 验证
        if category_ids == [0] and ann_category_ids == {0}:
            print(f"   ✅ {filename} 修复正确")
        else:
            print(f"   ❌ {filename} 仍有问题")

if __name__ == "__main__":
    main()
    verify_fix()