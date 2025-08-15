import json
from tqdm import tqdm

def process_annotations(input_file, output_file):
    print(f"正在处理 {input_file}...")
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # 只保留smoke类别，id改为0
    data['categories'] = [{"id": 0, "name": "smoke", "supercategory": "object"}]
    
    # 统计需要处理的annotations数量
    total_annotations = len(data['annotations'])
    print(f"总共有 {total_annotations} 个标注需要处理")
    
    # 使用列表推导式和进度条更新annotations中的category_id
    filtered_annotations = []
    
    with tqdm(total=total_annotations, desc="处理标注", unit="个") as pbar:
        for ann in data['annotations']:
            if ann['category_id'] == 1:  # smoke的原始id
                ann['category_id'] = 0
                filtered_annotations.append(ann)
            # 删除fire类别的标注（不添加到filtered_annotations）
            pbar.update(1)
    
    data['annotations'] = filtered_annotations
    
    print(f"保留了 {len(filtered_annotations)} 个smoke标注")
    print(f"正在保存到 {output_file}...")
    
    with open(output_file, 'w') as f:
        json.dump(data, f)
    
    print(f"处理完成！")

# 处理训练和验证集
print("开始处理数据集...")
process_annotations('coco_smoke_merged/annotations/instances_train2017_original.json', 
                   'coco_smoke_merged/annotations/instances_train2017.json')

process_annotations('coco_smoke_merged/annotations/instances_val2017_original.json', 
                   'coco_smoke_merged/annotations/instances_val2017.json')

print("所有文件处理完成！")