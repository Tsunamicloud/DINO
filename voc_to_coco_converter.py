import os
import json
import random
import shutil
from PIL import Image
from glob import glob
import argparse

def voc_to_coco_converter(voc_root_dir, output_dir, train_ratio=0.8):
    """
    将VOC格式数据集转换为COCO格式
    
    Args:
        voc_root_dir: VOC数据集根目录路径
        output_dir: 输出COCO数据集目录路径  
        train_ratio: 训练集比例 (默认0.8)
    """
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    annotations_dir = os.path.join(output_dir, "annotations")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)
    
    # COCO格式的基本结构
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 0,
                "name": "smoke",
                "supercategory": "object"
            }
        ]
    }
    
    # 初始化训练集和验证集
    train_coco = {
        "images": [],
        "annotations": [],
        "categories": coco_format["categories"]
    }
    
    val_coco = {
        "images": [],
        "annotations": [],
        "categories": coco_format["categories"]
    }
    
    # 收集所有图片和标注文件
    all_data = []
    groups = ["A组", "B组"]
    
    for group in groups:
        images_path = os.path.join(voc_root_dir, "input_data", group, "images")
        labels_path = os.path.join(voc_root_dir, "input_data", group, "labels")
        
        if not os.path.exists(images_path) or not os.path.exists(labels_path):
            print(f"警告: {group} 目录不存在，跳过")
            continue
        
        # 获取所有图片文件
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob(os.path.join(images_path, ext)))
            image_files.extend(glob(os.path.join(images_path, ext.upper())))
        
        for image_file in image_files:
            image_name = os.path.basename(image_file)
            label_name = os.path.splitext(image_name)[0] + ".txt"
            label_file = os.path.join(labels_path, label_name)
            
            if os.path.exists(label_file):
                all_data.append({
                    "image_file": image_file,
                    "label_file": label_file,
                    "image_name": image_name,
                    "group": group
                })
            else:
                print(f"警告: 找不到对应的标注文件 {label_file}")
    
    print(f"总共找到 {len(all_data)} 张图片")
    
    # 随机打乱数据
    random.shuffle(all_data)
    
    # 划分训练集和验证集
    train_size = int(len(all_data) * train_ratio)
    train_data = all_data[:train_size]
    val_data = all_data[train_size:]
    
    print(f"训练集: {len(train_data)} 张图片")
    print(f"验证集: {len(val_data)} 张图片")
    
    # 处理数据集
    image_id = 1
    annotation_id = 1
    
    def process_dataset(data_list, coco_dict, split_name):
        nonlocal image_id, annotation_id
        
        for data_item in data_list:
            image_file = data_item["image_file"]
            label_file = data_item["label_file"]
            image_name = data_item["image_name"]
            
            try:
                # 读取图片信息
                with Image.open(image_file) as img:
                    width, height = img.size
                
                # 复制图片到目标目录
                target_image_path = os.path.join(images_dir, image_name)
                shutil.copy2(image_file, target_image_path)
                
                # 添加图片信息到COCO格式
                image_info = {
                    "id": image_id,
                    "file_name": image_name,
                    "width": width,
                    "height": height
                }
                coco_dict["images"].append(image_info)
                
                # 读取VOC格式的标注文件
                if os.path.exists(label_file):
                    with open(label_file, 'r') as f:
                        lines = f.readlines()
                    
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        
                        parts = line.split()
                        if len(parts) >= 5:  # 类别 x1 y1 x2 y2 (置信度)
                            class_id = int(parts[0])
                            x1 = float(parts[1])
                            y1 = float(parts[2])
                            x2 = float(parts[3])
                            y2 = float(parts[4])
                            
                            # 确保坐标在图片范围内
                            x1 = max(0, min(x1, width))
                            y1 = max(0, min(y1, height))
                            x2 = max(0, min(x2, width))
                            y2 = max(0, min(y2, height))
                            
                            # 转换为COCO格式的bbox [x, y, width, height]
                            bbox_width = x2 - x1
                            bbox_height = y2 - y1
                            area = bbox_width * bbox_height
                            
                            # 只有当bbox有效时才添加
                            if bbox_width > 0 and bbox_height > 0:
                                annotation = {
                                    "id": annotation_id,
                                    "image_id": image_id,
                                    "category_id": 1,  # 烟雾类别ID为1
                                    "bbox": [x1, y1, bbox_width, bbox_height],
                                    "area": area,
                                    "iscrowd": 0
                                }
                                coco_dict["annotations"].append(annotation)
                                annotation_id += 1
                
                image_id += 1
                
            except Exception as e:
                print(f"处理文件 {image_file} 时出错: {str(e)}")
                continue
    
    # 处理训练集
    print("处理训练集...")
    process_dataset(train_data, train_coco, "train")
    
    # 处理验证集
    print("处理验证集...")
    process_dataset(val_data, val_coco, "val")
    
    # 保存COCO格式的标注文件
    train_json_path = os.path.join(annotations_dir, "train.json")
    val_json_path = os.path.join(annotations_dir, "val.json")
    
    with open(train_json_path, 'w', encoding='utf-8') as f:
        json.dump(train_coco, f, indent=2, ensure_ascii=False)
    
    with open(val_json_path, 'w', encoding='utf-8') as f:
        json.dump(val_coco, f, indent=2, ensure_ascii=False)
    
    print(f"\n转换完成!")
    print(f"训练集: {len(train_coco['images'])} 张图片, {len(train_coco['annotations'])} 个标注")
    print(f"验证集: {len(val_coco['images'])} 张图片, {len(val_coco['annotations'])} 个标注")
    print(f"数据集保存在: {output_dir}")
    print(f"训练集标注: {train_json_path}")
    print(f"验证集标注: {val_json_path}")

def main():
    # 设置随机种子以确保可重现性
    random.seed(42)
    
    # 输入和输出路径
    voc_root_dir = "datasets/data/longyuan_dataset"  # VOC数据集根目录
    output_dir = "datasets/data/longyuan_final"      # COCO输出目录
    
    print("开始VOC到COCO格式转换...")
    print(f"输入目录: {voc_root_dir}")
    print(f"输出目录: {output_dir}")
    
    # 检查输入目录是否存在
    if not os.path.exists(voc_root_dir):
        print(f"错误: 输入目录 {voc_root_dir} 不存在!")
        return
    
    # 执行转换
    voc_to_coco_converter(voc_root_dir, output_dir, train_ratio=0.8)

if __name__ == "__main__":
    main()