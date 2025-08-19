#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COCO Dataset Merger
把D-Fire数据集和FASDD_Reorganized数据集合并为FASDD_DFire_Dataset
"""

import os
import json
import shutil
from datetime import datetime

# 配置路径
DFIRE_ROOT = "datasets/data/DFire_COCO"
FASDD_ROOT = "datasets/data/FASDD_Reorganized"  
OUTPUT_ROOT = "datasets/data/FASDD_DFire_Dataset"

def merge_coco_files(dfire_file, fasdd_file, output_file):
    """合并两个COCO标注文件"""
    
    # 加载数据
    dfire_data = None
    fasdd_data = None
    
    if os.path.exists(dfire_file):
        with open(dfire_file, 'r') as f:
            dfire_data = json.load(f)
            print(f"Loaded D-Fire: {len(dfire_data.get('images', []))} images, {len(dfire_data.get('annotations', []))} annotations")
    
    if os.path.exists(fasdd_file):
        with open(fasdd_file, 'r') as f:
            fasdd_data = json.load(f)
            print(f"Loaded FASDD: {len(fasdd_data.get('images', []))} images, {len(fasdd_data.get('annotations', []))} annotations")
    
    # 创建合并数据
    merged_data = {
        "info": {
            "description": "Merged D-Fire and FASDD datasets (smoke only)",
            "version": "1.0",
            "year": 2024,
            "date_created": datetime.now().isoformat()
        },
        "categories": [
            {
                "id": 0,
                "name": "smoke",
                "supercategory": "fire_related"
            }
        ],
        "images": [],
        "annotations": []
    }
    
    image_id = 1
    annotation_id = 1
    
    # 处理D-Fire数据集
    if dfire_data:
        old_to_new_img_id = {}
        for img in dfire_data.get("images", []):
            old_id = img["id"]
            new_img = img.copy()
            new_img["id"] = image_id
            new_img["source_dataset"] = "D-Fire"
            merged_data["images"].append(new_img)
            old_to_new_img_id[old_id] = image_id
            image_id += 1
        
        # 添加标注
        for ann in dfire_data.get("annotations", []):
            if ann.get("category_id") == 0:  # 只要smoke类别
                new_ann = ann.copy()
                new_ann["id"] = annotation_id
                new_ann["image_id"] = old_to_new_img_id[ann["image_id"]]
                new_ann["category_id"] = 0
                merged_data["annotations"].append(new_ann)
                annotation_id += 1
    
    # 处理FASDD数据集
    if fasdd_data:
        old_to_new_img_id = {}
        for img in fasdd_data.get("images", []):
            old_id = img["id"]
            new_img = img.copy()
            new_img["id"] = image_id
            new_img["source_dataset"] = "FASDD"
            merged_data["images"].append(new_img)
            old_to_new_img_id[old_id] = image_id
            image_id += 1
        
        # 添加标注
        for ann in fasdd_data.get("annotations", []):
            if ann.get("category_id") == 0:  # 只要smoke类别
                new_ann = ann.copy()
                new_ann["id"] = annotation_id
                new_ann["image_id"] = old_to_new_img_id[ann["image_id"]]
                new_ann["category_id"] = 0
                merged_data["annotations"].append(new_ann)
                annotation_id += 1
    
    # 保存合并结果
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(merged_data, f, indent=2)
    
    print(f"Merged result: {len(merged_data['images'])} images, {len(merged_data['annotations'])} annotations")
    return merged_data

def copy_images_simple(src_dir, dst_dir, image_list, dataset_name):
    """简单复制图片文件"""
    if not os.path.exists(src_dir):
        print(f"Warning: {src_dir} not found")
        return
    
    os.makedirs(dst_dir, exist_ok=True)
    copied = 0
    
    for img_info in image_list:
        if img_info.get("source_dataset") == dataset_name:
            src_path = os.path.join(src_dir, img_info["file_name"])
            dst_path = os.path.join(dst_dir, img_info["file_name"])
            
            if os.path.exists(src_path):
                if not os.path.exists(dst_path):
                    try:
                        shutil.copy2(src_path, dst_path)
                        copied += 1
                    except Exception as e:
                        print(f"Error copying {src_path}: {e}")
    
    print(f"Copied {copied} images from {dataset_name}")

def main():
    print("Merging D-Fire and FASDD datasets...")
    print(f"D-Fire: {DFIRE_ROOT}")
    print(f"FASDD: {FASDD_ROOT}")
    print(f"Output: {OUTPUT_ROOT}")
    print("="*50)
    
    # 创建输出目录
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_ROOT, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_ROOT, "annotations"), exist_ok=True)
    
    # 合并训练集和验证集
    for split in ['train', 'val']:
        print(f"\nProcessing {split} set...")
        
        dfire_file = os.path.join(DFIRE_ROOT, "annotations", f"{split}.json")
        fasdd_file = os.path.join(FASDD_ROOT, "annotations", f"{split}.json")
        output_file = os.path.join(OUTPUT_ROOT, "annotations", f"{split}.json")
        
        # 合并标注文件
        merged_data = merge_coco_files(dfire_file, fasdd_file, output_file)
        
        # 复制图片
        if merged_data:
            copy_images_simple(
                os.path.join(DFIRE_ROOT, "images"),
                os.path.join(OUTPUT_ROOT, "images"),
                merged_data["images"],
                "D-Fire"
            )
            
            copy_images_simple(
                os.path.join(FASDD_ROOT, "images"),
                os.path.join(OUTPUT_ROOT, "images"),
                merged_data["images"],
                "FASDD"
            )
    
    print("\nMerge completed!")
    print(f"Check output directory: {OUTPUT_ROOT}")

if __name__ == "__main__":
    main()