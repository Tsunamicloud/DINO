#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D-Fire Dataset to COCO Format Converter
将D-Fire数据集从YOLO格式转换为COCO格式，只保留smoke类别
"""

import os
import json
import cv2
import shutil
from pathlib import Path
from datetime import datetime

# 配置路径 - 请根据实际情况修改
DFIRE_ROOT = "datasets/data/DFireDataset/D-Fire"
SPLIT_DIR = "datasets/data/DFireDataset/Data_splitting"
OUTPUT_DIR = "datasets/data/DFire_COCO"

def convert_yolo_to_coco_bbox(yolo_bbox, img_width, img_height):
    """将YOLO格式bbox转换为COCO格式"""
    x_center, y_center, width, height = yolo_bbox
    
    # 转换为像素坐标
    x_min = (x_center - width / 2) * img_width
    y_min = (y_center - height / 2) * img_height
    bbox_width = width * img_width
    bbox_height = height * img_height
    
    return [x_min, y_min, bbox_width, bbox_height]

def copy_image_to_output(src_path, dst_dir, img_name):
    """复制图片到输出目录"""
    dst_path = os.path.join(dst_dir, img_name)
    try:
        # 如果目标文件不存在或者源文件更新，则复制
        if not os.path.exists(dst_path) or os.path.getmtime(src_path) > os.path.getmtime(dst_path):
            shutil.copy2(src_path, dst_path)
            return True
        return False
    except Exception as e:
        print(f"Error copying {src_path} to {dst_path}: {e}")
        return False

def process_dataset():
    """主处理函数"""
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    images_output_dir = os.path.join(OUTPUT_DIR, "images")
    annotations_output_dir = os.path.join(OUTPUT_DIR, "annotations")
    os.makedirs(images_output_dir, exist_ok=True)
    os.makedirs(annotations_output_dir, exist_ok=True)
    
    # COCO格式基础信息
    base_info = {
        "info": {
            "description": "D-Fire Dataset (smoke only)",
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
        "licenses": []
    }
    
    # 处理train和test
    for split in ['train', 'test']:
        print(f"Processing {split} set...")
        
        # 读取数据集划分文件
        split_file = os.path.join(SPLIT_DIR, f"dfire_{split}.txt")
        if not os.path.exists(split_file):
            print(f"Warning: {split_file} not found!")
            continue
            
        with open(split_file, 'r') as f:
            image_files = [line.strip() for line in f.readlines() if line.strip()]
        
        # 初始化COCO数据
        coco_data = {
            **base_info,
            "images": [],
            "annotations": []
        }
        
        annotation_id = 1
        copied_images = 0
        images_with_smoke = 0
        images_without_smoke = 0
        
        # 处理每张图片
        for img_id, img_file in enumerate(image_files, 1):
            # 获取图片文件名
            img_name = os.path.basename(img_file)
            if not img_name.endswith(('.jpg', '.png')):
                img_name += '.jpg'
            
            # 构建完整路径
            img_path = os.path.join(DFIRE_ROOT, split, "images", img_name)
            label_path = os.path.join(DFIRE_ROOT, split, "labels", 
                                    img_name.replace('.jpg', '.txt').replace('.png', '.txt'))
            
            # 检查文件是否存在
            if not os.path.exists(img_path):
                print(f"Warning: {img_path} not found")
                continue
            
            # 读取图片尺寸
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img_height, img_width = img.shape[:2]
            except:
                continue
            
            # 复制所有图片到输出目录（不管是否有标注）
            if copy_image_to_output(img_path, images_output_dir, img_name):
                copied_images += 1
            
            # 添加图片信息
            image_info = {
                "id": img_id,
                "width": img_width,
                "height": img_height,
                "file_name": img_name
            }
            coco_data["images"].append(image_info)
            
            # 处理标注
            has_smoke_annotation = False
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_id = int(parts[0])
                            
                            # 只保留smoke类别 (class_id = 0)
                            if class_id == 0:
                                has_smoke_annotation = True
                                yolo_bbox = [float(x) for x in parts[1:5]]
                                coco_bbox = convert_yolo_to_coco_bbox(yolo_bbox, img_width, img_height)
                                
                                annotation = {
                                    "id": annotation_id,
                                    "image_id": img_id,
                                    "category_id": 0,
                                    "bbox": coco_bbox,
                                    "area": coco_bbox[2] * coco_bbox[3],
                                    "iscrowd": 0,
                                    "segmentation": []
                                }
                                coco_data["annotations"].append(annotation)
                                annotation_id += 1
            
            # 统计有无smoke标注的图片数量
            if has_smoke_annotation:
                images_with_smoke += 1
            else:
                images_without_smoke += 1
        
        # 保存结果
        if split == 'test':
            output_file = os.path.join(annotations_output_dir, "val.json")
        else:
            output_file = os.path.join(annotations_output_dir, f"{split}.json")
            
        with open(output_file, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        print(f"Saved {output_file}")
        print(f"  Total images: {len(coco_data['images'])}")
        print(f"  Images with smoke: {images_with_smoke}")
        print(f"  Images without smoke (negative samples): {images_without_smoke}")
        print(f"  Total annotations: {len(coco_data['annotations'])}")
        print(f"  Copied images: {copied_images}")

if __name__ == "__main__":
    print("Converting D-Fire dataset to COCO format...")
    print(f"Input: {DFIRE_ROOT}")
    print(f"Output: {OUTPUT_DIR}")
    print("="*50)
    
    process_dataset()
    
    print("\nConversion completed!")
    print("All images (including negative samples) have been copied to the output directory.")