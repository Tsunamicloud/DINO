#!/usr/bin/env python3
"""
FASDD数据集重组脚本
功能：
1. 合并CV和UAV两个数据集
2. 重新划分训练集和验证集比例为80% : 20%，将测试集也合并进去
3. 只保留smoke类别的标注，但保留所有图像（包括负样本）
4. 输出重组后的数据集到指定目录
"""

import json
import os
import shutil
from pathlib import Path
import random
from typing import Dict, List, Tuple
from collections import defaultdict

class FASSDDatasetReorganizer:
    def __init__(self, cv_root: str, uav_root: str, output_root: str):
        """
        初始化数据集重组器
        
        Args:
            cv_root: CV数据集根目录
            uav_root: UAV数据集根目录  
            output_root: 输出目录
        """
        self.cv_root = Path(cv_root)
        self.uav_root = Path(uav_root)
        self.output_root = Path(output_root)
        
        # 创建输出目录结构
        self.output_images_dir = self.output_root / "images"
        self.output_annotations_dir = self.output_root / "annotations"
        
        for dir_path in [self.output_images_dir, self.output_annotations_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def load_coco_annotation(self, annotation_path: str) -> Dict:
        """加载COCO格式标注文件"""
        with open(annotation_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def save_coco_annotation(self, data: Dict, output_path: str):
        """保存COCO格式标注文件"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    def filter_smoke_annotations(self, coco_data: Dict) -> Dict:
        """
        过滤只保留smoke类别的标注，但保留所有图像（包括负样本）
        
        Args:
            coco_data: COCO格式数据
            
        Returns:
            过滤后的COCO数据
        """
        # 找到smoke类别的ID
        smoke_category_id = None
        for category in coco_data['categories']:
            if category['name'].lower() == 'smoke':
                smoke_category_id = category['id']
                break
        
        if smoke_category_id is None:
            print("Warning: 未找到smoke类别")
            return coco_data
        
        # 过滤标注，只保留smoke类别
        filtered_annotations = []
        for ann in coco_data['annotations']:
            if ann['category_id'] == smoke_category_id:
                filtered_annotations.append(ann)
        
        # 保留所有图像（包括负样本）
        all_images = coco_data['images'].copy()
        
        # 统计信息
        image_ids_with_smoke = set(ann['image_id'] for ann in filtered_annotations)
        images_with_smoke = len(image_ids_with_smoke)
        negative_samples = len(all_images) - images_with_smoke
        
        # 更新数据
        result = coco_data.copy()
        result['annotations'] = filtered_annotations
        result['images'] = all_images
        result['categories'] = [{'id': 0, 'name': 'smoke', 'supercategory': 'object'}]  # 修改：ID改为0
        
        print(f"保留 {len(all_images)} 张图像（其中 {images_with_smoke} 张正样本，{negative_samples} 张负样本），{len(filtered_annotations)} 个smoke标注")
        return result
    
    def merge_datasets(self) -> Tuple[Dict, Dict]:
        """
        合并CV和UAV数据集
        
        Returns:
            合并后的所有数据 (images, annotations)
        """
        print("开始合并数据集...")
        
        all_images = []
        all_annotations = []
        next_image_id = 0
        next_annotation_id = 0
        
        # 处理CV数据集
        print("处理CV数据集...")
        cv_anno_dir = self.cv_root / "annotations" / "COCO_CV" / "Annotations"
        cv_img_dir = self.cv_root / "images"
        
        for split in ['train', 'val', 'test']:
            anno_file = cv_anno_dir / f"{split}.json"
            if not anno_file.exists():
                print(f"警告: {anno_file} 不存在，跳过")
                continue
                
            coco_data = self.load_coco_annotation(str(anno_file))
            coco_data = self.filter_smoke_annotations(coco_data)
            
            # 重新映射图像ID
            image_id_mapping = {}
            for img in coco_data['images']:
                old_id = img['id']
                img['id'] = next_image_id
                image_id_mapping[old_id] = next_image_id
                
                # 添加数据集来源标识
                img['dataset'] = 'CV'
                img['original_split'] = split
                all_images.append(img)
                next_image_id += 1
            
            # 重新映射标注ID和图像ID
            for ann in coco_data['annotations']:
                ann['id'] = next_annotation_id
                ann['image_id'] = image_id_mapping[ann['image_id']]
                ann['category_id'] = 0  # 修改：统一设为smoke类别ID 0
                all_annotations.append(ann)
                next_annotation_id += 1
        
        # 处理UAV数据集
        print("处理UAV数据集...")
        uav_anno_dir = self.uav_root / "annotations" / "COCO_UAV" / "Annotations"
        uav_img_dir = self.uav_root / "images"
        
        for split in ['train', 'val', 'test']:
            anno_file = uav_anno_dir / f"{split}.json"
            if not anno_file.exists():
                print(f"警告: {anno_file} 不存在，跳过")
                continue
                
            coco_data = self.load_coco_annotation(str(anno_file))
            coco_data = self.filter_smoke_annotations(coco_data)
            
            # 重新映射图像ID
            image_id_mapping = {}
            for img in coco_data['images']:
                old_id = img['id']
                img['id'] = next_image_id
                image_id_mapping[old_id] = next_image_id
                
                # 添加数据集来源标识
                img['dataset'] = 'UAV'
                img['original_split'] = split
                all_images.append(img)
                next_image_id += 1
            
            # 重新映射标注ID和图像ID
            for ann in coco_data['annotations']:
                ann['id'] = next_annotation_id
                ann['image_id'] = image_id_mapping[ann['image_id']]
                ann['category_id'] = 0  # 修改：统一设为smoke类别ID 0
                all_annotations.append(ann)
                next_annotation_id += 1
        
        print(f"合并完成: 总共 {len(all_images)} 张图像，{len(all_annotations)} 个标注")
        
        # 创建合并后的COCO数据结构
        merged_data = {
            'info': {
                'description': 'FASDD Reorganized Dataset (CV + UAV, Smoke annotations only, All images including negatives)',
                'version': '1.0',
                'year': 2024,
                'contributor': 'Dataset Reorganizer',
                'date_created': '2024-01-01'
            },
            'licenses': [{'url': None, 'id': 0, 'name': None}],
            'categories': [{'id': 0, 'name': 'smoke', 'supercategory': 'object'}],  # 修改：ID改为0
            'images': all_images,
            'annotations': all_annotations
        }
        
        return merged_data
    
    def split_train_val(self, merged_data: Dict, train_ratio: float = 0.8) -> Tuple[Dict, Dict]:
        """
        重新划分训练集和验证集
        
        Args:
            merged_data: 合并后的数据
            train_ratio: 训练集比例
            
        Returns:
            (train_data, val_data)
        """
        print(f"重新划分数据集，训练集比例: {train_ratio}")
        
        # 获取所有图像
        all_images = merged_data['images'].copy()
        random.shuffle(all_images)
        
        # 按比例分割
        split_idx = int(len(all_images) * train_ratio)
        train_images = all_images[:split_idx]
        val_images = all_images[split_idx:]
        
        print(f"训练集: {len(train_images)} 张图像")
        print(f"验证集: {len(val_images)} 张图像")
        
        # 获取对应的标注
        train_image_ids = set(img['id'] for img in train_images)
        val_image_ids = set(img['id'] for img in val_images)
        
        train_annotations = []
        val_annotations = []
        
        for ann in merged_data['annotations']:
            if ann['image_id'] in train_image_ids:
                train_annotations.append(ann)
            elif ann['image_id'] in val_image_ids:
                val_annotations.append(ann)
        
        print(f"训练集标注: {len(train_annotations)} 个")
        print(f"验证集标注: {len(val_annotations)} 个")
        
        # 创建训练集和验证集数据结构
        base_info = merged_data.copy()
        del base_info['images']
        del base_info['annotations']
        
        train_data = base_info.copy()
        train_data['images'] = train_images
        train_data['annotations'] = train_annotations
        
        val_data = base_info.copy()
        val_data['images'] = val_images
        val_data['annotations'] = val_annotations
        
        return train_data, val_data
    
    def copy_images(self, train_data: Dict, val_data: Dict):
        """复制图像文件到输出目录"""
        print("开始复制图像文件...")
        
        # 合并所有图像信息
        all_images = train_data['images'] + val_data['images']
        
        for img in all_images:
            filename = img['file_name']
            dataset = img['dataset']
            
            # 确定源文件路径
            if dataset == 'CV':
                src_path = self.cv_root / "images" / filename
            else:  # UAV
                src_path = self.uav_root / "images" / filename
            
            # 目标文件路径
            dst_path = self.output_images_dir / filename
            
            # 复制文件
            if src_path.exists():
                if not dst_path.exists():
                    shutil.copy2(src_path, dst_path)
            else:
                print(f"警告: 源文件不存在 {src_path}")
        
        print(f"图像复制完成，输出目录: {self.output_images_dir}")
    
    def reorganize(self, train_ratio: float = 0.8, random_seed: int = 42):
        """
        执行数据集重组
        
        Args:
            train_ratio: 训练集比例
            random_seed: 随机种子
        """
        print("=" * 60)
        print("FASDD数据集重组开始")
        print("=" * 60)
        
        # 设置随机种子
        random.seed(random_seed)
        
        # 1. 合并数据集并过滤smoke类别
        merged_data = self.merge_datasets()
        
        # 2. 重新划分训练集和验证集
        train_data, val_data = self.split_train_val(merged_data, train_ratio)
        
        # 3. 保存标注文件
        print("保存标注文件...")
        self.save_coco_annotation(train_data, str(self.output_annotations_dir / "train.json"))
        self.save_coco_annotation(val_data, str(self.output_annotations_dir / "val.json"))
        
        # 4. 复制图像文件
        self.copy_images(train_data, val_data)
        
        # 统计正负样本信息
        train_image_ids_with_smoke = set(ann['image_id'] for ann in train_data['annotations'])
        val_image_ids_with_smoke = set(ann['image_id'] for ann in val_data['annotations'])
        
        train_positive = len(train_image_ids_with_smoke)
        train_negative = len(train_data['images']) - train_positive
        val_positive = len(val_image_ids_with_smoke)
        val_negative = len(val_data['images']) - val_positive
        
        print("=" * 60)
        print("数据集重组完成!")
        print(f"输出目录: {self.output_root}")
        print(f"  - 训练集: {len(train_data['images'])} 张图像 (正样本: {train_positive}, 负样本: {train_negative}), {len(train_data['annotations'])} 个标注")
        print(f"  - 验证集: {len(val_data['images'])} 张图像 (正样本: {val_positive}, 负样本: {val_negative}), {len(val_data['annotations'])} 个标注")
        print("=" * 60)


def main():
    """主函数"""
    # 配置路径
    cv_root = "datasets/data/FASDD_CV"
    uav_root = "datasets/data/FASDD_UAV"
    output_root = "datasets/data/FASDD_Reorganized"
    
    # 创建重组器并执行重组
    reorganizer = FASSDDatasetReorganizer(cv_root, uav_root, output_root)
    reorganizer.reorganize(train_ratio=0.8, random_seed=42)


if __name__ == "__main__":
    main()