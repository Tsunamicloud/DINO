#!/usr/bin/env python3
"""
FASDD数据集标注可视化脚本
将FASDD_Reorganized数据集中的标注绘制到图片上，保存到FASDD_Visualized目录
用于视觉检查标注的正确性
"""

import json
import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm


class AnnotationVisualizer:
    def __init__(self, dataset_root: str, output_root: str):
        """
        初始化可视化器
        
        Args:
            dataset_root: FASDD_Reorganized数据集根目录
            output_root: 可视化结果输出目录
        """
        self.dataset_root = Path(dataset_root)
        self.output_root = Path(output_root)
        
        # 输入路径
        self.images_dir = self.dataset_root / "images"
        self.annotations_dir = self.dataset_root / "annotations"
        
        # 输出路径
        self.output_images_dir = self.output_root / "images"
        self.output_stats_dir = self.output_root / "statistics"
        
        # 创建输出目录
        for dir_path in [self.output_images_dir, self.output_stats_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 可视化配置
        self.colors = {
            'smoke': (0, 255, 0),      # 绿色
            'fire': (0, 0, 255),       # 红色
            'default': (255, 0, 0)     # 蓝色
        }
        
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.thickness = 2
        self.text_thickness = 1
    
    def load_coco_annotations(self, annotation_file: str) -> Dict:
        """加载COCO格式标注文件"""
        try:
            with open(annotation_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error: 加载标注文件失败 {annotation_file}: {e}")
            return None
    
    def draw_bbox(self, image: np.ndarray, bbox: List[float], category_name: str, 
                  annotation_id: int = None) -> np.ndarray:
        """
        在图像上绘制边界框
        
        Args:
            image: 输入图像
            bbox: COCO格式边界框 [x, y, width, height]
            category_name: 类别名称
            annotation_id: 标注ID
            
        Returns:
            绘制后的图像
        """
        x, y, w, h = [int(coord) for coord in bbox]
        
        # 获取颜色
        color = self.colors.get(category_name.lower(), self.colors['default'])
        
        # 绘制边界框
        cv2.rectangle(image, (x, y), (x + w, y + h), color, self.thickness)
        
        # 准备标签文本
        label = category_name
        if annotation_id is not None:
            label += f" #{annotation_id}"
        
        # 计算文本尺寸
        (text_width, text_height), baseline = cv2.getTextSize(
            label, self.font, self.font_scale, self.text_thickness
        )
        
        # 绘制文本背景
        text_bg_top_left = (x, y - text_height - baseline - 5)
        text_bg_bottom_right = (x + text_width + 5, y)
        cv2.rectangle(image, text_bg_top_left, text_bg_bottom_right, color, -1)
        
        # 绘制文本
        text_pos = (x + 2, y - baseline - 2)
        cv2.putText(image, label, text_pos, self.font, self.font_scale, 
                   (255, 255, 255), self.text_thickness)
        
        return image
    
    def visualize_image(self, image_info: Dict, annotations: List[Dict], 
                       categories: Dict[int, str]) -> Tuple[np.ndarray, Dict]:
        """
        可视化单张图像的标注
        
        Args:
            image_info: 图像信息
            annotations: 该图像的所有标注
            categories: 类别映射 {id: name}
            
        Returns:
            (可视化后的图像, 统计信息)
        """
        # 加载图像
        image_path = self.images_dir / image_info['file_name']
        if not image_path.exists():
            print(f"Warning: 图像文件不存在 {image_path}")
            return None, None
        
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Warning: 无法读取图像 {image_path}")
            return None, None
        
        # 统计信息
        stats = {
            'image_id': image_info['id'],
            'file_name': image_info['file_name'],
            'image_size': (image_info['width'], image_info['height']),
            'annotations_count': len(annotations),
            'categories_count': {}
        }
        
        # 绘制所有标注
        for ann in annotations:
            category_id = ann['category_id']
            category_name = categories.get(category_id, f'unknown_{category_id}')
            
            # 更新统计
            if category_name not in stats['categories_count']:
                stats['categories_count'][category_name] = 0
            stats['categories_count'][category_name] += 1
            
            # 绘制边界框
            image = self.draw_bbox(image, ann['bbox'], category_name, ann['id'])
        
        # 在图像上添加总体信息
        info_text = f"Image: {image_info['file_name']} | Annotations: {len(annotations)}"
        cv2.putText(image, info_text, (10, 30), self.font, 0.7, (255, 255, 255), 2)
        cv2.putText(image, info_text, (10, 30), self.font, 0.7, (0, 0, 0), 1)
        
        return image, stats
    
    def create_summary_image(self, stats_list: List[Dict], split_name: str) -> np.ndarray:
        """创建统计摘要图像"""
        # 创建空白图像
        img_height = 800
        img_width = 1200
        summary_img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
        
        # 标题
        title = f"{split_name.upper()} Split Visualization Summary"
        cv2.putText(summary_img, title, (50, 50), self.font, 1.2, (0, 0, 0), 2)
        
        # 统计信息
        total_images = len(stats_list)
        total_annotations = sum(s['annotations_count'] for s in stats_list if s)
        
        # 类别统计
        category_counts = {}
        for stats in stats_list:
            if stats and 'categories_count' in stats:
                for cat, count in stats['categories_count'].items():
                    category_counts[cat] = category_counts.get(cat, 0) + count
        
        # 正负样本统计
        positive_samples = sum(1 for s in stats_list if s and s['annotations_count'] > 0)
        negative_samples = total_images - positive_samples
        
        # 绘制统计信息
        y_offset = 120
        line_height = 35
        
        stats_text = [
            f"Total Images: {total_images:,}",
            f"Total Annotations: {total_annotations:,}",
            f"Positive Samples: {positive_samples:,} ({positive_samples/total_images*100:.1f}%)",
            f"Negative Samples: {negative_samples:,} ({negative_samples/total_images*100:.1f}%)",
            "",
            "Category Distribution:"
        ]
        
        for cat, count in category_counts.items():
            percentage = count / total_annotations * 100 if total_annotations > 0 else 0
            stats_text.append(f"  {cat}: {count:,} annotations ({percentage:.1f}%)")
        
        for i, text in enumerate(stats_text):
            y_pos = y_offset + i * line_height
            if y_pos < img_height - 50:
                cv2.putText(summary_img, text, (50, y_pos), self.font, 0.8, (0, 0, 0), 1)
        
        return summary_img
    
    def process_split(self, split_name: str, visualize_negatives: bool = True) -> List[Dict]:
        """
        处理单个split的数据
        
        Args:
            split_name: split名称 (train/val)
            visualize_negatives: 是否可视化负样本
            
        Returns:
            统计信息列表
        """
        print(f"\n开始可视化 {split_name} split...")
        
        # 加载标注文件
        annotation_file = self.annotations_dir / f"{split_name}.json"
        if not annotation_file.exists():
            print(f"Error: 标注文件不存在 {annotation_file}")
            return []
        
        coco_data = self.load_coco_annotations(str(annotation_file))
        if not coco_data:
            return []
        
        # 构建类别映射
        categories = {cat['id']: cat['name'] for cat in coco_data.get('categories', [])}
        
        # 构建图像ID到标注的映射
        annotations_by_image = {}
        for ann in coco_data.get('annotations', []):
            image_id = ann['image_id']
            if image_id not in annotations_by_image:
                annotations_by_image[image_id] = []
            annotations_by_image[image_id].append(ann)
        
        # 获取所有图像列表
        images = coco_data.get('images', [])
        
        # 分类图像
        positive_images = []
        negative_images = []
        
        for img in images:
            if img['id'] in annotations_by_image:
                positive_images.append(img)
            else:
                negative_images.append(img)
        
        print(f"  图像分布: {len(positive_images)} 正样本, {len(negative_images)} 负样本")
        
        # 创建输出目录
        split_output_dir = self.output_images_dir / split_name
        positive_dir = split_output_dir / "positive"
        negative_dir = split_output_dir / "negative"
        
        for dir_path in [positive_dir, negative_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        stats_list = []
        
        # 处理正样本
        if positive_images:
            print(f"  处理 {len(positive_images)} 个正样本...")
            for img_info in tqdm(positive_images, desc="  正样本"):
                annotations = annotations_by_image.get(img_info['id'], [])
                
                # 可视化
                vis_image, stats = self.visualize_image(img_info, annotations, categories)
                if vis_image is not None and stats is not None:
                    # 保存图像
                    output_path = positive_dir / img_info['file_name']
                    cv2.imwrite(str(output_path), vis_image)
                    stats_list.append(stats)
        
        # 处理负样本
        if negative_images and visualize_negatives:
            print(f"  处理 {len(negative_images)} 个负样本...")
            for img_info in tqdm(negative_images, desc="  负样本"):
                # 可视化（无标注）
                vis_image, stats = self.visualize_image(img_info, [], categories)
                if vis_image is not None and stats is not None:
                    # 保存图像
                    output_path = negative_dir / img_info['file_name']
                    cv2.imwrite(str(output_path), vis_image)
                    stats_list.append(stats)
        
        # 创建并保存统计摘要
        summary_img = self.create_summary_image(stats_list, split_name)
        summary_path = self.output_stats_dir / f"{split_name}_summary.png"
        cv2.imwrite(str(summary_path), summary_img)
        
        print(f"  {split_name} split 完成，可视化了 {len(stats_list)} 张图像")
        
        return stats_list
    
    def visualize_dataset(self, visualize_negatives: bool = True):
        """
        可视化整个数据集
        
        Args:
            visualize_negatives: 是否可视化负样本
        """
        print("=" * 60)
        print("FASDD数据集标注可视化开始")
        print("=" * 60)
        
        if not self.dataset_root.exists():
            print(f"Error: 数据集目录不存在 {self.dataset_root}")
            return
        
        all_stats = {}
        
        # 处理每个split
        for split in ['train', 'val']:
            stats = self.process_split(split, visualize_negatives=visualize_negatives)
            all_stats[split] = stats
        
        # 生成整体统计
        self.generate_overall_summary(all_stats)
        
        print("\n" + "=" * 60)
        print("可视化完成")
        print(f"输出目录: {self.output_root}")
        print("目录结构:")
        print("  images/")
        print("    train/")
        print("      positive/  - 包含标注的图像")
        print("      negative/  - 不包含标注的图像")
        print("    val/")
        print("      positive/")
        print("      negative/")
        print("  statistics/")
        print("    train_summary.png  - 训练集统计摘要")
        print("    val_summary.png    - 验证集统计摘要")
        print("    overall_summary.png - 整体统计摘要")
        print("=" * 60)
    
    def generate_overall_summary(self, all_stats: Dict[str, List[Dict]]):
        """生成整体统计摘要"""
        # 创建整体摘要图像
        img_height = 1000
        img_width = 1400
        summary_img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
        
        # 标题
        title = "FASDD Dataset Visualization Overall Summary"
        cv2.putText(summary_img, title, (50, 50), self.font, 1.5, (0, 0, 0), 3)
        
        y_offset = 120
        line_height = 40
        
        # 为每个split生成统计
        for split_name, stats_list in all_stats.items():
            if not stats_list:
                continue
                
            cv2.putText(summary_img, f"{split_name.upper()} Split:", 
                       (50, y_offset), self.font, 1.1, (0, 0, 255), 2)
            y_offset += line_height
            
            total_images = len(stats_list)
            total_annotations = sum(s['annotations_count'] for s in stats_list if s)
            positive_samples = sum(1 for s in stats_list if s and s['annotations_count'] > 0)
            negative_samples = total_images - positive_samples
            
            split_info = [
                f"  Images visualized: {total_images:,}",
                f"  Annotations: {total_annotations:,}",
                f"  Positive samples: {positive_samples:,} ({positive_samples/total_images*100:.1f}%)",
                f"  Negative samples: {negative_samples:,} ({negative_samples/total_images*100:.1f}%)"
            ]
            
            for info in split_info:
                cv2.putText(summary_img, info, (70, y_offset), self.font, 0.8, (0, 0, 0), 1)
                y_offset += line_height - 10
            
            y_offset += 20
        
        # 保存整体摘要
        summary_path = self.output_stats_dir / "overall_summary.png"
        cv2.imwrite(str(summary_path), summary_img)


def main():
    """主函数"""
    # 配置路径
    dataset_root = "datasets/data/FASDD_DFire_Dataset"
    output_root = "datasets/data/FASDD_DFire_Dataset_Visualized"
    
    # 创建可视化器
    visualizer = AnnotationVisualizer(dataset_root, output_root)
    
    # 开始可视化
    visualizer.visualize_dataset(visualize_negatives=True)


if __name__ == "__main__":
    main()