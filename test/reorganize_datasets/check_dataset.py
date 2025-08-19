#!/usr/bin/env python3
"""
FASDD数据集统计脚本
统计FASDD_UAV、FASDD_CV、FASDD_Reorganized三个数据集的详细信息
"""

import json
import os
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple


class DatasetStatistics:
    def __init__(self):
        self.datasets_info = {
            'FASDD_UAV': {
                'path': 'datasets/data/FASDD_UAV/annotations/COCO_UAV/Annotations',
                'splits': ['train', 'val', 'test']
            },
            'FASDD_CV': {
                'path': 'datasets/data/FASDD_CV/annotations/COCO_CV/Annotations', 
                'splits': ['train', 'val', 'test']
            },
            'FASDD_Reorganized': {
                'path': 'datasets/data/FASDD_Reorganized/annotations',
                'splits': ['train', 'val']  # 没有test split
            },
            'DFire_COCO': {
                'path': 'datasets/data/DFire_COCO/annotations',
                'splits': ['train', 'val']  # 没有test split
            },
            'FASDD_DFire_Dataset': {
                'path': 'datasets/data/FASDD_DFire_Dataset/annotations',
                'splits': ['train', 'val']  # 没有test split
            },
            'longyuan_final': {
                'path': 'datasets/data/longyuan_final/annotations',
                'splits': ['train', 'val']  # 没有test split
            }
        }
    
    def load_coco_file(self, file_path: str) -> Dict:
        """加载COCO格式文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"警告: 文件不存在 {file_path}")
            return None
        except Exception as e:
            print(f"错误: 加载文件失败 {file_path}: {e}")
            return None
    
    def analyze_split(self, coco_data: Dict) -> Dict:
        """分析单个split的统计信息"""
        if not coco_data:
            return {}
        
        stats = {
            'images_count': len(coco_data.get('images', [])),
            'annotations_count': len(coco_data.get('annotations', [])),
            'categories': {},
            'category_annotations': defaultdict(int),
            'category_images': defaultdict(set),
            'images_with_annotations': set(),
            'negative_samples': 0
        }
        
        # 分析类别信息
        for cat in coco_data.get('categories', []):
            stats['categories'][cat['id']] = cat['name']
        
        # 分析标注信息
        for ann in coco_data.get('annotations', []):
            cat_id = ann['category_id']
            cat_name = stats['categories'].get(cat_id, f'unknown_{cat_id}')
            
            stats['category_annotations'][cat_name] += 1
            stats['category_images'][cat_name].add(ann['image_id'])
            stats['images_with_annotations'].add(ann['image_id'])
        
        # 计算负样本数量
        stats['negative_samples'] = stats['images_count'] - len(stats['images_with_annotations'])
        
        # 转换set为count
        for cat_name in stats['category_images']:
            stats['category_images'][cat_name] = len(stats['category_images'][cat_name])
        
        return stats
    
    def print_dataset_stats(self, dataset_name: str, all_stats: Dict):
        """打印数据集统计信息"""
        print(f"\n{'='*70}")
        print(f"{dataset_name} 数据集统计分析")
        print(f"{'='*70}")
        
        total_images = 0
        total_annotations = 0
        total_fire_annotations = 0
        total_smoke_annotations = 0
        total_fire_images = 0
        total_smoke_images = 0
        total_negative_samples = 0
        
        # 打印每个split的详细信息
        for split, stats in all_stats.items():
            if not stats:
                print(f"\n{split.upper()} 数据集: 数据不可用")
                continue
                
            print(f"\n{split.upper()} 数据集分析:")
            print(f"  图像总数: {stats['images_count']:,}")
            print(f"  标注总数: {stats['annotations_count']:,}")
            print(f"  负样本数: {stats['negative_samples']:,}")
            
            if stats['categories']:
                print(f"  类别分布:")
                for cat_id, cat_name in stats['categories'].items():
                    ann_count = stats['category_annotations'].get(cat_name, 0)
                    img_count = stats['category_images'].get(cat_name, 0)
                    print(f"    {cat_name}: {ann_count:,} 个标注，覆盖 {img_count:,} 张图像")
            
            # 累计统计
            total_images += stats['images_count']
            total_annotations += stats['annotations_count']
            total_negative_samples += stats['negative_samples']
            total_fire_annotations += stats['category_annotations'].get('fire', 0)
            total_smoke_annotations += stats['category_annotations'].get('smoke', 0)
            total_fire_images += stats['category_images'].get('fire', 0)
            total_smoke_images += stats['category_images'].get('smoke', 0)
        
        # 打印总计信息
        if total_images > 0:
            print(f"\n汇总统计:")
            print(f"  图像总数: {total_images:,}")
            print(f"  标注总数: {total_annotations:,}")
            print(f"  负样本总数: {total_negative_samples:,}")
            print(f"  火灾标注: {total_fire_annotations:,} 个 (覆盖 {total_fire_images:,} 张图像)")
            print(f"  烟雾标注: {total_smoke_annotations:,} 个 (覆盖 {total_smoke_images:,} 张图像)")
            
            if total_annotations > 0:
                fire_ratio = (total_fire_annotations / total_annotations) * 100
                smoke_ratio = (total_smoke_annotations / total_annotations) * 100
                print(f"  标注分布: 火灾 {fire_ratio:.1f}%, 烟雾 {smoke_ratio:.1f}%")
            
            positive_samples = total_images - total_negative_samples
            if total_images > 0:
                positive_ratio = (positive_samples / total_images) * 100
                negative_ratio = (total_negative_samples / total_images) * 100
                print(f"  样本分布: 正样本 {positive_samples:,} ({positive_ratio:.1f}%), 负样本 {total_negative_samples:,} ({negative_ratio:.1f}%)")
    
    def generate_comparison_table(self, all_dataset_stats: Dict):
        """生成数据集对比表"""
        print(f"\n{'='*90}")
        print(f"Dataset Comparison Summary")
        print(f"{'='*90}")
        
        # 表头
        print(f"{'Dataset':<20} {'Images':<10} {'Annotations':<12} {'Fire Ann.':<10} {'Smoke Ann.':<11} {'Negative':<10}")
        print(f"{'-'*90}")
        
        for dataset_name, stats_by_split in all_dataset_stats.items():
            total_images = sum(s.get('images_count', 0) for s in stats_by_split.values() if s)
            total_annotations = sum(s.get('annotations_count', 0) for s in stats_by_split.values() if s)
            total_fire = sum(s.get('category_annotations', {}).get('fire', 0) for s in stats_by_split.values() if s)
            total_smoke = sum(s.get('category_annotations', {}).get('smoke', 0) for s in stats_by_split.values() if s)
            total_negative = sum(s.get('negative_samples', 0) for s in stats_by_split.values() if s)
            
            print(f"{dataset_name:<20} {total_images:<10,} {total_annotations:<12,} {total_fire:<10,} {total_smoke:<11,} {total_negative:<10,}")
    
    def run_analysis(self):
        """运行完整的数据集分析"""
        print("正在启动FASDD数据集统计分析...")
        
        all_dataset_stats = {}
        
        for dataset_name, dataset_info in self.datasets_info.items():
            print(f"\n正在处理 {dataset_name} 数据集...")
            
            dataset_stats = {}
            for split in dataset_info['splits']:
                file_path = os.path.join(dataset_info['path'], f"{split}.json")
                print(f"  正在加载 {split}.json...")
                
                coco_data = self.load_coco_file(file_path)
                dataset_stats[split] = self.analyze_split(coco_data)
            
            all_dataset_stats[dataset_name] = dataset_stats
            self.print_dataset_stats(dataset_name, dataset_stats)
        
        # 生成对比表
        self.generate_comparison_table(all_dataset_stats)
        
        print(f"\n{'='*70}")
        print(f"数据集统计分析完成")
        print(f"{'='*70}")


def main():
    """主函数"""
    analyzer = DatasetStatistics()
    analyzer.run_analysis()


if __name__ == "__main__":
    main()