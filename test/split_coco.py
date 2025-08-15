import json
import os
from collections import defaultdict

def split_coco_annotations(annotation_file, output_dir):
    """
    将一个大的 COCO JSON标注文件分割成每个图片一个小JSON文件。

    参数:
    - annotation_file (str): 原始COCO标注文件的路径。
    - output_dir (str): 保存分割后的小JSON文件的目录。
    """
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建目录: {output_dir}")

    print("正在加载原始JSON文件，请稍候...")
    with open(annotation_file, 'r') as f:
        data = json.load(f)

    # 提取公共信息
    info = data.get('info', {})
    licenses = data.get('licenses', [])
    categories = data.get('categories', [])

    print("按 image_id 对标注进行分组...")
    annotations_by_image = defaultdict(list)
    for ann in data['annotations']:
        annotations_by_image[ann['image_id']].append(ann)

    print(f"开始为 {len(data['images'])} 张图片生成独立的JSON文件...")
    # 为每张图片创建独立的JSON文件
    for image_info in data['images']:
        image_id = image_info['id']
        
        # 构建单个图片的COCO格式数据
        single_image_data = {
            'info': info,
            'licenses': licenses,
            'categories': categories,
            'images': [image_info],  # 只包含当前图片的信息
            'annotations': annotations_by_image.get(image_id, []) # 获取该图片对应的所有标注
        }
        
        # 定义输出文件名
        output_filename = os.path.join(output_dir, f"{image_id}.json")
        
        # 保存为新的JSON文件
        with open(output_filename, 'w') as f:
            json.dump(single_image_data, f, indent=4)

    print(f"处理完成！分割后的文件已保存到 {output_dir}")

# --- 使用方法 ---
if __name__ == '__main__':
    # 设置你的文件路径
    original_annotation_path = 'datasets/data/FASDD_CV/annotations/COCO_CV/Annotations/test.json'
    output_directory = 'test/splitted_annotations/FASDD_CV_test'

    # 运行分割函数
    split_coco_annotations(original_annotation_path, output_directory)