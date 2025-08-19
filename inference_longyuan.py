import os, sys
import torch, json
import numpy as np
from glob import glob

from main import build_model_main
from util.slconfig import SLConfig
from util import box_ops

from PIL import Image
import datasets.transforms as T

def inference_batch_longyuan():
    # Model configuration
    model_config_path = "config/DINO/DINO_4scale_swin_FASDD_DFire_dist.py"  # 您的模型配置文件路径
    model_checkpoint_path = "logs/DINO/checkpoint0009.pth"  # 您的模型权重文件路径
    
    # Load model
    args = SLConfig.fromfile(model_config_path) 
    args.device = 'cuda' 
    model, criterion, postprocessors = build_model_main(args)
    checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model = model.cuda()
    _ = model.eval()
    
    # Image preprocessing
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Input and output directories
    input_dirs = [
        "datasets/data/longyuan_dataset/input_data/A组/images",
        "datasets/data/longyuan_dataset/input_data/B组/images"
    ]
    
    output_base_dir = "datasets/data/longyuan_dataset/DINO_output_results"
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Detection threshold
    threshold = 0.3  # 您可以根据需要调整阈值
    
    print("开始批量推理...")
    
    for input_dir in input_dirs:
        if not os.path.exists(input_dir):
            print(f"目录不存在: {input_dir}")
            continue
            
        # 获取目录名称用于输出文件夹命名
        group_name = input_dir.split(os.sep)[-2]  # A组 或 B组
        output_dir = os.path.join(output_base_dir, group_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取所有图片文件
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob(os.path.join(input_dir, ext)))
            image_files.extend(glob(os.path.join(input_dir, ext.upper())))
        
        print(f"处理 {group_name}: 发现 {len(image_files)} 张图片")
        
        for i, image_path in enumerate(image_files):
            try:
                # 加载和预处理图片
                image = Image.open(image_path).convert("RGB")
                original_size = image.size  # (width, height)
                
                # 图片变换
                processed_image, _ = transform(image, None)
                
                # 模型推理
                with torch.no_grad():
                    output = model(processed_image[None].cuda())
                    # 传入原始图像尺寸 [height, width] 格式
                    target_sizes = torch.tensor([[original_size[1], original_size[0]]]).cuda()  # [height, width]
                    output = postprocessors['bbox'](output, target_sizes)[0]
                
                # 获取预测结果 - 现在已经是绝对像素坐标
                scores = output['scores']
                labels = output['labels']
                boxes = output['boxes']  # 绝对像素坐标，XYXY格式
                
                # 应用阈值过滤
                select_mask = scores > threshold
                
                # 准备输出文件
                image_name = os.path.splitext(os.path.basename(image_path))[0]
                output_txt_path = os.path.join(output_dir, f"{image_name}.txt")
                
                # 写入检测结果
                with open(output_txt_path, 'w') as f:
                    if select_mask.sum() > 0:
                        filtered_scores = scores[select_mask]
                        filtered_labels = labels[select_mask]
                        filtered_boxes = boxes[select_mask]
                        
                        for j in range(len(filtered_scores)):
                            # 格式: 类别代码(0) x1 y1 x2 y2 置信度
                            # 因为是烟雾检测，类别代码统一为0
                            class_id = 0
                            x1, y1, x2, y2 = filtered_boxes[j].cpu().numpy()
                            confidence = filtered_scores[j].cpu().item()
                            
                            # 确保坐标为整数像素值
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            
                            # 确保坐标在图像范围内
                            x1 = max(0, min(x1, original_size[0]))
                            y1 = max(0, min(y1, original_size[1]))
                            x2 = max(0, min(x2, original_size[0]))
                            y2 = max(0, min(y2, original_size[1]))
                            
                            f.write(f"{class_id} {x1} {y1} {x2} {y2} {confidence:.4f}\n")
                
                # 打印进度
                if (i + 1) % 10 == 0 or (i + 1) == len(image_files):
                    print(f"  已处理: {i + 1}/{len(image_files)} 张图片")
                    
            except Exception as e:
                print(f"处理图片 {image_path} 时出错: {str(e)}")
                continue
    
    print("批量推理完成！")
    print(f"结果保存在: {output_base_dir}")

if __name__ == "__main__":
    inference_batch_longyuan()