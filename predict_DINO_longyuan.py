# !/usr/bin/env python3 
# -*- coding: utf-8 -*-
"""
Copyright 2024 Baidu Inc. All Rights Reserved.
2024/1/8, by zhangyi82@baidu.com, create

DESCRIPTION
【选手编写】预测脚本
"""
import pandas as pd
import os
import time
import logging
import torch
import torchvision
from PIL import Image
import cv2
import csv
import numpy as np
from glob import glob

# DINO相关导入
from main import build_model_main
from util.slconfig import SLConfig
from util import box_ops
import datasets.transforms as T

# 系统默认配置
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
# 此处设置您的队伍名称
os.environ['MATCH_TEAM_NAME'] = "硅纺锤体"

class Predictor(object):
    """
    InitModel函数  模型初始化参数，注意不能自行增加删除函数入参
    ret            是否正常: 正常True,异常False
    err_message    错误信息: 默认normal
    return ret,err_message
    备注说明:比赛使用上传模型的方式，模型路径等请使用相对路径
    """

    def __init__(self):
        # 模型参数
        self.model = None
        self.criterion = None
        self.postprocessors = None
        self.transform = None
        self.threshold = 0.3  # 检测阈值
    
    def InitModel(self):
        ret = True
        err_message = None
        '''
        模型初始化,由用户自行编写
        加载出错时,给ret和err_message赋值相应的错误
        '''
        try:
            logging.info("开始初始化DINO模型...")
            
            # 模型配置路径（使用相对路径）
            model_config_path = "config/DINO/DINO_4scale_swin_FASDD_DFire_dist.py"
            model_checkpoint_path = "logs/DINO/checkpoint0009.pth"
            
            # 检查文件是否存在
            if not os.path.exists(model_config_path):
                ret = False
                err_message = f"[Error] 模型配置文件不存在: {model_config_path}"
                return ret, err_message
                
            if not os.path.exists(model_checkpoint_path):
                ret = False
                err_message = f"[Error] 模型权重文件不存在: {model_checkpoint_path}"
                return ret, err_message
            
            # 加载模型配置
            args = SLConfig.fromfile(model_config_path)
            args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # 构建模型
            self.model, self.criterion, self.postprocessors = build_model_main(args)
            
            # 加载权重
            checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
            self.model.load_state_dict(checkpoint['model'])
            
            # 将模型移到GPU（如果可用）
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            
            # 设置为评估模式
            self.model.eval()
            
            # 设置图像预处理
            self.transform = T.Compose([
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            logging.info("DINO模型初始化成功！")
            
        except Exception as e:
            ret = False
            err_message = f"[Error] 模型初始化失败: {str(e)}"
            logging.error(err_message)
            
        return ret, err_message

    def Detect(self, input_data_file):
        """
        模型预测函数，注意调用该函数时会进行计时，后续将用来计算到模型性能得分
        input_data_file: 输入数据的绝/相对路径
        
        return:
        detect_result: 推理结果的存放路径
        err_message: 模型预测错误信息
        """
        err_message = None
        detect_result = None
        
        try:
            logging.info(f"开始处理输入目录: {input_data_file}")
            
            # 确定输出目录
            if "A组" in input_data_file:
                output_dir = './dataset/pre_res/A'
            else:
                output_dir = './dataset/pre_res/B'
            
            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)
            detect_result = output_dir
            
            # 获取所有图片文件
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
            image_files = []
            for ext in image_extensions:
                image_files.extend(glob(os.path.join(input_data_file, ext)))
                image_files.extend(glob(os.path.join(input_data_file, ext.upper())))
            
            if len(image_files) == 0:
                err_message = f"[Warning] 在目录 {input_data_file} 中未找到图片文件"
                logging.warning(err_message)
                return detect_result, err_message
            
            logging.info(f"发现 {len(image_files)} 张图片")
            
            # 处理每张图片
            for i, image_path in enumerate(image_files):
                try:
                    # 获取图片文件名（不包含扩展名）
                    base_name = os.path.splitext(os.path.basename(image_path))[0]
                    txt_path = os.path.join(output_dir, f"{base_name}.txt")
                    
                    # 加载和预处理图片
                    image = Image.open(image_path).convert("RGB")
                    original_size = image.size  # (width, height)
                    
                    # 图片变换
                    processed_image, _ = self.transform(image, None)
                    
                    # 模型推理
                    with torch.no_grad():
                        # 将图片移到对应设备
                        if torch.cuda.is_available():
                            processed_image = processed_image.cuda()
                        
                        # 前向推理
                        output = self.model(processed_image[None])
                        
                        # 后处理，传入原始图像尺寸 [height, width] 格式
                        target_sizes = torch.tensor([[original_size[1], original_size[0]]])
                        if torch.cuda.is_available():
                            target_sizes = target_sizes.cuda()
                        
                        output = self.postprocessors['bbox'](output, target_sizes)[0]
                    
                    # 获取预测结果
                    scores = output['scores']
                    labels = output['labels']
                    boxes = output['boxes']  # 绝对像素坐标，XYXY格式
                    
                    # 应用阈值过滤
                    select_mask = scores > self.threshold
                    
                    # 清空txt文件并写入检测结果
                    with open(txt_path, 'w') as f:
                        if select_mask.sum() > 0:
                            filtered_scores = scores[select_mask]
                            filtered_labels = labels[select_mask]
                            filtered_boxes = boxes[select_mask]
                            
                            for j in range(len(filtered_scores)):
                                # 格式: 类别代码(0) x1 y1 x2 y2 置信度
                                class_id = 0  # 烟雾检测，类别代码统一为0
                                x1, y1, x2, y2 = filtered_boxes[j].cpu().numpy()
                                confidence = filtered_scores[j].cpu().item()
                                
                                # 确保坐标为整数像素值
                                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                
                                # 确保坐标在图像范围内
                                x1 = max(0, min(x1, original_size[0]))
                                y1 = max(0, min(y1, original_size[1]))
                                x2 = max(0, min(x2, original_size[0]))
                                y2 = max(0, min(y2, original_size[1]))
                                
                                # 写入结果
                                f.write(f"{class_id} {x1} {y1} {x2} {y2} {confidence:.4f}\n")
                    
                    # 打印进度
                    if (i + 1) % 10 == 0 or (i + 1) == len(image_files):
                        logging.info(f"已处理: {i + 1}/{len(image_files)} 张图片")
                        
                except Exception as e:
                    logging.error(f"处理图片 {image_path} 时出错: {str(e)}")
                    continue
            
            logging.info(f"批量推理完成！结果保存在: {detect_result}")
            
        except Exception as e:
            err_message = f"[Error] 检测过程中发生错误: {str(e)}"
            logging.error(err_message)
        
        return detect_result, err_message

    def data_preprocess(self, input_data_file):
        """
        数据预处理
        param input_data_file: 输入数据的路径
        return df: pandas.DataFrame格式，预处理好的数据
               err_message: 处理中途报错信息，如果没有就是None
        """
        err_message = None
        logging.info(f"Predictor.Detect函数的input_data_file is: {input_data_file}")
        df = None
        return df, err_message


if __name__ == '__main__':
    # 备注说明:main函数提供给用户内测,修改后[不影响]评估
    predictor = Predictor()
    
    # 初始化模型
    ret, err_message = predictor.InitModel()
    
    # 模型预测（测试用）
    if ret:
        # 测试A组数据
        test_input_A = "datasets/data/longyuan_dataset/input_data/A组/images"
        if os.path.exists(test_input_A):
            detect_result, detect_err_message = predictor.Detect(test_input_A)
            if detect_err_message is None:
                logging.info(f"A组检测完成，结果保存在: {detect_result}")
            else:
                logging.error(f"[Error] A组检测失败: {detect_err_message}")
        
        # 测试B组数据
        test_input_B = "datasets/data/longyuan_dataset/input_data/B组/images"
        if os.path.exists(test_input_B):
            detect_result, detect_err_message = predictor.Detect(test_input_B)
            if detect_err_message is None:
                logging.info(f"B组检测完成，结果保存在: {detect_result}")
            else:
                logging.error(f"[Error] B组检测失败: {detect_err_message}")
    else:
        logging.error(f"[Error] InitModel failed. ret is {ret}, err_message is {err_message}")