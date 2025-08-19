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

# 系统默认配置
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
#此处设置您的队伍名称
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
    
    def InitModel(self):
        ret = True
        err_message = None
        '''
        模型初始化,由用户自行编写
        加载出错时,给ret和err_message赋值相应的错误
        例如
        ret=False
        err_message = "[Error] model_path: [{}] init failed".format(model_path)
        '''
        try:
            #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            #print(f'Using device: {device}')
            # # 加载模型
            #model_path = "model/test.onnx"
        
        except:
            ret, err_message = False, "loading model error"
        return ret, err_message

    def Detect(self, input_data_file):
        """
        模型预测函数，注意调用该函数时会进行计时，后续将用来计算到模型性能得分
        input_data_file: 输入数据的绝/相对路径，举例如下:
        dataset.json的data_path的字段值 + "/input_data/images/"
        选手本地测试时，也需要按照上面的目录结构进行存放
        
        return:(可参考下面的示例代码)
        detect_result: path   为推理结果的存放路径，是选手自己定义的临时目录（多为程序运行时自动生成）
        推理结果为txt文件，文件前缀为推理的图片名称前缀
        txt文件中的内容为:
        0 595 554 1009 735 0.78 
        第1位‘0’为检测的类别代码，如果检测类别只有一个，则为0
        第2-5为检测目标的坐标框，VOC格式为Xmin,Ymin,Xmax,Ymax
        第6位检测目标的置信度

        err_message: 模型预测错误信息
        """
        err_message = None
        # 数据预处理
        #processed_input_data, err_message = self.data_preprocess(input_data_file)
        # 模拟模型预测
        #output = self.model(input_data_file)
        
        for file in os.listdir(input_data_file):
            if file.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                img_path = os.path.join(input_data_file, file)
                image = cv2.imread(img_path)
                boxes, scores, class_ids = self.model(image)
                base_name = os.path.splitext(file)[0]  # 获取文件名前缀
                txt_path = f"{base_name}.txt"  # 生成txt文件名
                
                #生成临时结果存放文件夹
                if "A组" in os.path.dirname(input_data_file):
                  txtpath = './dataset/pre_res/A'
                else:
                  txtpath = './dataset/pre_res/B'
                if not os.path.exists(txtpath):
                  os.makedirs(txtpath)
                detect_result = txtpath
                txtfilepath = os.path.join(txtpath,txt_path)
                if os.path.exists(txtfilepath) and os.path.getsize(txtfilepath) > 0:
                    with open(txtfilepath, 'w') as f:
                      f.write('')
                      f.close()
                with open(txtfilepath, 'w') as f:
                    for box, score, class_id in zip(boxes, scores, class_ids):
                       # 假设box是[x1, y1, x2, y2]格式，并且已经归一化
                       x1, y1, x2, y2 = box
                       # 转换回原始尺寸
                       x1_orig = int(x1)
                       y1_orig = int(y1)
                       x2_orig = int(x2)
                       y2_orig = int(y2)
                       # 写入txt文件
                       f.write(f"{class_id} {x1_orig} {y1_orig} {x2_orig} {y2_orig} {score:.4f}\n")
                    f.close()
                if len(os.listdir('./dataset/pre_res')) == 0:
                  err_message = False
              #print(detect_result)
        return detect_result,err_message

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

    # 模型预测
    if ret:
        detect_result, detect_err_message = predictor.Detect()
        if err_message is None:
            logging.info(f"Detect_result:\n{detect_result}")
        else:
            logging.error(f"[Error] Detect failed. {detect_err_message}")
    else:
        logging.error(f"[Error] InitModel failed. ret is {ret}, err_message is {err_message}")
