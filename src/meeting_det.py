from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import sys
from opts import opts
from detectors.detector_factory import detector_factory

opt = opts().init()
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
opt.debug = max(opt.debug, 1)
Detector = detector_factory[opt.task]
detector = Detector(opt)

def meeting_det(image, opt=opt):
    start_time = time.time()
    concentrate_rate, activity_rate = detector.run(image)
    total_time = time.time() - start_time
    print('total_time:{:.2f}s'.format(total_time))
    return concentrate_rate, activity_rate

if __name__ == '__main__':
    # # 以下代码是读取图片生成视频视频，不用时注释
    # fourcc = cv2.VideoWriter_fourcc(*'XVID') #使用XVID编码器
    # out = cv2.VideoWriter('../output/output.avi',fourcc, 2.0, (1920,1080))#生成视频用，参数分别是：保存文件名、编码器、帧率、视频宽高
    # path = '../images/'
    # images = os.listdir('../images/')
    # images.sort()
    # print(images)
    # for image in images:
      # img=cv2.imread(path + image)
      # out.write(img)
    # print('视频制作已完成！')
  # 以下是测试代码
  concentrate_rate, activity_rate = [0,0]
  if opt.demo == 'webcam' or \
    opt.demo[opt.demo.rfind('.') + 1:].lower() in ['avi']:
    cam = cv2.VideoCapture(0 if opt.demo == 'webcam' else opt.demo)
    detector.pause = False
    
    #先读取一张图片，得到视频长宽
    _, img = cam.read()
    h, w = img.shape[:2]
    #保存视频
    fps = cam.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID') #使用XVID编码器
    out = cv2.VideoWriter('../output/output.avi',fourcc, fps, (w,h))#读取视频用，参数分别是：保存文件名、编码器、帧率、视频宽高
    

    concentrate = []
    activity = []
    while True:
        try:
            start_time = time.time()
            _, img = cam.read()
            concentrate_rate, activity_rate = meeting_det(img) #对单一图片测试
            concentrate.append(concentrate_rate)
            activity.append(activity_rate)
            # print('concentrate:{} activity:{}'.format(len(concentrate),len(activity)))
        except TypeError:
            plt.plot(concentrate)
            plt.show()
            sys.exit()