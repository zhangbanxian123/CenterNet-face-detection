from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import cv2
import time
import numpy as np

from opts import opts
from detectors.detector_factory import detector_factory

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']

def demo(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.debug = max(opt.debug, 1)
  Detector = detector_factory[opt.task]
  detector = Detector(opt)

  if opt.demo == 'webcam' or \
    opt.demo[opt.demo.rfind('.') + 1:].lower() in video_ext:
    cam = cv2.VideoCapture(0 if opt.demo == 'webcam' else opt.demo)
    detector.pause = False
    
    #先读取一张图片，得到视频长宽
    _, img = cam.read()
    h, w = img.shape[:2]
    #保存视频
    fps = cam.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID') #使用XVID编码器
    # out = cv2.VideoWriter('../output/output.avi',fourcc, 2.0, (1920,1080))#参数分别是：保存文件名、编码器、帧率、视频宽高
    out = cv2.VideoWriter('../output/output.avi',fourcc, fps, (w,h))#参数分别是：保存文件名、编码器、帧率、视频宽高
    
    # '''
    # 会议图片转为视频
    # '''
    # path = '../images/'
    # images = os.listdir('../images/')
    # images.sort()
    # for image in images:
      # img=cv2.imread(path + image)
      # out.write(img)
    # print('已全部读取完成！')
    m=0
    while True:
        start_time = time.time()
        _, img = cam.read()
        ret = detector.run(img, out=out) #对单一图片测试
        # time_str = ''
        # for stat in time_stats:
          # time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
        
        # 提取识别到的人脸，通常情况下用不到
        # faces = ret['results'][1] #[0],[1],[2]三个不同类别,都要提取
        # for i, face in enumerate(faces):
          # if face[4] > 0.2:
            # face = np.array(face, dtype=np.int32)
            # cv2.imwrite('../output/{}_{}.jpg'.format(m,i),img[face[1]:face[3], face[0]:face[2]])
        # m +=1
        
        print('total time:',time.time()-start_time)
            
# #测试用
    # while True:
        # start_time = time.time()
        # _, img = cam.read()
        # ret = detector.run(img, out=out) #对单一图片测试
        # time_str = ''
        # for stat in time_stats:
          # time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
            # # print(time_str)
        # print('total time:',time.time()-start_time)

  else:
    if os.path.isdir(opt.demo):
      image_names = []
      ls = os.listdir(opt.demo)
      out = cv2.VideoWriter('../output/output.avi',1, 10, (1920,1080))#参数分别是：保存文件名、编码器、帧率、视频宽高
      for file_name in sorted(ls):
          ext = file_name[file_name.rfind('.') + 1:].lower()
          if ext in image_ext:
              image_names.append(os.path.join(opt.demo, file_name))
    else:
      image_names = [opt.demo]
      out = cv2.VideoWriter('../output/output.avi',1, 10, (1920,1080))#参数分别是：保存文件名、编码器、帧率、视频宽高
    for (image_name) in image_names:
      start_time = time.time()
      ret = detector.run(image_name, out)
      # time_str = ''
      # for stat in time_stats:
        # time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
      print('total time:',time.time()-start_time)
if __name__ == '__main__':
  opt = opts().init()
  demo(opt)
