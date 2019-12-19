from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch
from PIL import Image
try:
  from external.nms import soft_nms
except:
  print('NMS not imported! If you need it,'
        ' do \n cd $CenterNet_ROOT/src/lib/external \n make')
from models.decode import ctdet_decode
from models.utils import flip_tensor
from utils.image import get_affine_transform
from utils.post_process import ctdet_post_process
from utils.debugger import Debugger

from .base_detector import BaseDetector

class CtdetDetector(BaseDetector):
  def __init__(self, opt):
    super(CtdetDetector, self).__init__(opt)
  
  def process(self, images, return_time=False):
    with torch.no_grad():
      output = self.model(images)[-1] #使用加载的模型预测得到热力图
      hm = output['hm'].sigmoid_()
      wh = output['wh']
      reg = output['reg'] if self.opt.reg_offset else None #回归偏移
      if self.opt.flip_test: #翻转数据扩充
        hm = (hm[0:1] + flip_tensor(hm[1:2])) / 2
        wh = (wh[0:1] + flip_tensor(wh[1:2])) / 2
        reg = reg[0:1] if reg is not None else None
      torch.cuda.synchronize()
      forward_time = time.time()
      dets = ctdet_decode(hm, wh, reg=reg, K=self.opt.K) #将热力图解码成包围盒，dets是1*100*6的张量
      
    if return_time:
      return output, dets, forward_time
    else:
      return output, dets

  def gray_preprocess(self, dets):
    pass

  def post_process(self, dets, meta, scale=1):
    dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])
    dets = ctdet_post_process(
        dets.copy(), [meta['c']], [meta['s']],
        meta['out_height'], meta['out_width'], self.opt.num_classes)
    for j in range(1, self.num_classes + 1):
      dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
      dets[0][j][:, :4] /= scale
    return dets[0]

  def merge_outputs(self, detections):
    results = {}
    for j in range(1, self.num_classes + 1):
      results[j] = np.concatenate(
        [detection[j] for detection in detections], axis=0).astype(np.float32)
      if len(self.scales) > 1 or self.opt.nms:
         soft_nms(results[j], Nt=0.5, method=2)
    scores = np.hstack(
      [results[j][:, 4] for j in range(1, self.num_classes + 1)])
    if len(scores) > self.max_per_image:
      kth = len(scores) - self.max_per_image
      thresh = np.partition(scores, kth)[kth]
      for j in range(1, self.num_classes + 1):
        keep_inds = (results[j][:, 4] >= thresh)
        results[j] = results[j][keep_inds]
    return results

  def debug(self, debugger, images, dets, output, scale=1):
    detection = dets.detach().cpu().numpy().copy()
    detection[:, :, :4] *= self.opt.down_ratio
    for i in range(1):
      img = images[i].detach().cpu().numpy().transpose(1, 2, 0)
      img = ((img * self.std + self.mean) * 255).astype(np.uint8)
      pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())  #预测的 color_map
      debugger.add_blend_img(img, pred, 'pred_hm_{:.1f}'.format(scale))  # 叠加color_map在图上
      debugger.add_img(img, img_id='out_pred_{:.1f}'.format(scale))
      for k in range(len(dets[i])):
        if detection[i, k, 4] > self.opt.center_thresh:
          debugger.add_coco_bbox(detection[i, k, :4], detection[i, k, -1],
                                 detection[i, k, 4], 
                                 img_id='out_pred_{:.1f}'.format(scale))

                                 
  def show_results(self, debugger, image, results, out):
    debugger.add_img(image, img_id='ctdet')
    faces = torch.empty(0,3,224,224).cuda() #创建空张量
    boxes=[]
    up = 0
    down = 0
    rotate = 0
    happy=0
    normal=0
    none=0
    for j in range(1, self.num_classes + 1):
      for bbox in results[j]:
        if bbox[4] > 0.3:  #设置的最低阈值，高于它则画出来
          boxes.append(bbox)
          face_box = np.array(bbox[0:4], dtype=np.int32)
          #有时会出现回归的人脸两个横坐标相同，导致face为空报错，这里加一个错误处理方法
          try:
            face = image[(abs(face_box[1])):abs(face_box[3]), (abs(face_box[0])):abs(face_box[2])]
            face = Image.fromarray(face)
          except:
            face = np.array(image)[abs(face_box[1]):abs(face_box[3])+1, abs(face_box[0]):abs(face_box[2])+1]
            face = Image.fromarray(face)
          face = self.image_trans(face)
          faces = torch.cat((faces,face),dim=0) #所有人脸的张量
      if j==1:
        down = len(boxes)
        # boxes = []
      elif j==2:
        up = len(boxes) - down
        # boxes = []
      elif j==3:
        rotate = len(boxes)-down - up
        # boxes = []
    num = len(faces) #检测到的人数
    if num >= 10:
      emotions = self.model_emotion(faces)
      _, emotions = torch.max(emotions, 1)
      emotions = emotions.cpu().numpy()
      emotions = emotions.tolist()
      # print(emotions)
      happy = emotions.count(0)
      normal = emotions.count(2)
      none = emotions.count(1)
      
      concentrate_rate = round((100*up/float(num)),2)
      activity_rate = round((float(happy*100+normal*60+none*10)/num),2)
      print('----------------')
      print('总人数：',len(faces))
      print('抬头:{},低头:{},扭头:{}'.format(up,down,rotate))
      print('微笑:{},正常:{},无表情:{}'.format(happy,normal,none))
      print('参会人员专注度:{:.2f} 会场活跃度:{:.2f}'.format(concentrate_rate, activity_rate))
      debugger.save_all_imgs(genID=True) #保存所有图片
    elif num !=0 and num < 10: #当人数小于10时，控制专注度与活跃度的大小，将num设置为固定10
    # 增加未检测到人脸时的操作
      # concentrate_rate = 0
      # activity_rate = 0
      # print('----------------')
      # print('未检测到人脸！')
      emotions = self.model_emotion(faces)
      _, emotions = torch.max(emotions, 1)
      emotions = emotions.cpu().numpy()
      emotions = emotions.tolist()
      # print(emotions)
      happy = emotions.count(0)
      normal = emotions.count(2)
      none = emotions.count(1)
      
      concentrate_rate = round((100*up/float(10)),2)
      activity_rate = round((float(happy*100+normal*60+none*10)/10),2)
      print('----------------')
      print('总人数：',num)
      print('抬头:{},低头:{},扭头:{}'.format(up,down,rotate))
      print('微笑:{},正常:{},无表情:{}'.format(happy,normal,none))
      print('参会人员专注度:{:.2f} 会场活跃度:{:.2f}'.format(concentrate_rate, activity_rate))
      # # 下面是一些保存图片或视频的函数，勿删
      debugger.save_all_imgs(genID=True) #保存所有图片
      # # debugger.save_video(out) # 保存视频
      # # debugger.show_imgs() #展示所有图片
      # return concentrate_rate, activity_rate
    else :
      concentrate_rate = 0
      activity_rate = 0
      print('----------------')
      print('未检测到人脸！')
    return concentrate_rate, activity_rate
    
  # def show_results(self, debugger, image, results, out):
    # debugger.add_img(image, img_id='ctdet')
    # faces = []
    # up = 0
    # down = 0
    # rotate = 0
    # happy=0
    # normal=0
    # none=0
    # for j in range(1, self.num_classes + 1):
      # for bbox in results[j]:
        # if bbox[4] > 0.3:  #设置的最低阈值，高于它则画出来
          # if j==1:
            # down +=1
          # elif j==2:
            # up +=1
          # elif j==3:
            # rotate +=1
          # face = np.array(bbox[0:4], dtype=np.int32)
          # face = image[abs(face[1]):abs(face[3]), abs(face[0]):abs(face[2])]
          # face = Image.fromarray(face)
          # faces.append(face)
          # emotion = self.emotion_test(face)
          # if emotion == 'happy':
            # happy +=1
          # elif emotion == 'normal':
            # normal  +=1
          # elif emotion == 'none':
            # none +=1
          # debugger.add_coco_bbox(emotion, bbox[:4], j - 1, bbox[4], img_id='ctdet')
          
    # # 增加未检测到人脸时的操作
    # try:
      # concentrate_rate = round((100*up/float(len(faces))),2)
      # activity_rate = round((float(happy*100+normal*60+none*10)/len(faces)),2)
      # print('----------------')
      # print('总人数：',len(faces))
      # print('抬头:{},低头:{},扭头:{}'.format(up,down,rotate))
      # print('微笑:{},正常:{},无表情:{}'.format(happy,normal,none))
      # print('参会人员专注度:{:.2f} 会场活跃度:{:.2f}'.format(concentrate_rate, activity_rate))
    # except:
      # concentrate_rate = 0
      # activity_rate = 0
      # print('----------------')
      # print('未检测到人脸！')

