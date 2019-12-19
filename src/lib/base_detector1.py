from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch


from torchvision import datasets, transforms
from models.model import create_model, load_model
from utils.image import get_affine_transform
from utils.debugger import Debugger

trans = transforms.Compose(
                    [  
                      transforms.Resize(size=(224,224)),
                      transforms.ToTensor(),
                      transforms.Normalize(mean=(2.133921, 2.271873, 2.351718),std=(5.289286, 5.630496, 5.829290))
                    ])

class_names = ['happy', 'none', 'normal']

class BaseDetector(object):
  def __init__(self, opt):
    if opt.gpus[0] >= 0:
      opt.device = torch.device('cuda')
    else:
      opt.device = torch.device('cpu')
    
    print('Creating model...')
    self.model = create_model(opt.arch, opt.heads, opt.head_conv)
    self.model_emotion = torch.load('../models/emotion_model.pth', map_location={'cuda:1':'cuda:0'})
    self.model_emotion = self.model_emotion.to(opt.device)
    self.model_emotion.eval()
    self.model = load_model(self.model, opt.load_model)
    self.model = self.model.to(opt.device)
    self.model.eval()

    self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
    self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)
    self.max_per_image = 100
    self.num_classes = opt.num_classes
    self.scales = opt.test_scales
    self.opt = opt
    self.pause = True
    
    #test
    # pic = torch.rand(2,3,224,224)
    # print(pic.shape)
    # predss = self.model_emotion(pic.cuda())
    # _, predss = torch.max(predss, 1)
    # print(class_names[predss[0]])
    
  def image_trans(self, img):
    img = trans(img)
    img =img.to(self.opt.device)
    img = img.unsqueeze(0)

    return img
    
  def pre_process(self, image, scale, meta=None):
    height, width = image.shape[0:2]
    new_height = int(height * scale)
    new_width  = int(width * scale)
    if self.opt.fix_res: #如果为真，则将图像尺寸归一化为512*512
      inp_height, inp_width = self.opt.input_h, self.opt.input_w
      c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
      s = max(height, width) * 1.0
    else:
      inp_height = (new_height | self.opt.pad) + 1
      inp_width = (new_width | self.opt.pad) + 1
      c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
      s = np.array([inp_width, inp_height], dtype=np.float32)

    trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
    resized_image = cv2.resize(image, (new_width, new_height))
    inp_image = cv2.warpAffine(
      resized_image, trans_input, (inp_width, inp_height),
      flags=cv2.INTER_LINEAR)
    inp_image = ((inp_image / 255. - self.mean) / self.std).astype(np.float32)

    images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
    if self.opt.flip_test:
      images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
    images = torch.from_numpy(images)
    meta = {'c': c, 's': s,    #meta中包含c：resized_img的中心位置，s:最长宽度 
            'out_height': inp_height // self.opt.down_ratio, 
            'out_width': inp_width // self.opt.down_ratio}
    return images, meta

  def process(self, images, return_time=False):
    raise NotImplementedError

  def post_process(self, dets, meta, scale=1):
    raise NotImplementedError

  def merge_outputs(self, detections):
    raise NotImplementedError

  def debug(self, debugger, images, dets, output, scale=1):
    raise NotImplementedError

  def show_results(self, debugger, image, results):
   raise NotImplementedError

  def run(self, image_or_path_or_tensor, out, meta=None):
    load_time, pre_time, net_time, dec_time, post_time = 0, 0, 0, 0, 0
    merge_time, tot_time = 0, 0
    debugger = Debugger(dataset=self.opt.dataset, ipynb=(self.opt.debug==3),
                        theme=self.opt.debugger_theme)
    start_time = time.time()
    pre_processed = False  #图像载入：判断命令行给的是图片、路径还是张量
    if isinstance(image_or_path_or_tensor, np.ndarray):
      image = image_or_path_or_tensor
    elif type(image_or_path_or_tensor) == type (''): 
      image = cv2.imread(image_or_path_or_tensor)
    else:
      image = image_or_path_or_tensor['image'][0].numpy()
      pre_processed_images = image_or_path_or_tensor
      pre_processed = True
    
    loaded_time = time.time()
    load_time += (loaded_time - start_time) # 载入图片的时间
    
    detections = []
    for scale in self.scales: #scales 应该是将图片扩大一定倍数后检测
      # print(self.scales)
      scale_start_time = time.time()
      if not pre_processed: #如果给的是图片或路径那就预处理一下
        images, meta = self.pre_process(image, scale, meta)
      else:
        # import pdb; pdb.set_trace()
        images = pre_processed_images['images'][scale][0]
        meta = pre_processed_images['meta'][scale]
        meta = {k: v.numpy()[0] for k, v in meta.items()}
      images = images.to(self.opt.device) #image放入GPU
      torch.cuda.synchronize() #让所有核同步，测得真实时间
      pre_process_time = time.time()
      pre_time += pre_process_time - scale_start_time #预处理时间
      
      output, dets, forward_time = self.process(images, return_time=True) #送入预测得到预测数据与包围盒及当前时间,dets是一个len=类别数的张量，每个元素是一个N * 5的ndarray
     
      torch.cuda.synchronize()
      net_time += forward_time - pre_process_time #预测热力图的时间
      decode_time = time.time()
      dec_time += decode_time - forward_time #热力图解码的时间
      
      if self.opt.debug >= 2: #debug大于2，则输出三种图：预测图，缩resizw预测，热力图
        self.debug(debugger, images, dets, output, scale)
      
      dets = self.post_process(dets, meta, scale) # 根据热力图回归坐标系数
      # print(dets)
      torch.cuda.synchronize()
      post_process_time = time.time()
      post_time += post_process_time - decode_time #坐标系数回归过程的时间

      detections.append(dets)
    
    results = self.merge_outputs(detections) # 回归到真实坐标
    torch.cuda.synchronize()
    end_time = time.time()
    merge_time += end_time - post_process_time #回归到真实坐标时间
    tot_time += end_time - start_time #总时间
    
    # print(results[2])
    # faces = results[1]
    # for i, face in enumerate(faces):
      # if face[4] > 0.2:
        # face = np.array(face, dtype=np.int32)
        # face = image[face[1]:face[3], face[0]:face[2]]
        # emotion = self.emotion_test(face)
    
    if self.opt.debug >= 1:
      self.show_results(debugger, image, results, out=out)
    
    return {'results': results, 'tot': tot_time, 'load': load_time,
            'pre': pre_time, 'net': net_time, 'dec': dec_time,
            'post': post_time, 'merge': merge_time}