from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch
import os
import matplotlib.pyplot as plt
# import tensorflow as tf
# from keras.models import load_model as load_keras_model
# import keras.backend.tensorflow_backend as KTF

from torchvision import datasets, models, transforms
from models.model import create_model, load_model
from utils.image import get_affine_transform
from utils.debugger import Debugger

data_dir = "data_transfer"

mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

#data transformation
data_transforms = {
   'train': transforms.Compose([
       transforms.Resize(size=(224,224)),
       transforms.ToTensor(),
       transforms.Normalize(mean, std)
   ]),
   'validation': transforms.Compose([
       transforms.Resize(size=(224,224)),
       transforms.ToTensor(),
       transforms.Normalize(mean, std)
   ]),
}

image_datasets = {
   x: datasets.ImageFolder(
       os.path.join(data_dir, x),
       transform=data_transforms[x]
   )
   for x in ['train', 'validation']
}

dataloaders = {
   x: torch.utils.data.DataLoader(
       image_datasets[x], batch_size=32,
       shuffle=True, num_workers=0
   )
   for x in ['train', 'validation']
}

class_names = image_datasets['train'].classes

def visualize_model(model, num_images=6):
    was_training = model.training # 检验是否是训练模式
    model.eval() # 模式设置为测试模式
    images_so_far = 0
    # fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['validation']):
            inputs = inputs.to(torch.device("cuda:0"))
            labels = labels.to(torch.device("cuda:0"))
            # print(inputs.shape)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            # print(preds)
            for j in range(inputs.size()[0]):
                images_so_far += 1 # 加上这个时间会减少
                # ax = plt.subplot(num_images//2, 2, images_so_far)
                # ax.axis('off')
                # ax.set_title('predicted: {} truth: {}'.format(class_names[preds[j]], class_names[labels[j]]))
                # img = inputs.cpu().data[j].numpy().transpose((1, 2, 0))
                # img = std * img + mean
                # ax.imshow(img)
                # plt.savefig('test.png')
                # if images_so_far == num_images:
                    # model.train(mode=was_training)
                    # return
                
                if images_so_far == num_images:
                    model.train(mode=was_training)
                    print(preds)
                    return
            
        # 在测试之后将模型恢复之前的形式
        model.train(mode=was_training)


def test(img):
  pass
        
class BaseDetector(object):
  def __init__(self, opt):
    if opt.gpus[0] >= 0:
      opt.device = torch.device('cuda')
    else:
      opt.device = torch.device('cpu')
    
    print('Creating model...')
    self.model = create_model(opt.arch, opt.heads, opt.head_conv)
    self.model = load_model(self.model, opt.load_model)
    self.model1 = torch.load('../models/model.pth', map_location={'cuda:1':'cuda:0'})
    self.model1 = self.model1.to(opt.device)
    self.model1.eval()
    self.model = self.model.to(opt.device)
    self.model.eval()

    self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
    self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)
    self.max_per_image = 100
    self.num_classes = opt.num_classes
    self.scales = opt.test_scales
    self.opt = opt
    self.pause = True
    

    
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

  def gray_preprocess(self, dets):
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
      
      output, dets, forward_time = self.process(images, return_time=True) #送入预测得到预测数据与包围盒及当前时间，dets是一个len=80的张量，每个元素是一个N * 5的ndarray

      torch.cuda.synchronize()
      net_time += forward_time - pre_process_time #预测热力图的时间
      decode_time = time.time()
      dec_time += decode_time - forward_time #热力图解码的时间
      
      if self.opt.debug >= 2: #debug大于2，则输出三种图：预测图，resize后预测图，热力图
        self.debug(debugger, images, dets, output, scale)
      
      dets = self.post_process(dets, meta, scale) #
      torch.cuda.synchronize()
      post_process_time = time.time()
      post_time += post_process_time - decode_time #坐标系数回归过程的时间

      detections.append(dets)
      # print(detections)
    results = self.merge_outputs(detections) # 回归到真实坐标
    torch.cuda.synchronize()
    end_time = time.time()
    merge_time += end_time - post_process_time #回归到真实坐标时间
    tot_time += end_time - start_time #总时间

    #表情识别接口
    # emotion_labels = {'0':'angry', '1':'disgust', '2':'fear', '3':'happy', '4':'sad', '5':'surprise', '6':'netural'}
    # emotion_model_path = '../models/emotion_models/simple_CNN.985-0.66.hdf5'
    # emotion_classifier = load_keras_model(emotion_model_path) 
    
    # img = cv2.imread('depressed_412.jpg')
    # img = cv2.resize(img,(224,224))
    # img = transforms.ToTensor()(img)
    # with torch.no_grad():
      # pt = self.model1(img)
    visualize_model(self.model1, num_images=2)

    # print(pt)
    # for detection in detections:
      # faces = self.gray_preprocess(results)
      # print(faces)
      # for face in faces:
        # emotion_predict = self.model1(face)
        # print(emotion_predict)
        # emotion_text = emotion_labels[emotion_predict]
        

    if self.opt.debug >= 1:
      self.show_results(debugger, image, results, out=out)
    
    return {'results': results, 'tot': tot_time, 'load': load_time,
            'pre': pre_time, 'net': net_time, 'dec': dec_time,
            'post': post_time, 'merge': merge_time}