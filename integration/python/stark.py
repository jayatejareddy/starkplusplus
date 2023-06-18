# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import tempfile
from argparse import ArgumentParser

import cv2
import mmcv
import vot
import cv2
import numpy as np
from mmtrack.apis import inference_sot, init_model
import time
import collections

# parser = ArgumentParser()
# parser.add_argument('config', default = '/mnt/DATA/jas123/Downloads/mmtracking/configs/sot/stark/stark_st1_r50_500e_got10k.py',help='Config file')

# parser.add_argument('--checkpoint', default = '/mnt/DATA/jas123/Downloads/mmtracking/weights.pth', help='Checkpoint file')
# parser.add_argument(
#         '--device', default='cuda:0', help='Device used for inference')
# parser.add_argument(
#         '--color', default=(0, 255, 0), help='Color of tracked bbox lines.')
# parser.add_argument(
#         '--thickness', default=3, type=int, help='Thickness of bbox lines.')
# args = parser.parse_args()


handle = vot.VOT("rectangle", multiobject=True)
objects = handle.objects()
model = [init_model('/mnt/DATA/jas123/Downloads/mmtracking/configs/sot/stark/stark_st2_r50_50e_got10k.py', '/mnt/DATA/jas123/Downloads/mmtracking/stark_st2_r50_50e_lasot_20220416_170201-b1484149.pth', device='cuda:0') for i in range(len(objects))]


imagefile = handle.frame()

image = cv2.imread(imagefile, cv2.IMREAD_GRAYSCALE)
init_bbox = [[objects[i].x,objects[i].y,objects[i].width +objects[i].x,objects[i].height+objects[i].y] for i in range(len(objects))]
img = mmcv.imread(imagefile)
result = [inference_sot(model[i], img, init_bbox[i], frame_id=0) for i in range(len(objects))]
i =1
while True:
    imagefile = handle.frame()
    
    if not imagefile:
        # Terminate if no new frame was received.
        break
    
    image = cv2.imread(imagefile, cv2.IMREAD_GRAYSCALE)
    
    img = mmcv.imread(imagefile)

    result = [inference_sot(model[j], img, init_bbox[j], frame_id=i) for j in range(len(objects))]
    results = [np.array(result[i]["track_bboxes"]).tolist() for i in range(len(objects))]
    
    obj =[]
    for i in range(len(objects)):
        if results[i][4]>0.5:
            obj.append(vot.Rectangle(results[i][0],results[i][1],results[i][2] - results[i][0],results[i][3] -results[i][1]))
        else:
            obj.append(vot.Empty())

    # obj = [vot.Rectangle(results[i][0],results[i][1],results[i][2] - results[i][0],results[i][3] -results[i][1]) for i in range(len(objects)) if results[i][4]<0.5 else vot.Empty()]
    i = i+1
    # obj1 = [vot.Empty() for i in range(len(objects))]
    handle.report(obj)
    




