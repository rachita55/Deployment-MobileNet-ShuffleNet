#!/usr/bin/env python
# coding: utf-8

# In[34]:


import torch


# In[35]:


import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import argparse


# In[94]:


from flask import Flask, request, jsonify


# In[37]:


from PIL import Image
import torchvision.transforms.functional as TF


# In[38]:


import numpy as np
import requests
from io import BytesIO


# In[ ]:





# In[39]:


class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class MobileNet(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512,512, (1024,2), 1024]  #change the depth of convolution.(Removed 3X512)

    def __init__(self, num_classes=10):
        super(MobileNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.linear = nn.Linear(1024, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

net = MobileNet()


# In[96]:


#PORT = 8090

app = Flask(__name__)
@app.route("/")
def hello():
    return "classification\n"
@app.route('/get_prediction', methods=['GET'])
def get_prediction():
    img_path_ = request.args['url']
    img_path = requests.get(img_path_)
    img = Image.open(BytesIO(img_path.content)).convert('RGB')
    image = img.resize((32, 32))
    width,height = image.size
    #print(width,height)
    img = TF.to_tensor(image)
    img.unsqueeze_(0)
    with torch.no_grad():
        model = torch.load('mobilenetbestss.pth',map_location='cpu')
        outputs = model(img)
        # loss = criterion(outputs, targets)
        # test_loss += loss.item()
        # _, predicted = outputs.max(1)
        # total += targets.size(0)
        # correct += predicted.eq(targets).sum().item()
        softmax = torch.exp(outputs).cpu()
        prob = list(softmax.numpy())
        predictions = np.argmax(prob, axis=1)
        #print(predictions)
        op_val = None
        dic = {0 :'Airplane', 1:'automobile',2:'bird',3:'cat',4:'deer',5:'dog',6:'frog',7:'horse',8:'ship',9:'truck'}
        for i in dic :
            if i == predictions:
                op_val = dic[i]
        return jsonify(op_val)
        #print(predictions_1)
        
if __name__ == '__main__':
    app.run(debug=True)




















# In[ ]:





# In[ ]:





# In[ ]:




