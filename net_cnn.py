# -*- coding: utf-8 -*-
"""
Chainer simple code
name    : Convolutional neural network
layer   : conv -> maxpool -> conv -> maxpool -> conv -> fullconnect -> dropout -> fullconnect

"""
#---Chainer---
import chainer
import chainer.links as L
import chainer.functions as F

"""
model class
__init__(in_units, hid_units, out_units):
    in_channel: input image channel
    out_units : output units num
forward(x):
    x : input data [chainer.Variable]
"""

class CNN(chainer.Chain):
    def __init__(self, in_channel, out_units):
        # init model by chainer.Chain
        super(CNN,self).__init__(
            conv1=L.Convolution2D(3,64,ksize=(16,16),stride=1),
            conv2=L.Convolution2D(64,128,ksize=(8,8),stride=1),
            conv3=L.Convolution2D(128,128,ksize=(4,4),stride=1),
            linear1=L.Linear(512,256),
            linear2=L.Linear(256,2)
        )
    #forward function
    def forward(self,x):
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(h,ksize=(8,8),stride=2)
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(h,ksize=(8,8),stride=2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.linear1(h))
        #h = F.dropout(h)
        result = F.softmax(self.linear2(h))
        return result

    def __call__(self, x):
        h = self.forward(x)
        return h
