#-*- coding: utf-8 -*-
# --- System --- #
import os
import time
import sys
from time import time
import argparse

#---Image---
import cv2
import matplotlib.pyplot as plt

# --- tools --- #
import numpy as np

# --- sklearn --- #
from sklearn.model_selection import train_test_split

# --- Chainer --- #
import chainer
from chainer.datasets import tuple_dataset
import chainer.functions as F
import chainer.links as L
from chainer import cuda    # GPU
from chainer import optimizers, serializers, iterators, Variable
from chainer import training
from chainer.training import extensions
from net_cnn import CNN                 # ネットワークモデル

#---Matlab function---
def showImagePLT(im):
    im_list = np.asarray(im)
    plt.imshow(im_list)
    plt.show()

parser = argparse.ArgumentParser(description='Convolution Neural Network')
parser.add_argument('--gpu', '-g', dest='gpu', type=int, default=-1, help= '0: use gpu, -1: use cpu')
parser.add_argument('--epoch', '-e', dest='epoch', type=int, default=10000, help='number of epochs to learn')
parser.add_argument('--batch', '-b', dest='batch', type=int, default=30, help='number of batch size to learn')
parser.add_argument('--load', '-l', dest='load', type=int, default=0, help='1: load model, 0: new model')
parser.add_argument('--num', '-n', dest='num', type=int, default=1000, help='The number of data')
parser.add_argument('--trigger', '-t', dest='trigger', type=int, default=10, help='set trigger')
args=parser.parse_args()
print('gpu      : ', args.gpu)
print('epoch    : ', args.epoch)
print('batch    : ', args.batch)
print('load     : ', args.load)
print('data num : ', args.num)
print('trigger  : ', args.trigger)

# --- initialize model --- #
input_channel = 3
output_units = 2

# --- prepare dataset --- #
image_size = (64, 64)
image_list = []
label_list = []

# --- read images directory --- #
data_dir  = "./images/test/"
label_dir = os.listdir(data_dir)
print(label_dir)
label_num = len(label_dir)

print('label name : {}'.format(label_dir))
print('num : {}'.format(label_num))

for label_name in label_dir:
    set_label = []
    if label_name == 'Motorbikes':
        set_label = [1,0]
    elif label_name == 'airplanes':
        set_label = [0,1]

    dir_path  = data_dir + label_name
    image_dir = os.listdir(dir_path)
    # sortする
    image_dir = sorted(image_dir, key=str.lower)
    image_num = len(image_dir)
    print('image num : {}'.format(image_num))

    read_num = 0
    for image_num in image_dir:
        image    = cv2.imread(dir_path + '/' + image_num)
        re_image = cv2.resize(image, (image_size))
        """use debug : show image & resized image"""
        if read_num == 0:
            showImagePLT(image)
            showImagePLT(re_image)
        t_image    = re_image.transpose(2, 0, 1)
        t_image    = t_image.astype(np.float32)
        norm_image = t_image / 255.0
        image_list.append(norm_image)
        label_list.append(set_label)

        read_num += 1

        if read_num >= 10000:
            break

train_data_array  = np.array(image_list).astype(np.float32)
train_label_array = np.array(label_list).astype(np.float32)

# --- 学習用データとテスト用に分ける --- #
#X_train, X_test, y_train, y_test = train_test_split(train_data_array, train_label_array, test_size=0.3)
X_test = train_data_array
y_test = train_label_array

# -------------------------------- #
# 配列からタプルへ
# (特徴量, ラベル)
# (array[...], [0, 1,..])
# -------------------------------- #
test_data  = tuple_dataset.TupleDataset(X_test, y_test)

# --- initialize model ---#
input_channel = 3
output_units = 2

#------------------ ここまでcnn_trainer.pyと一緒 ------------------ #
#------------------ 呼び出しファイル名が違うだけ    ------------------- #
with chainer.using_config('train', False):
    # --- モデルをセット --- #
    model = CNN(input_channel, output_units)
    # --- モデルをロード --- #
    #serializers.load_npz('./result/mlp_end.model', model)
    serializers.load_npz('./result/end.model', model)

    # --- テスト --- #
    # X_testを Variable型に変換
    xt = Variable(X_test)
    # forwardする
    yy = model.forward(xt)
    ans = yy.data
    nrow, col = ans.shape
    print('Test num: {}'.format(nrow))
    ok = 1
    no = []
    for i in range(nrow):
        # 最大値を持つ表のインデックスを取得 #
        cls = np.argmax(ans[i,:])
        print('{}, index number : {}'.format(ans[i,:], cls))
        if cls == np.argmax(y_test[i,:]):
            ok += 1
        else:
            no.append(np.argmax(y_test[i,:]))
    print('{}, / {}, = {}'.format(ok, nrow, (ok *1.0) / nrow))
    print('mistake num : {}'.format(len(no)))
    print(no)
