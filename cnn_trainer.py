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
from chainer import optimizers, serializers, iterators
from chainer import training
from chainer.training import extensions
from net_cnn import CNN                 # ネットワークモデル

#---Matlab function---
def showImagePLT(im):
    """画像を表示"""
    im_list = np.asarray(im)
    plt.imshow(im_list)
    plt.show()

parser = argparse.ArgumentParser(description='Sample: Convolution Neural Network')
parser.add_argument('--gpu', '-g', dest='gpu', type=int, default=-1, help= '0: use gpu, -1: use cpu')
parser.add_argument('--epoch', '-e', dest='epoch', type=int, default=100, help='number of epochs to learn')
parser.add_argument('--batch', '-b', dest='batch', type=int, default=30, help='number of batch size to learn')
parser.add_argument('--load', '-l', dest='load', type=int, default=0, help='10: load model snapshot number, 0: new model')
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
data_dir  = "./images/train/"
# data_dirのファイル名を読み込み #
label_dir = os.listdir(data_dir)
print(label_dir)
# data_driのファイルの数 (クラス数)#
label_num = len(label_dir)

print('label name : {}'.format(label_dir))
print('num : {}'.format(label_num))

# 全画像の呼び出し処理 #
for label_name in label_dir:
    set_label = []
    # one hotの作成 #
    if label_name == 'Motorbikes':
        set_label = [1,0]
    elif label_name == 'airplanes':
        set_label = [0,1]


    dir_path  = data_dir + label_name
    # trainの 1クラスの全画像ファイル名取得 #
    image_dir = os.listdir(dir_path)
    # sortする
    image_dir = sorted(image_dir, key=str.lower)
    # trainの 1クラスのデータ数 #
    image_num = len(image_dir)
    print('image num : {}'.format(image_num))

    read_num = 0
    # 1クラスの全データ呼び出し #
    for image_num in image_dir:
        image    = cv2.imread(dir_path + '/' + image_num)
        # サイズ変換 #
        re_image = cv2.resize(image, (image_size))
        # --- use debug : show image & resized image --- #
        if read_num == 0:
            showImagePLT(image)
            showImagePLT(re_image)
        t_image    = re_image.transpose(2, 0, 1)
        t_image    = t_image.astype(np.float32)
        norm_image = t_image / 255.0
        # 全画像データをimage_listに格納 #
        image_list.append(norm_image)
        # 全画像データ数のlabelをlabel_listに格納 #
        label_list.append(set_label)

        read_num += 1

        # 保険のためにある #
        if read_num >= 10000:
            break
# ---------------------------------#
#           リストから配列へ           #
# ---------------------------------#
train_data_array  = np.array(image_list).astype(np.float32)
train_label_array = np.array(label_list).astype(np.float32)
print('train data num  : {}'.format(len(train_data_array)))
print('train_label num : {}'.format(len(train_label_array)))

# --- 学習用データとテスト用に分ける --- #
#  train: 70%      test: 30%      #
# ------------------------------- #
X_train, X_test, y_train, y_test = train_test_split(train_data_array, train_label_array, test_size=0.3)

# -------------------------------- #
#           配列からタプルへ           #
#         (特徴量, ラベル)           #
#     (array[...], [0, 1,..])      #
# -------------------------------- #
train_data = tuple_dataset.TupleDataset(X_train, y_train)
test_data  = tuple_dataset.TupleDataset(X_test, y_test)

# --- initialize model ---#
input_channel = 3
output_units = 2

# ------------------------------- モデルをセット ------------------------------- #
model = L.Classifier(CNN(input_channel, output_units), lossfun=F.mean_squared_error)    # lossfunで損失関数を決める
# gpu設定 #
if args.gpu >= 0:
    print('----- use GPU -----')
    chainer.cuda.get_device(args.gpu).use()
    model.to_gpu()
model.compute_accuracy=False    # accuracyを計算しない
optimizer = optimizers.AdaGrad()
optimizer.setup(model)

# ------------------------------- 学習 ------------------------------- #
train_iter = iterators.SerialIterator(train_data, args.batch)
test_iter  = iterators.SerialIterator(test_data, args.batch, repeat=False, shuffle=False)
updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)

# iteration回数またはepochの回数を定義 #
trainer = training.Trainer(updater, (args.epoch, 'epoch'), out='result')
trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
trainer.extend(extensions.dump_graph('main/loss', out_name="cg.dot"))
# snapshotする間隔を決める #
trainer.extend(extensions.snapshot(filename='snapshot_iter_{.updater.epoch}'), trigger=(args.trigger, 'epoch'))
# --- モデルをロード
if args.load >= 1:
    snapshot_num = './result/snapshot_iter_' + str(args.load)
    serializers.load_npz(snapshot_num, trainer)
# modelの保存 #
trainer.extend(extensions.snapshot_object(model.predictor, 'model_snapshot_{.updater.epoch}', savefun=serializers.save_npz), trigger=(args.trigger, 'epoch'))
trainer.extend(extensions.LogReport())      # lossを 1 epochごとに出力
#trainer.extend(extensions.LogReport(), trigger=(args.trigger, 'epoch'))    # lossを triggerごとに出力
trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss']))
trainer.extend(extensions.PlotReport(['main/loss','validation/main/loss'],'epoch',file_name='loss.png', marker=""))
trainer.extend(extensions.ProgressBar())
trainer.run()
# --- save model & optimizer --- #
print('save the model')
serializers.save_npz('./result/end.model', model.predictor)
print('save the optimezer')
serializers.save_npz('./result/end.state', optimizer)
