#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import time
# import matplotlib.pyplot as plt
# import multiprocessing as mp
# import threading

# proc = 8 # 8並列

n_neuron = 800
noise_rate = 0.00 # ノイズ割合
lr = 0.003 # ReLUを用いる場合は学習率を小さくしないとWが発散してしまう
mr = 0.5 # 慣性項係数

# データの読み込み
# dir = '../dataset/txt/' # datasetのあるディレクトリ
dir = '/home/itohdak/dataset/mnist/python-mnist/data/' # datasetのあるディレクトリ

start = time.time()

# 訓練データ
x_train = np.loadtxt(dir + 'train-images.txt').reshape((-1, 784)).astype(np.float32) / 255.
y_train = np.loadtxt(dir + 'train-labels.txt').astype(np.int32)
y_train = np.eye(10)[y_train].astype(np.int32) # 識別は0-9の10種類
n_train = len(x_train)

# ノイズ
cols = np.size(x_train, 1)
for i in xrange(0, np.size(x_train)):
    if np.random.random_sample() < noise_rate:
        x_train[i/cols][i%cols] = np.random.random_sample()

# テストデータ
x_test = np.loadtxt(dir + 'test-images.txt').reshape((-1, 784)).astype(np.float32) / 255.
y_test = np.loadtxt(dir + 'test-labels.txt').astype(np.int32)
y_test = np.eye(10)[y_test].astype(np.int32)
n_test = len(x_test)

print 'DATA LOADING COMPLETED.'
print 'NUMBER OF DATA = %d' %(n_train)
print 'NOISE RATE = %.2f' %(noise_rate)
print 'ELAPSED TIME = {} [sec]'.format(time.time() - start)
print "train shape: {}".format(x_train.shape)
print "label shape: {}".format(y_train.shape)

# 活性化関数
# ランプ関数（中間層）
class ReLU:
    def __call__(self, x):
        return x * (x > 0)
    
    def diff(self, x): # 微分
        return 1. * (x > 0)

# シグモイド関数（中間層）
class Sigmoid:
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def diff(self, x): # 微分
        return self(x) * (1 - self(x))

# ソフトマックス関数（出力層）
class Softmax:
    def __call__(self, x):
        x_temp = x - np.max(x, axis=1).reshape(len(x), -1) # オーバーフロー防止
        return np.exp(x_temp) / np.sum(np.exp(x_temp), axis=1, keepdims=True)

    def diff(self, x): # 微分
        return self(x) * (1 - self(x))


# パーセプトロン
class Perceptron:
    # in_dim : 入力データの次元
    # out_dim : 出力データの次元
    # activation : 活性化関数
    def __init__(self, in_dim, out_dim, p_dropout, activation):
        self.W = np.random.uniform(low=-0.08, high=0.08, size=(in_dim, out_dim)).astype(np.float32)
        self.b = np.zeros(out_dim).astype(np.float32)
        self.delta = None
        self.activation = activation()
        self.p = p_dropout # ドロップアウトの選出確率
        self.mask = np.where(np.random.sample(size=out_dim)<self.p, 1., 0.) # ドロップアウト用のマスク
        self.pre_dW = None
        self.pre_db = None

    def __call__(self, x):
        self.u = np.dot(x, self.W) + self.b # 評価値
        self.z = self.activation(self.u) # 出力値
        return self.z


# 多層パーセプトロン(Multi-Layer Perceptron)
class MLP():
    def __init__(self, layers):
        self.layers = layers
        
    # x : 入力データ
    # t : 教師データ
    # learning_rate : 学習率
    def train(self, x, t, learning_rate, momentum_rate):
        # 順伝播
        self.y = x
        for layer in self.layers:
            layer.mask = np.where(np.random.sample(len(layer.mask))<layer.p, 1., 0.)
            self.y = layer(self.y)
            self.y = self.y * layer.mask # ドロップアウト

        self.loss = np.sum(-np.log(self.y[np.where(t == 1)])) / len(x) # 交差エントロピー誤差

        # 誤差逆伝播
        delta = self.y - t
        self.layers[-1].delta = delta
        W = self.layers[-1].W

        for layer in self.layers[-2::-1]:
            delta = np.dot(delta, W.T) * layer.activation.diff(layer.u)
            delta = delta * layer.mask
            layer.delta = delta
            W = layer.W
        
        # 重みの更新
        z = x
        for layer in self.layers:
            dW = np.dot(z.T, layer.delta)
            db = np.dot(np.ones(len(z)), layer.delta)
            layer.W -= learning_rate * dW
            layer.b -= learning_rate * db
            # 慣性項
            if layer.pre_dW is not None and layer.pre_db is not None:
                layer.W += momentum_rate * layer.pre_dW
                layer.b += momentum_rate * layer.pre_db
            layer.pre_dW = - learning_rate * dW
            layer.pre_db = - learning_rate * db
            z = layer.z
            
        return self.loss
    
    
    def test(self, x, t):
        # 順伝播
        self.y = x
        for layer in self.layers:
            self.y = layer(self.y)
            self.y = self.y * layer.p # 学習時の1/p倍のニューロン数を使うため
        self.loss = np.sum(-np.log(self.y[np.where(t == 1)])) / len(x)
        return self.loss

# def subcalc(p):
#     sub_sum_loss = 0
#     sub_pred_y = []

#     ini = n_train * p / proc
#     fin = n_train * (p+1) / proc

#     for i in xrange(ini, fin, batchsize):
#         x = x_train[perm[i:i+batchsize]]
#         t = y_train[perm[i:i+batchsize]]
        
#         sub_sum_loss += model.train(x, t, lr) * len(x)
#         sub_pred_y.extend(np.argmax(model.y, axis=1))

#     return sub_sum_loss, sub_pred_y

# class MyThread(threading.Thread):
#     def __init__(self, num):
#         threading.Thread.__init__(self)
#         self.num = num

#     def run(self):
#         global sum_loss, pred_y

#         sub_sum_loss, sub_pred_y = subcalc(self.num)
#         sum_loss += sub_sum_loss
#         pred_y.extend(sub_pred_y)

# model = MLP([Perceptron(784, 1000, Sigmoid),
#              Perceptron(1000, 1000, Sigmoid),
#              Perceptron(1000, 10, Softmax)])
# model = MLP([Perceptron(784, 1000, Sigmoid),
#              Perceptron(1000, 1000, Sigmoid),
#              Perceptron(1000, 1000, Sigmoid),
#              Perceptron(1000, 10, Softmax)])
model = MLP([Perceptron(784, n_neuron, 0.5, ReLU),
             Perceptron(n_neuron, n_neuron, 0.5, ReLU),
             Perceptron(n_neuron, 10, 1.0, Softmax)])


n_epoch = 20 # 試行回数
batchsize = 1
# loss_list = []

print 'neuron number = %d' %n_neuron
print 'learning rate = %.3f' %lr
print 'momentum rate = %.1f' %mr

print '         | Train             | Test              |'
print 'epoch    | loss     accuracy | loss     accuracy |'

for epoch in range(n_epoch):
    if epoch == 6:
        lr /= 10.
    print 'epoch %2d |' %(epoch+1),
    
    # Training
    sum_loss = 0
    pred_y = []
    perm = np.random.permutation(n_train) # 訓練データをランダムに並び替え
    
    for i in xrange(0, n_train, batchsize):
        x = x_train[perm[i:i+batchsize]]
        t = y_train[perm[i:i+batchsize]]
        
        sum_loss += model.train(x, t, lr, mr) * len(x)
        pred_y.extend(np.argmax(model.y, axis=1))

    # pool = mp.Pool(proc)
    # callback = pool.map(subcalc, range(8))
    # print len(callback)
    # for i in range(len(callback)):
    #     print callback[i][0]
    #     sum_loss += callback[i][0]
    #     print callback[i][1]
    #     pred_y.extend(callback[i][1])
    
    # threads = []

    # for t in range(8):
    #     mt = MyThread(t)
    #     mt.start()
    #     threads.append(mt)

    # for mt in threads:
    #     mt.join()

    loss = sum_loss / n_train
    accuracy = np.sum(np.eye(10)[pred_y] * y_train[perm]) / n_train
    # print 'Train loss %.3f, accuracy %.4f |' %(loss, accuracy),
    print '%.3f     %.4f  |' %(loss, accuracy),
    
    # Testing
    sum_loss = 0
    pred_y = []
    
    for i in xrange(0, n_test, batchsize):
        x = x_test[i: i+batchsize]
        t = y_test[i: i+batchsize]
    
        sum_loss += model.test(x, t) * len(x)
        pred_y.extend(np.argmax(model.y, axis=1))

    loss = sum_loss / n_test
    accuracy = np.sum(np.eye(10)[pred_y] * y_test) / n_test
    # print 'Test loss %.3f, accuracy %.4f' %(loss, accuracy)
    print '%.3f     %.4f  |' %(loss, accuracy)
    # loss_list.append(loss)
    
# x = np.arange(1, n_epoch, 1)
# y = loss_list
# plt.plot(x, y)
# plt.show()
