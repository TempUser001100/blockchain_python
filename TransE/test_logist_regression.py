# #!/usr/bin/python
# # -*- coding: UTF-8 -*-
 
# import threading
# import time
 
# exitFlag = 0
 
# class myThread (threading.Thread):   #继承父类threading.Thread
#     def __init__(self, threadID, name, counter):
#         threading.Thread.__init__(self)
#         self.threadID = threadID
#         self.name = name
#         self.counter = counter
#     def run(self):                   #把要执行的代码写到run函数里面 线程在创建后会直接运行run函数 
#         print ("Starting " + self.name)
#         print_time(self.name, self.counter, 5)
#         print ("Exiting " + self.name)
 
# def print_time(threadName, delay, counter):
#     while counter:
#         if exitFlag:
#             (threading.Thread).exit()
#         time.sleep(delay)
#         print ("%s: %s" % (threadName, time.ctime(time.time())))
#         counter -= 1
 
# # 创建新线程
# thread1 = myThread(1, "Thread-1", 1)
# thread2 = myThread(2, "Thread-2", 2)
 
# # 开启线程
# thread1.start()
# thread2.start()
 
# print ("Exiting Main Thread")

import os
import csv
import pandas as pd
import numpy as np
import scipy.io as scio
from math import sqrt
from matplotlib import pyplot
from pandas import DataFrame
from pandas import concat
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression


def add_data(path):
	data = scio.loadmat(path)
	a = np.concatenate((data['entity_embedding'],data['relation_embedding']),axis=0)
	c_hat = PCA(n_components=5).fit_transform(a)
	c = np.concatenate(c_hat,axis=0)
	c = np.append(c,data['loss'])
	return c

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	global n_vars
	n_vars = 1 if type(data) is list else data.shape[1]
	# n_vars = 8
	df = DataFrame(data)
	# print(df)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
		# print(cols)
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		# print(cols)
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	# print(agg.shape)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg
def filter_file(files):
	Loss = 99999999
	new_file=[]
	for item in range(len(files)):
		file_path = 'data/test_data/'+str(item+1)+'.mat'
		data = scio.loadmat(file_path)
		if data['loss']<Loss:
			Loss = data['loss']
			new_file.append(item)
	return new_file

files = os.listdir('data/test_data/')

rows = []
if len(files)==0:
    print('no files')
else:
	for item in range(len(files)):
		path = 'data/test_data/'+str(item+1)+".mat"
		row = add_data(path)
		rows.append(row)
rows = np.array(rows)
rows = rows.astype('float32')

reframed = DataFrame(rows)
values = reframed.values
train_y = values[:1000,-1]
test_y = values[1000:,-1]
train_y = np.array(train_y,dtype=int)
test_y = np.array(test_y,dtype=int)
# drop_col = list(range(n_vars,values.shape[1]-1))
# reframed.drop(reframed.columns[drop_col], axis=1, inplace=True)
# values = reframed.values
train_X = values[:1000,:-1]
test_X = values[1000:,:-1]
print(test_X.shape)
train_X = PCA(n_components=200).fit_transform(train_X)
# print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
clf = LogisticRegression(random_state=0,solver='lbfgs').fit(train_X,train_y)
# make a prediction

# yhat = clf.predict(test_X)
# print(test_y,yhat)
# calculate RMSE
# rmse = sqrt(mean_squared_error(test_y, yhat))
# print('Test RMSE: %.3f' % rmse)

#read the four node data
dataset = ['fb15k-1','fb15k-2','fb15k-3','fb15k-4']
result,node = [],[]
for item in dataset:
	data_path = 'data/all_node_data/'+item+'.mat'
	data = scio.loadmat(data_path)
	a = np.concatenate((data['entity_embedding'],data['relation_embedding']),axis=0)
	c_hat = PCA(n_components=5).fit_transform(a)
	c = np.concatenate(c_hat,axis=0)
	node.append(c)
node = np.array(node)
node = node.astype('float32')
print(node)
test_X = np.concatenate((test_X,node),axis=0)
print(test_X[-4:-1,:])
test_X = PCA(n_components=200).fit_transform(test_X)
# node_values = DataFrame(node).values
# node_values = PCA(n_components=200).fit_transform(node_values)
# print(node_values.shape)
_loss = clf.predict(test_X)
result.append(_loss)
print(result)