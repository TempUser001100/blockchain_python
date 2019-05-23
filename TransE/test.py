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
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras import optimizers


def add_data(path):
	data = scio.loadmat(path)
	a = np.concatenate((data['entity_embedding'],data['relation_embedding']),axis=0)
	# print(a.shape)
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
# data = pd.read_csv('data/data.csv')
# print(data)
files = os.listdir('data/test_data/')
# files = filter_file(files)

rows = []
if len(files)==0:
    print('no files')
else:
	for item in range(len(files)):
		path = 'data/test_data/'+str(item+1)+".mat"
		row = add_data(path)
		rows.append(row)
# 	print(path)
# with open("data/data.csv",'w') as csvfile:
# 	writer = csv.writer(csvfile)
# 	writer.writerows(rows)
# print(rows.shape)
# exit()
rows = np.array(rows)
rows = rows.astype('float32')
# X = rows[:,:-1]
# Y = rows[:,-1]
# print(X.shape,Y.shape)
# X = preprocessing.scale(_X)
# Y = preprocessing.scale(_Y)
# reframed = series_to_supervised(rows,1,1)
reframed = DataFrame(rows)
values = reframed.values
train_y = values[:1000,-1]
test_y = values[1000:,-1]
# drop_col = list(range(n_vars,values.shape[1]-1))
# reframed.drop(reframed.columns[drop_col], axis=1, inplace=True)
# values = reframed.values
train_X = values[:1000,:-1]
test_X = values[1000:,:-1]
train_X = PCA(n_components=200).fit_transform(train_X)
test_X = PCA(n_components=200).fit_transform(test_X)
# reshape the train/test dataset
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dropout(0.5))
model.add(Dense(1))
Adam = optimizers.Adam(lr=0.01)
model.compile(loss='mae', optimizer=Adam)
#checkpoint
filepath = 'model/150.h5'
checkpoint = ModelCheckpoint(filepath,monitor='val_loss',verbose=1,save_best_only=True,mode='min')
callbacks_list = [checkpoint]
# fit network
history = model.fit(train_X, train_y, epochs=300, batch_size=64, callbacks=callbacks_list, validation_data=(test_X, test_y), verbose=2, shuffle=False)
#save the model
# model.save('model/model.h5')
# make a prediction

yhat = model.predict(test_X)
print(test_y,yhat)
# test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# # invert scaling for forecast
# inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
# inv_yhat = scaler.inverse_transform(inv_yhat)
# inv_yhat = inv_yhat[:,0]
# print(inv_yhat)
# # invert scaling for actual
# test_y = test_y.reshape((len(test_y), 1))
# inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
# inv_y = scaler.inverse_transform(inv_y)
# inv_y = inv_y[:,0]
# calculate RMSE
rmse = sqrt(mean_squared_error(test_y, yhat))
print('Test RMSE: %.3f' % rmse)
# plot history
pyplot.switch_backend('agg')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.savefig("pic/150.png", format = "png", dpi = 150, bbox_inches = "tight")