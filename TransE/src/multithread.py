
#!/usr/bin/python
# -*- coding: UTF-8 -*-
 
import threading
import time
import subprocess
import os
import main
from dataset import KnowledgeGraph
from model import TransE

import tensorflow as tf
 
exitFlag = 0
Max = 9999
n = 10 

class myThread (threading.Thread):   #继承父类threading.Thread
    def __init__(self, threadID,path,embedding_dims,max_epoch,batch_size,learning_rate):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.path = path
        self.embedding_dims = embedding_dims
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.score_func = 'L1'
        self.n_generator = 12
        self.n_rank_calculator = 12
        self.margin_value =1
    def run(self):                   #把要执行的代码写到run函数里面 线程在创建后会直接运行run函数 
        print ("Starting " + self.name)
        # print_time(self.name, self.max_epoch, 5)
        # self.output = execute_bash(self.name)
        # e,r = main.main(self.path,self.embedding_dim,self.max_epoch,self.batch_size,self.learning_rate)
        transe(self.path,self.embedding_dims,self.margin_value,self.score_func,self.batch_size,
        	self.learning_rate,self.n_generator,self.n_rank_calculator,self.max_epoch)
        # print(e,r)
        print ("Exiting " + self.name)
 
def print_time(threadName, delay, counter=5):
    while counter:
        if exitFlag:
            (threading.Thread).exit()
        time.sleep(delay)
        print ("%s: %s" % (threadName, time.ctime(time.time())))
        counter -= 1

def transe(data_path,embedding_dims,margin_value,score_func,batch_size,learning_rate,n_generator,n_rank_calculator,max_epoch):
    kg = KnowledgeGraph(data_dir=data_path)
    kge_model = TransE(kg=kg, embedding_dim=embedding_dims, margin_value=margin_value,
                       score_func=score_func, batch_size=batch_size, learning_rate=learning_rate,
                       n_generator=n_generator, n_rank_calculator=n_rank_calculator)
    gpu_config = tf.GPUOptions(allow_growth=True)
    sess_config = tf.ConfigProto(gpu_options=gpu_config)
    with tf.Session(config=sess_config) as sess:
        print('-----Initializing tf graph-----')
        tf.global_variables_initializer().run()
        print('-----Initialization accomplished-----')
        entity_embedding,relation_embedding = kge_model.check_norm(session=sess)
        summary_writer = tf.summary.FileWriter(logdir='../summary/', graph=sess.graph)
        for epoch in range(max_epoch):
            print('=' * 30 + '[EPOCH {}]'.format(epoch) + '=' * 30)
            kge_model.launch_training(session=sess, summary_writer=summary_writer)
            if (epoch + 1) % 10 == 0:
                kge_model.launch_evaluation(session=sess)
    return entity_embedding,relation_embedding

def record(e,r,loss):
     if loss<Max:
         pass

# while n>0:
    # 创建新线程
thread1 = myThread(1, "../data/FB15k-1/",100,10,10000,0.003)
# thread2 = myThread(2, "../data/FB15k-2/",100,20,10000,0.003)
# thread3 = myThread(3, "../data/FB15k-3/",100,20,10000,0.003)

# 开启线程
thread1.start()
# thread2.start()
# thread3.start()
# thread1.join()
# thread2.join()
# thread3.join()
