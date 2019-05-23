from multiprocessing import Process,Manager
from dataset import KnowledgeGraph
import tensorflow as tf
from model import TransE
import scipy.io as scio
import datetime
import time
import os

def transe(id,data_path,embedding_dims,margin_value,score_func,batch_size,learning_rate,n_generator,n_rank_calculator,max_epoch,d):
    kg = KnowledgeGraph(data_dir=data_path)
    content = []
    kge_model = TransE(kg=kg, embedding_dim=embedding_dims, margin_value=margin_value,
                       score_func=score_func, batch_size=batch_size, learning_rate=learning_rate,
                       n_generator=n_generator, n_rank_calculator=n_rank_calculator)
    gpu_config = tf.GPUOptions(allow_growth=True)
    sess_config = tf.ConfigProto(gpu_options=gpu_config)
    with tf.Session(config=sess_config) as sess:
        print('-----Initializing tf graph-----')
        tf.global_variables_initializer().run()
        print('-----Initialization accomplished-----')
        # loss,entity_embedding,relation_embedding = kge_model.check_norm(session=sess)
        summary_writer = tf.summary.FileWriter(logdir='../summary/', graph=sess.graph)
        for epoch in range(max_epoch):
            print('=' * 30 + '[EPOCH {}]'.format(epoch) + '=' * 30)            
            # print(loss)
            kge_model.launch_training(session=sess, summary_writer=summary_writer)
            if (epoch + 1) % 1000 == 0:
                kge_model.launch_evaluation(session=sess)
        loss,entity_embedding,relation_embedding = kge_model.check_norm(session=sess)
        content.append(loss)
        content.append(entity_embedding)
        content.append(relation_embedding)
        d[id] = content
        # print(type(d))
    return entity_embedding,relation_embedding

def record(d,num):
    global Loss
    min_id = 0
    max = 99999999999
    for key,value in d.items():
        # print(key,value)
        if float(value[0])<max:
            max = value[0]
            min_id = key
    if d[min_id][0]<Loss:
        print("loss:"+str(d[min_id][0]))
        Loss = d[min_id][0]
        scio.savemat('../data/train_data2/'+str(num),{'loss':d[min_id][0],'entity_embedding':d[min_id][1],'relation_embedding':d[min_id][2]})
        return True
    else:
        print('损失小于前一个，不作记录')
        return False
	# file_name =datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')





 
if __name__ == '__main__':
    if len(os.listdir('../data/train_data2/')):
        num = len(os.listdir('../data/train_data2/'))
    else:
        num = 1
    global Loss 
    Loss = 999999
    while num<5000:
        manager = Manager()
        d = manager.dict()
        jobs =[]
        for i in range(1,5):
            path = "../data/FB15k-"+str(i)
            jobs.append(Process(target=transe, args=(i,path,100,1,'L1',10000,0.003,12,12,50,d)))
        for j in jobs:
            j.start()
        for j in jobs:
            j.join()
        print('--------第{}次---------'.format(num))
        flag = record(d,num)
        if flag:
            num += 1
    # path = "../data/FB15k-1"
    # p = Process(target=transe, args=(1,path,100,1,'L1',10000,0.003,12,12,3,d))
    # p.start()
    # p.join()
