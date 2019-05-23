import os
import random 
import pandas as pd 
import numpy as np
import argparse
parser = argparse.ArgumentParser(description='Process')
parser.add_argument('--train_path', type=str, default='data/FB15k-1/train.txt')
parser.add_argument('--valid_path', type=str, default='data/FB15k-1/valid.txt')
parser.add_argument('--test_path', type=str, default='data/FB15k-1/test.txt')
parser.add_argument('--per', type=float, default=0.9)
args = parser.parse_args()
data_dir = 'data/FB15k'
entity_dict_file = 'entity2id.txt'
relation_dict_file = 'relation2id.txt'
training_file = 'train.txt'
validation_file = 'valid.txt'
test_file = 'test.txt'
base_triple = []
# per = 0.9
# read the data 
entity_df = pd.read_table(os.path.join(data_dir, entity_dict_file), header=None)
relation_df = pd.read_table(os.path.join(data_dir, relation_dict_file), header=None)
train_df = pd.read_table(os.path.join(data_dir, training_file), header=None)
valid_df = pd.read_table(os.path.join(data_dir, validation_file), header=None)
test_df = pd.read_table(os.path.join(data_dir, test_file), header=None)
triple_df = pd.concat([train_df,valid_df,test_df],axis = 0,ignore_index = True)

#save the triple_df as .txt
# np.savetxt(r'data/triple.txt',triple_df.values,fmt='%s	%s	%s',delimiter='\t')

for r in relation_df[0]:
	for t in triple_df.values:
		# print(t,base_triple)
		if r in t:
			base_triple.append(list(t))
			triple_df.values.tolist().remove(list(t))
			break
# print(base_triple)
# print(len(base_triple))
for e in entity_df[0]:
	tmp1 = [x[0] for x in base_triple]
	tmp2 = [x[1] for x in base_triple]
	if e in tmp1 or e in tmp2:
		continue
	else:
		for t in triple_df.values:
			if e in t:
				base_triple.append(list(t))
				triple_df.values.tolist().remove(list(t))
				break
# print(len(base_triple))
# base_triple = list(set(base_triple))
# print(len(base_triple))

n_need = int(len(triple_df)*args.per)-int(len(base_triple))
base_triple = base_triple+triple_df.sample(n_need).values.tolist()
train_triple = random.sample(base_triple,int(0.7*len(base_triple)))
print(len(train_triple))
for item in train_triple:
	base_triple.remove(item)
valid_triple = random.sample(base_triple,int(0.5*len(base_triple)))
for item in valid_triple:
	base_triple.remove(item)
test_triple = base_triple

print(len(valid_triple))
print(len(test_triple))
np.savetxt(args.train_path,train_triple,fmt='%s	%s	%s')
np.savetxt(args.valid_path,valid_triple,fmt='%s	%s	%s')
np.savetxt(args.test_path,test_triple,fmt='%s	%s	%s')