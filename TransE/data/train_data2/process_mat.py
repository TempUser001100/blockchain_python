import scipy.io as scio
import numpy as np
from sklearn.decomposition import PCA
data = scio.loadmat('1.mat')
en_1 = np.concatenate(data['entity_embedding_1'],axis=0)
en_2 = np.concatenate(data['entity_embedding_2'],axis=0)
en_3 = np.concatenate(data['entity_embedding_3'],axis=0)
en_4 = np.concatenate(data['entity_embedding_4'],axis=0)
rel_1 = np.concatenate(data['relation_embedding_1'],axis=0)
rel_2 = np.concatenate(data['relation_embedding_2'],axis=0)
rel_3 = np.concatenate(data['relation_embedding_3'],axis=0)
rel_4 = np.concatenate(data['relation_embedding_4'],axis=0)
entity = np.concatenate(([en_1],[en_2],[en_3],[en_4]),axis=0)
relation = np.concatenate(([rel_1],[rel_2],[rel_3],[rel_4]),axis=0)
_entity = np.transpose(entity)
_relation = np.transpose(relation)
_entity = PCA(n_components=1).fit_transform(_entity).reshape((-1,100))
_relation = PCA(n_components=1).fit_transform(_relation).reshape((-1,100))

print(en_1.shape,rel_1.shape,_entity.shape,_relation.shape)
