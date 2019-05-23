import scipy.io as scio
from scipy import sparse
import numpy as np
#from numpy import mat
alpha = 0.9
data = scio.loadmat('1.mat')
W1 = data['entity_embedding_1']
W2 = data['entity_embedding_2']
W3 = data['entity_embedding_3']
W4 = data['entity_embedding_4']

#threshold = np.sort(-np.abs(W.flatten()))[int((len(W.flatten()-1)*0.4))]
threshold1 = alpha*np.std(W1.flatten())
threshold2 = alpha*np.std(W2.flatten())
threshold3 = alpha*np.std(W3.flatten())
threshold4 = alpha*np.std(W4.flatten())

mask1 = (np.abs(W1))>(np.abs(threshold1))
mask2 = (np.abs(W2))>(np.abs(threshold2))
mask3 = (np.abs(W3))>(np.abs(threshold3))
mask4 = (np.abs(W4))>(np.abs(threshold4))

mask1 = np.bool_(mask1)
mask2 = np.bool_(mask2)
mask3 = np.bool_(mask3)
mask4 = np.bool_(mask4)

_W1 = W1*mask1
_W2 = W2*mask2
_W3 = W3*mask3
_W4 = W4*mask4

sparse.save_npz('test1.npz',sparse.csc_matrix(_W1))
sparse.save_npz('test2.npz',sparse.csc_matrix(_W2))
sparse.save_npz('test3.npz',sparse.csc_matrix(_W3))
sparse.save_npz('test4.npz',sparse.csc_matrix(_W4))
