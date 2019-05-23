import scipy.io as scio
import matplotlib.pyplot as plt
import os
loss = []
for item in range(len(os.listdir('data/train_data3/'))):
	path = 'data/train_data/'+str(item+1)+'.mat'
	data = scio.loadmat(path)
	loss.append(data['loss'].tolist()[0][0])
plt.switch_backend('agg')
plt.xlabel('x')
plt.ylabel('y')
plt.plot(list(range(len(loss))),loss,'ro')
plt.legend()
plt.savefig("pic/pic3.png", format = "png", dpi = 150, bbox_inches = "tight")