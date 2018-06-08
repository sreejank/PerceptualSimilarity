import numpy as np
from keras.datasets import cifar10
import matplotlib.pyplot as plt
from scipy.misc import imresize
plt.switch_backend('agg')
(X_train, _), (_, _) = cifar10.load_data()
mean = 120.707
std = 64.15
#X_train=(X_train-mean)/(std+1e-7)

fig=plt.figure()
print(X_train.shape)
#plt.imshow(X_train[0,:,:,:])
first=X_train[0,:,:,:]
for i in range(X_train.shape[0]):
	if i%10==0:
		print(i)
	first2=imresize(X_train[i,:,:,:],(256,256,3))
plt.imsave("test2.png",first) 