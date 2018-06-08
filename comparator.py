from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import merge
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.models import Model
from keras.models import Sequential
from keras.models import load_model
from keras import backend as K
from keras.optimizers import Adam
from keras.layers.core import Lambda
from keras.applications.vgg16 import VGG16
from keras import regularizers
import numpy as np
import argparse

import numpy as np
import matplotlib.pyplot as plt

class Comparator():
	def __init__(self,network='VGG16',weights_path=None,trainable=False,lr=0.00001,beta1=0.5):
		if network=='VGG16_cifar':
			model=VGG_16_cifar()
		elif network=='VGG16':
			model=VGG_16(weights_path=weights_path)
		elif network=='AlexNet':
			model=AlexNet(weights_path=weights_path)
		else:
			raise ValueError("Only VGG16 available")
		
		adam=Adam(lr=lr,beta_1=beta1)
		model.trainable=False
		model.compile(optimizer=adam,loss='categorical_crossentropy')
		self.model=model
		self.lr=lr
		self.beta1=beta1

	def save_model(self,path):
		self.model.save(path)
	def load_model(self,path):
		self.model=load_model(path)

	def get_activations(self, model_inputs, print_shape_only=True, layer_name=None):
		model=self.model
		print('----- activations -----')
		activations = []
		inp = model.input

		model_multi_inputs_cond = True
		if not isinstance(inp, list):
			# only one input! let's wrap it in a list.
			inp = [inp]
			model_multi_inputs_cond = False

		outputs = [layer.output for layer in model.layers if
				   layer.name == layer_name or layer_name is None]  # all layer outputs

		funcs = [K.function(inp + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions

		if model_multi_inputs_cond:
			list_inputs = []
			list_inputs.extend(model_inputs)
			list_inputs.append(0.)
		else:
			list_inputs = [model_inputs, 0.]

		# Learning phase. 0 = Test mode (no dropout or batch normalization)
		# layer_outputs = [func([model_inputs, 0.])[0] for func in funcs]
		layer_outputs = [func(list_inputs)[0] for func in funcs]
		for layer_activations in layer_outputs:
			activations.append(layer_activations)
			if print_shape_only:
				print(layer_activations.shape)
			else:
				print(layer_activations)
		return activations
	
	def display_activations(self,activation_maps):
		batch_size = activation_maps[0].shape[0]
		assert batch_size == 1, 'One image at a time to visualize.'
		for i, activation_map in enumerate(activation_maps):
			print('Displaying activation map {}'.format(i))
			shape = activation_map.shape
			if len(shape) == 4:
				activations = np.hstack(np.transpose(activation_map[0], (2, 0, 1)))
			elif len(shape) == 2:
				# try to make it square as much as possible. we can skip some activations.
				activations = activation_map[0]
				num_activations = len(activations)
				if num_activations > 1024:  # too hard to display it on the screen.
					square_param = int(np.floor(np.sqrt(num_activations)))
					activations = activations[0: square_param * square_param]
					activations = np.reshape(activations, (square_param, square_param))
				else:
					activations = np.expand_dims(activations, axis=0)
			else:
				raise Exception('len(shape) = 3 has not been implemented.')
			plt.figure()
			plt.imshow(activations, interpolation='None', cmap='jet')
			plt.savefig("activations.png")





"""
Layer Names
[conv2d_1', 'activation_1', 'batch_normalization_1', 'dropout_1', 'conv2d_2', 'activation_2', 'batch_normalization_2', 
'max_pooling2d_1', 'conv2d_3', 'activation_3', 'batch_normalization_3', 'dropout_2', 'conv2d_4', 'activation_4', 
'batch_normalization_4', 'max_pooling2d_2', 'conv2d_5', 'activation_5', 'batch_normalization_5', 'dropout_3', 
'conv2d_6', 'activation_6', 'batch_normalization_6', 'dropout_4', 'conv2d_7', 'activation_7', 'batch_normalization_7', 
'max_pooling2d_3', 'conv2d_8', 'activation_8', 'batch_normalization_8', 'dropout_5', 'conv2d_9', 'activation_9', 
'batch_normalization_9', 'dropout_6', 'conv2d_10', 'activation_10', 'batch_normalization_10', 'max_pooling2d_4', 
'conv2d_11', 'activation_11', 'batch_normalization_11', 'dropout_7', 'conv2d_12', 'activation_12', 'batch_normalization_12', 
'dropout_8', 'conv2d_13', 'activation_13', 'batch_normalization_13', 'max_pooling2d_5', 'dropout_9', 'flatten_1', 'dense_1', 
'activation_14', 'batch_normalization_14', 'dropout_10', 'dense_2', 'activation_15']"""
def VGG_16_cifar(weights_path='/home/fas/chun/sk2436/project/cifar-vgg/cifar10vgg.h5'):
	model = Sequential()
	weight_decay = 0.0005
	shape=[32,32,3]

	model.add(Conv2D(64, (3, 3), padding='same',input_shape=shape,kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.3))

	model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())

	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.4))

	model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())

	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.4))

	model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.4))

	model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())

	model.add(MaxPooling2D(pool_size=(2, 2)))


	model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.4))

	model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.4))

	model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())

	model.add(MaxPooling2D(pool_size=(2, 2)))


	model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.4))

	model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.4))

	model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())

	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.5))

	model.add(Flatten())
	model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())

	model.add(Dropout(0.5))
	model.add(Dense(10))
	model.add(Activation('softmax'))
	return model


#Layer names ['input_1', 'block1_conv1', 'block1_conv2', 'block1_pool', 'block2_conv1', 'block2_conv2', 
#'block2_pool', 'block3_conv1', 'block3_conv2', 'block3_conv3', 'block3_pool', 'block4_conv1', 
#'block4_conv2', 'block4_conv3', 'block4_pool', 'block5_conv1', 'block5_conv2', 'block5_conv3', 
# 'block5_pool', 'flatten', 'fc1', 'fc2', 'predictions']
def VGG_16(weights_path=None):
	return VGG16(weights='imagenet', include_top=True)


def crosschannelnormalization(alpha=1e-4, k=2, beta=0.75, n=5, **kwargs):
	"""
	This is the function used for cross channel normalization in the original
	Alexnet
	"""

	def f(X):
		b, ch, r, c = X.shape
		half = n // 2
		square = K.square(X)
		extra_channels = K.spatial_2d_padding(K.permute_dimensions(square, (0, 2, 3, 1)) 
											  , (0, half))
		extra_channels = K.permute_dimensions(extra_channels, (0, 3, 1, 2))
		scale = k
		for i in range(n):
			scale += alpha * extra_channels[:, i:i + ch, :, :] #Indexing issue here
		scale = scale ** beta
		return X / scale

	return Lambda(f, output_shape=lambda input_shape: input_shape, **kwargs)
	

def AlexNet(weights_path=None):
	inputs = Input(shape=(227, 227,3))

	conv_1 = Conv2D(96, 11, 11, subsample=(4, 4), activation='relu',
						   name='conv_1')(inputs)

	conv_2 = MaxPooling2D((3, 3), strides=(2, 2))(conv_1)
	conv_2 = crosschannelnormalization(name='convpool_1')(conv_2)
	conv_2 = ZeroPadding2D((2, 2))(conv_2)
	conv_2 = merge([
					   Conv2D(128, 5, 5, activation='relu', name='conv_2_' + str(i + 1))(
						   splittensor(ratio_split=2, id_split=i)(conv_2)
					   ) for i in range(2)], mode='concat', concat_axis=1, name='conv_2')

	conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
	conv_3 = crosschannelnormalization()(conv_3)
	conv_3 = ZeroPadding2D((1, 1))(conv_3)
	conv_3 = Conv2D(384, 3, 3, activation='relu', name='conv_3')(conv_3)

	conv_4 = ZeroPadding2D((1, 1))(conv_3)
	conv_4 = merge([
					   Conv2D(192, 3, 3, activation='relu', name='conv_4_' + str(i + 1))(
						   splittensor(ratio_split=2, id_split=i)(conv_4)
					   ) for i in range(2)], mode='concat', concat_axis=1, name='conv_4')

	conv_5 = ZeroPadding2D((1, 1))(conv_4)
	conv_5 = merge([
					   Conv2D(128, 3, 3, activation='relu', name='conv_5_' + str(i + 1))(
						   splittensor(ratio_split=2, id_split=i)(conv_5)
					   ) for i in range(2)], mode='concat', concat_axis=1, name='conv_5')

	dense_1 = MaxPooling2D((3, 3), strides=(2, 2), name='convpool_5')(conv_5)

	
	dense_1 = Flatten(name='flatten')(dense_1)
	dense_1 = Dense(4096, activation='relu', name='dense_1')(dense_1)
	dense_2 = Dropout(0.5)(dense_1)
	dense_2 = Dense(4096, activation='relu', name='dense_2')(dense_2)
	dense_3 = Dropout(0.5)(dense_2)
	dense_3 = Dense(1000, name='dense_3')(dense_3)
	prediction = Activation('softmax', name='softmax')(dense_3)

	model = Model(input=inputs, output=prediction)

	if weights_path:
		model.load_weights(weights_path)

	return model