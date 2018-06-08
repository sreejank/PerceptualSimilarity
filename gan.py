import numpy as np
from comparator import Comparator

from keras.applications.imagenet_utils import preprocess_input
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2DTranspose
from keras.layers import Conv2D
from keras.layers import GaussianNoise
from keras.layers import LeakyReLU
from keras.layers import Flatten
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Lambda
from keras.initializers import VarianceScaling
from keras.layers import AveragePooling2D
from keras import backend as K
from keras.optimizers import Adam
from keras.datasets import cifar10
from keras.models import load_model
from keras.models import Sequential, Model

from keras.applications.vgg16 import preprocess_input
from keras.utils import plot_model

import tensorflow as tf


from scipy.misc import imresize

class PerceptualSimilarityGAN():




	def __init__(self,batch_size,w_i,w_f,w_a,comparator_network='VGG16',gan_name='gan',lr=.002,decay=0.5):
		self.img_shape=(256,256,3)
		input_image=Input(shape=(256,256,3)) 

		self.comp_network=Comparator(network=comparator_network)
		self.feat_comp_size=self.comp_network.model.layers[19].output_shape
		self.gen_inp_size=self.comp_network.model.layers[20].output_shape
		self.noise_std=2/256.0
		self.batch_size=batch_size

		#K.set_learning_phase(1) 

		self.generator=self.build_generator()

		#self.feature_model=K.function([self.comp_network.model.layers[0].input],[self.comp_network.model.layers[53].output,self.comp_network.model.layers[54].output])
		x=Lambda(lambda inputs: tf.image.resize_bilinear(inputs,tf.stack([224,224]),align_corners=True))(input_image)
		l1=Model(self.comp_network.model.input,self.comp_network.model.layers[19].output)
		l2=Model(self.comp_network.model.input,self.comp_network.model.layers[20].output)
		y1=l1(x)
		y2=l2(x)
		self.feature_model=Model(input_image,[y1,y2])
		#feature_model=Model(self.comp_network.model.input,[self.comp_network.model.layers[19].output,self.comp_network.model.layers[20].output],name="Comparator")
		#feature_model.trainable=False



		feats=self.feature_model(input_image)
		image_comp_features=feats[0]
		image_gen_input=feats[1]



		synth_image=self.generator(image_gen_input)

		synth_feats=self.feature_model(synth_image)
		synth_comp_features=synth_feats[0]
		synth_gen_input=synth_feats[1]

		self.discriminator=self.build_discriminator(self.noise_std)
		dis_optimizer=Adam(lr=lr*.1,decay=decay)
		self.discriminator.compile(loss='binary_crossentropy',optimizer=dis_optimizer,metrics=['accuracy'])

		validity=self.discriminator([synth_image,synth_comp_features])

		gen_optimizer=Adam(lr=lr,decay=decay)

		
		self.combined=Model(input_image,[validity,synth_image,synth_comp_features])
		self.combined.compile(optimizer=gen_optimizer,
			loss=['binary_crossentropy','mean_squared_error','mean_squared_error'],
			loss_weights=[w_a,w_i,w_f])



		#self.image_validity=self.discriminator([self.input_image,self.image_comp_features])
		#self.synth_validity=self.discriminator([self.synth_image,self.synth_comp_features])


	def build_generator(self):
		var_initializer=VarianceScaling(scale=2.0,mode='fan_in',distribution='normal')
		
		reverse_layer=Input(shape=(self.gen_inp_size[1],))
		x=Dense(4096, input_dim=self.gen_inp_size[1],name='defc7',activation='relu',init=var_initializer)(reverse_layer)
		x=BatchNormalization()(x)
		
		x=Dense(4096,name='defc6',activation='relu',init=var_initializer)(x)
		x=BatchNormalization()(x)
		x=Dense(4096,name='defc5',activation='relu',init=var_initializer)(x)
		x=BatchNormalization()(x)
		x=Reshape([4,4,256])(x)
		x=Conv2DTranspose(256,4,strides=2,name='deconv5',activation='relu',use_bias=False,kernel_initializer=var_initializer,padding='same')(x)
		x=BatchNormalization()(x)
		x=Conv2DTranspose(512,3,strides=1,name='deconv5_1',activation='relu',use_bias=False,kernel_initializer=var_initializer,padding='same')(x)
		x=BatchNormalization()(x)
		x=Conv2DTranspose(256,4,strides=2,name='deconv4',activation='relu',use_bias=False,kernel_initializer=var_initializer,padding='same')(x)
		x=BatchNormalization()(x)
		x=Conv2DTranspose(256,3,strides=1,name='deconv4_1',activation='relu',use_bias=False,kernel_initializer=var_initializer,padding='same')(x)
		x=BatchNormalization()(x)
		x=Conv2DTranspose(128,4,strides=2,name='deconv3',activation='relu',use_bias=False,kernel_initializer=var_initializer,padding='same')(x)
		x=BatchNormalization()(x)
		x=Conv2DTranspose(128,3,strides=1,name='deconv3_1',activation='relu',use_bias=False,kernel_initializer=var_initializer,padding='same')(x)
		x=BatchNormalization()(x)
		x=Conv2DTranspose(64,4,strides=2,name='deconv2',activation='relu',use_bias=False,kernel_initializer=var_initializer,padding='same')(x)
		x=BatchNormalization()(x)
		x=Conv2DTranspose(32,4,strides=2,name='deconv1',activation='relu',use_bias=False,kernel_initializer=var_initializer,padding='same')(x)
		x=BatchNormalization()(x)
		x=Conv2DTranspose(3,4,strides=2,name='deconv0',activation=None,use_bias=False,kernel_initializer=var_initializer,padding='same')(x)
		#model.add(Lambda(lambda x: K.resize_images(x,1.0/8,1.0/8,"channels_last")))
		x=Reshape(self.img_shape)(x)

		#model.summary()

		#reverse_layer=Input(shape=(self.gen_inp_size[1],))
		model=Model(reverse_layer,x,name="Generator")

		return model

	def build_discriminator(self,noise_param):
		leaky=LeakyReLU(alpha=0.3)
		image=Input(shape=(256,256,3))



		features=Input(shape=(self.feat_comp_size[1],))

		features_noised=GaussianNoise(noise_param*2)(features)



		x=GaussianNoise(noise_param)(image)
		x=Conv2D(32,(7,7),strides=(4,4),name='conv1',activation=leaky)(x)
		x=Conv2D(64,(5,5),strides=(1,1),name='conv2',activation=leaky)(x)
		x=BatchNormalization()(x)
		x=Conv2D(128,(3,3),strides=(2,2),name='conv3',activation=leaky)(x)
		x=BatchNormalization()(x)
		x=Conv2D(256,(3,3),strides=(1,1),name='conv4',activation=leaky)(x) 
		x=BatchNormalization()(x)
		x=Conv2D(256,(3,3),strides=(2,2),name='conv5',activation=leaky)(x)
		x=BatchNormalization()(x)
		x=AveragePooling2D(pool_size=(11,11),strides=(11,11),name='pool5')(x)
		y0=Flatten(name='pool5_reshape')(x)
		x=Dense(1024,name='feat_fc1',activation=leaky)(features_noised)
		x=BatchNormalization()(x)
		y1=Dense(512,name='feat_fc2',activation=leaky)(x)
		y1=BatchNormalization()(y1)
		x=Concatenate(axis=1,name='concat_fc5')([y0,y1])
		x=Dropout(0.5,name='drop5')(x)
		x=Dense(512,activation=leaky,name='fc6')(x)
		x=BatchNormalization()(x)
		x=Dropout(0.5,name='drop6')(x)
		x=Dense(2,name='fc7',activation=None)(x)

		model=Model([image,features],x)

		return model

	def save_model(self):
		self.combined.save("GANs/Combined")
		self.generator.save("GANs/Generator")
		self.discriminator.save("GANs/Discriminator")

	def load_model(self):

		self.combined=load_model("GANs/Combined")
		self.generator=load_model("GANs/Generator")
		self.discriminator=load_model("GANs/Discriminator")

	def train(self,epochs):
		batch_size=self.batch_size

		(X_train, _), (_, _) = cifar10.load_data()

		mean = 120.707
		std = 64.15
		X_train=(X_train-mean)/(std+1e-7)

		



		valid=np.asarray([[1,0] for i in range(batch_size)])
		fake=np.asarray([[0,1] for i in range(batch_size)])
		try:
			for step in range(epochs):
				idx = np.random.randint(0, X_train.shape[0], batch_size)
				imgs_raw = X_train[idx]
				
				imgs=np.zeros((imgs_raw.shape[0],256,256,3))
				#imgs2=np.zeros((imgs_raw.shape[0],224,224,3))
				for i in range(imgs.shape[0]):
					imgs[i,:,:,:]=imresize(imgs_raw[i,:,:,:],(256,256,3))
					#imgs2[i,:,:,:]=imresize(imgs_raw[i,:,:,:],(224,224,3))
				
				imgs=preprocess_input(imgs)

				feats=self.feature_model.predict_on_batch(imgs)

				imgs_comp_features=feats[0]
				imgs_gen_input=feats[1]

				synthetic_images=self.generator.predict_on_batch(imgs_gen_input)
				
				synthetic_images=preprocess_input(synthetic_images)
				
				feats_synthetic=self.feature_model.predict_on_batch(synthetic_images)
				synth_comp_features=feats_synthetic[0]
				synth_gen_input=feats_synthetic[1]

				#Train Discriminator
				d_loss_real = self.discriminator.train_on_batch([imgs,imgs_comp_features], valid)
				d_loss_fake = self.discriminator.train_on_batch([synthetic_images,synth_comp_features], fake)
				d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

				#Train Generator
				


				g_loss=self.combined.train_on_batch(imgs,[valid,imgs,imgs_comp_features])
				print(str(step)+"/"+str(epochs)+"[D loss: "+str(d_loss)+"] [G loss: "+str(g_loss)+"]")
				#print ("%d/%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (step, epochs,d_loss[0], 100*d_loss[1], g_loss))

				if step%30==0:
					print("SAVING MODEL")
					self.save_model()


		except KeyboardInterrupt:
			print("Ending Training...")
			self.save_model()
		
		self.save_model()
		print("Finish Training")

if __name__=='__main__':
	gan=PerceptualSimilarityGAN(20,1,.1,.001)
	plot_model(gan.combined, to_file='model.png')
	gan.train(30000)



































