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
from keras.layers import Activation
from keras.initializers import VarianceScaling
from keras.layers import AveragePooling2D
#from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K
from keras.optimizers import Adam
from keras.datasets import cifar10
from keras.models import load_model
from keras.models import Sequential, Model

from keras.applications.vgg16 import preprocess_input
from keras.utils import plot_model

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

from keras.callbacks import TensorBoard


import tensorflow as tf


from scipy.misc import imresize
from cv2 import imwrite

from tensorflow.python.client import device_lib

class PerceptualSimilarityGAN():




	def __init__(self,batch_size,w_i,w_f,w_a,comparator_network='VGG16',gan_name='gan',lr=.002,decay=0.5):
		self.img_shape=(256,256,3)
		input_image=Input(shape=(256,256,3)) 

		self.comp_network=Comparator(network=comparator_network)
		self.feat_comp_size=self.comp_network.model.layers[19].output_shape
		self.gen_inp_size=self.comp_network.model.layers[20].output_shape
		self.noise_std=K.variable([0],dtype='float32')
		self.batch_size=batch_size

		#K.set_learning_phase(1) 

		self.generator=self.build_generator()

		#self.feature_model=K.function([self.comp_network.model.layers[0].input],[self.comp_network.model.layers[53].output,self.comp_network.model.layers[54].output])
		x=Lambda(lambda inputs: tf.image.resize_bilinear(inputs,tf.stack([224,224]),align_corners=True))(input_image)
		l1=Model(self.comp_network.model.input,self.comp_network.model.layers[19].output)
		l2=Model(self.comp_network.model.input,self.comp_network.model.layers[20].output)
		y1=l1(x)
		y2=l2(x)
		self.feature_model=Model(input_image,[y1,y2],name="Comparator")
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
		x=Activation('tanh')(x)
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
		x=Conv2D(32,(7,7),strides=(4,4),name='conv1')(x)

		x=Conv2D(64,(5,5),strides=(1,1),name='conv2')(x)
		x=LeakyReLU(alpha=0.3)(x)
		x=BatchNormalization()(x)
		x=Conv2D(128,(3,3),strides=(2,2),name='conv3')(x)
		x=LeakyReLU(alpha=0.3)(x)
		x=BatchNormalization()(x)
		x=Conv2D(256,(3,3),strides=(1,1),name='conv4')(x) 
		x=LeakyReLU(alpha=0.3)(x)
		x=BatchNormalization()(x)
		x=Conv2D(256,(3,3),strides=(2,2),name='conv5')(x)
		x=LeakyReLU(alpha=0.3)(x)
		x=BatchNormalization()(x)
		x=AveragePooling2D(pool_size=(11,11),strides=(11,11),name='pool5')(x)
		y0=Flatten(name='pool5_reshape')(x)
		x=Dense(1024,name='feat_fc1')(features_noised)
		x=LeakyReLU(alpha=0.3)(x)
		x=BatchNormalization()(x)
		y1=Dense(512,name='feat_fc2')(x)
		y1=LeakyReLU(alpha=0.3)(y1)
		y1=BatchNormalization()(y1)
		x=Concatenate(axis=1,name='concat_fc5')([y0,y1])
		x=Dropout(0.5,name='drop5')(x)
		x=Dense(512,name='fc6')(x)
		x=LeakyReLU(alpha=0.3)(x)
		x=BatchNormalization()(x)
		x=Dropout(0.5,name='drop6')(x)
		x=Dense(2,name='fc7',activation=None)(x)

		model=Model([image,features],x,name="Discriminator")

		return model

	def save_model(self):
		np.save("models/GANs/Combined",self.combined.get_weights())
		np.save("models/GANs/Generator",self.generator.get_weights())
		np.save("models/GANs/Discriminator",self.discriminator.get_weights())

	def load_model(self):

		self.combined.set_weights(np.load("models/GANs/Combined.npy"))
		self.generator.set_weights(np.load("models/GANs/Generator.npy"))
		self.discriminator.set_weights(np.load("models/GANs/Discriminator.npy"))

	def train(self,epochs):
		batch_size=self.batch_size

		datagen=ImageDataGenerator()
		train_generator=datagen.flow_from_directory('/home/fas/chun/sk2436/project/imagenet/training',target_size=(256,256),
			batch_size=batch_size)

		log_d=open('disc_loss','a')
		#log_d.write("START\n")
		log_g=open('gen_loss','a')
		#log_g.write("START\n")



		valid=np.asarray([[1,0] for i in range(batch_size)])
		fake=np.asarray([[0,1] for i in range(batch_size)])
		
		#tbCallback=TensorBoard(log_dir='Graph',batch_size=16)
		#tbCallback.set_model(self.combined) 
		try:
			for step in range(1,epochs+1):
				K.set_value(self.noise_std,[2/256.0*(1-step/500000.0)])
				batch_data=train_generator.next()
				imgs_raw=batch_data[0]
				print(imgs_raw.shape)
				
				imgs=preprocess_input(imgs_raw)

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

				g_loss=self.combined.train_on_batch(imgs,[valid,synthetic_images,synth_comp_features])
				if step%50==0 or step==0:
					log_d.write(str(step)+","+str(d_loss[0])+","+str(d_loss[1])+"\n")
					log_d.flush()
					log_g.write(str(step)+","+str(g_loss[0])+","+str(g_loss[1])+","+str(g_loss[2])+","+str(g_loss[3])+"\n")
					log_g.flush()
				#print ("%d/%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (step, epochs,d_loss[0], 100*d_loss[1], g_loss))
				
				#if step>=epochs:
				#	break
				if step%10000==0:
					print("SAVING MODEL")
					self.save_model()


		except KeyboardInterrupt:
			print("Ending Training...")
			self.save_model()
		
		self.save_model()
		log_d.close()
		log_g.close()
		print("Finish Training")

	from scipy.stats import pearsonr
	
	def test_single_image(self,img_path):
		from PIL import Image
		from scipy.stats import pearsonr
		input_image=image.load_img(img_path,target_size=(256,256))
		input_image=image.img_to_array(input_image)
		input_image=np.expand_dims(input_image, axis = 0)
		input_image=preprocess_input(input_image)
		#print(input_image.shape)
		features=self.feature_model.predict(input_image)[1] 
		synthetic_images=self.generator.predict(features)
		synthetic_image=synthetic_images[0]
		formatted = (synthetic_image * 255 / np.max(synthetic_image)).astype('uint8')
		features2=self.feature_model.predict(synthetic_images)[1]
		img = Image.fromarray(formatted)
		img.save("recon.png")
		print(features.shape)
		print(features2.shape)
		print(pearsonr(features[0,:],features2[0,:]))
		


		

if __name__=='__main__':
	print("DEVICE(S) USED")
	print(device_lib.list_local_devices())
	gan=PerceptualSimilarityGAN(50,1,.01,.001)
	plot_model(gan.combined, to_file='model.png')
	#gan.load_model()
	#gan.test_single_image("dog.jpg")
	gan.train(200000)



































