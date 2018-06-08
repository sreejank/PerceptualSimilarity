from comparator import *
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.vgg16 import decode_predictions
# CPU MODE
#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""


"""c=Comparator(network="VGG16")
c.load_model("models/comparators/VGG16")

img_path="test_images/dog.jpg"
img=image.load_img(img_path, target_size=(224, 224))

x = image.img_to_array(img)
x=np.expand_dims(x,axis=0)
x=preprocess_input(x)"""


pretrained = VGG16(weights='imagenet', include_top=False)
print([layer.name for layer in pretrained.layers])

"""preds=c.model.predict(x)
print(decode_predictions(preds))

activations=c.get_activations(x)

import pickle
with open('dog_activations.pickle', 'wb') as handle:
	pickle.dump(activations, handle, protocol=pickle.HIGHEST_PROTOCOL)

print([layer.name for layer in c.model.layers])
"""