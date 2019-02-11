import sys
from PIL import Image
import numpy as np
import cv2 as cv

from keras.models import load_model
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array 


json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("model.h5")

print("Loaded model from disk")

dim = 1
target = int(sys.argv[2])

fails = 0
all = 0

root = sys.argv[1]
for p in os.listdir(root):
    img = cv2.imread(root + p, cv2.IMREAD_GRAYSCALE)
    img = np.reshape(img,[1,64,64,dim])
    c = loaded_model.predict_classes(img)
    print p, c[0]
    all += 1
    if target is not None:
        if c != target:
            fails += 1

print ("Failures", fails, "of", all)

