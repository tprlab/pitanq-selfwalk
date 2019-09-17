import sys, os
import tensorflow as tf
from tensorflow.python.platform import gfile
import cv2 as cv
import numpy as np



f = gfile.FastGFile("./ktf_model.pb", 'rb')
graph_def = tf.GraphDef()
graph_def.ParseFromString(f.read())
f.close()

sess = tf.Session()
sess.graph.as_default()
tf.import_graph_def(graph_def)



def handle_file(path):
    softmax_tensor = sess.graph.get_tensor_by_name('import/activation_4/Softmax:0')
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = np.reshape(img,[1,64,64,1])
    predictions = sess.run(softmax_tensor, {'import/conv2d_1_input:0': img})
    return np.argmax(predictions[0])

def check_dir(path, target = None):
    total = 0
    fails = 0
    has_target = target is not None
    for p in os.listdir(path):
        fpath = path + p
        c = handle_file(fpath)
        total += 1
        if has_target:
            if target != c:
                fails += 1
                print("Failed", fpath, c, "instead of", target)
        else:
            print(path, c[0])


    if has_target:
        print (path, "failures", fails, "of", total)


#p = handle_file("data/train/l/l1.jpg")
#print (p)

check_dir("data/train/l/", 0)
check_dir("data/train/r/", 1)
check_dir("data/train/s/", 2)






