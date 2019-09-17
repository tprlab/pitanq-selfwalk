# pitanq-selfwalk
## Overview
This repository contains the code and data to train a neural network to decide in which direction (forward, left, right) a robot should go.
It was developed for <a href="https://pitanq.com">Raspberry Pi powered robot PiTanq</a> but can be applied to any similar vehicle.

### Repo content
- data - Data used to train the model
- models - The trained model in Keras, Tensorlow and OpenVino formats
- train.py - Trains a Keras network using data from a dataset in data folder
- check_nn.py - Classifies a single image or folder via the trained model
- keras2tf.py - Converts the model from original Keras format to Tensorflow
- check_tf.py - Checks the TF conversion via classification all the data against of the converted model


## Data
The neural network is a multi-label classifier trained with 3 classes of pictures: forward, left and right.
The data was received by road recognition on photos, then the photos were reduced and their centers 64x64 were taken as input for this network.
A zip-archive with the initial pics is in this repo.

### Forward
The forward pictures look like that:
![](https://pitanq.com/img/nn/straight.png)
The forward feature is the angle of the right edge is around 45 grades.
That is how the picture normally looks if the robot driving along the right edge of the road.

### Left
![](https://pitanq.com/img/nn/left.png)
When the robot makes too much right then the right edge started to lean to the left. This is the key feature of the left pictures. Sometimes they can look similar like forward pictures.

### Right
![](https://pitanq.com/img/nn/right.png)
When the robot makes too much left the right edge is disappeared from the photo. So if there is no explicit edge - it is time to go right.

## Network implementation
I used Keras backed by Tensorflow to train the network.
Raspberry Pi is able to use this network since Google released TF 1.9 for Raspbian Stretch 9.
Just make sure the current available versions of TF for Raspberry and TF for Keras are compatible.
There is no requirement to have Keras on Raspberry but TF itself contains an adapter to read the network graph.

### Code

There was used a simpliest architecture for multi-label classification.
Turned out it works pretty good.

```
def createModel(input_shape, cls_n ):
    model = Sequential()

    activation = "relu"

    model.add(Conv2D(20, 5, padding="same", input_shape=input_shape))
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # define the second set of CONV => ACTIVATION => POOL layers
    model.add(Conv2D(50, 5, padding="same"))
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # define the first FC => ACTIVATION layers
    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation(activation))

    # define the second FC layer
    model.add(Dense(cls_n))

    opt = SGD(lr=0.01)
    # lastly, define the soft-max classifier
    model.add(Activation("softmax"))

    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model
```

## Tensorflow conversion

I needed to convert the model to TF format.
The code is in keras2tf.py

```
import tensorflow as tf
from tensorflow.python.framework.graph_util import convert_variables_to_constants

from keras import backend as K
from keras.models import load_model
from keras.models import model_from_json


def load_keras_model(json_file, model_file):
    jf = open(json_file, 'r')
    loaded_model_json = jf.read()
    jf.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(model_file)
    return loaded_model


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph



model = load_keras_model('./model.json', './model.h5')
frozen_graph = freeze_session(K.get_session(),
                              output_names=[out.op.name for out in model.outputs])

tf.train.write_graph(frozen_graph, ".", "ktf_model.pb", as_text=False)
```

The converted model is in this repo: model/ktf_model.pb

## OpenVino

Then I converted TF model into OpenVino format to run with Intel NCS.

```
python mo_tf.py --input_model "model/ktf_model.pb" --log_level=DEBUG -b1 --data_type FP16
````

The OpenVino model consists of 2 files: model/ktf_model.xml and model/ktf_model.bin.
