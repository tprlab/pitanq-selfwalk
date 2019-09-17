# pitanq-selfwalk
## Overview
This repository contains the code and data to train a neural network to decide in which direction (forward, left, right) a robot should go.
It was developed for <a href="https://pitanq.com">Raspberry Pi powered robot PiTanq</a> but can be applied to any similar vehicle.

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
