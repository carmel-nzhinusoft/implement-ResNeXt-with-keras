"""
@author : carmel wenga
"""
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.utils import to_categorical
from keras.datasets import cifar10
from keras.optimizers import Adam
from resnext import ResNeXt50

import numpy as np

# hyperparameter
batch_size = 32
epochs = 200
input_shape = (32, 32, 3)
num_classes = 10

def load_data():
    """
    Load and preprocess cifar10 data 
    """
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # Normalize data.
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # subtract pixel mean
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    print('y_train shape:', y_train.shape)

    # Convert class vectors to binary class matrices.
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    
    return (x_train, x_test), (y_train, y_test)

def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

# load cifar10 data
(x_train, x_test), (y_train, y_test) = load_data()

# building de model
model = ResNeXt50(input_shape=input_shape, num_classes=num_classes)
resnext50 = model.build()

# defining learning rate scheduler for callbacks
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patiente=5, min_lr=0.5e-6)
lr_scheduler = LearningRateScheduler(lr_schedule)

callbacks = [lr_reducer, lr_scheduler]

# compiling de model
resnext50.compile(optimizer=Adam(lr=lr_schedule(0)), loss='categorical_crossentropy', metrics=['accuracy'])

# fit the network
resnext50.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), callbacks=callbacks)

# saving the model
model_name = "resnext50_cifar10.h5"
resnext50.save(model_name)