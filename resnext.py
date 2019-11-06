"""
@author : carmel wenga

This script describes the resnext50 class and its differents functions.

the main function of that class is the `Build()` function.

The other functions (split(), transform(), transition(), downsampling() and identity_block()) are usefull in the Build function to instantiate the model.

For more details on how each function works, please visit https://github.com/carmel-nzhinusoft/implement-ResNeXt-with-keras/blob/master/resnext50-with-keras.ipynb

Note that resnext50.Build() function here correspond to the ResNeXt50() function in that documentation.
"""

from keras.layers import Activation
from keras.layers import Add
from keras.layers import AveragePooling2D
from keras.layers import BatchNormalization
from keras.layers import Concatenate
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import GlobalMaxPooling2D
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import MaxPooling2D
from keras.layers import ZeroPadding2D
from keras.models import Model
from keras.initializers import glorot_uniform

class ResNeXt50:
    
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
    
    def split(self, inputs, cardinality):
        inputs_channels = inputs.shape[3]
        group_size = inputs_channels // cardinality    
        groups = list()
        for number in range(1, cardinality+1):
            begin = int((number-1)*group_size)
            end = int(number*group_size)
            block = Lambda(lambda x:x[:,:,:,begin:end])(inputs)
            groups.append(block)
        return groups
    
    def transform(self, groups, filters, strides, stage, block):
        f1, f2 = filters    
        conv_name = "conv2d-{stage}{block}-branch".format(stage=str(stage), block=str(block))
        bn_name = "batchnorm-{stage}{block}-branch".format(stage=str(stage), block=str(block))

        transformed_tensor = list()
        i = 1

        for inputs in groups:
            # first conv of the transformation phase
            x = Conv2D(filters=f1, kernel_size=(1,1), strides=strides, padding="valid", 
                       name=conv_name+'1a_split'+str(i), kernel_initializer=glorot_uniform(seed=0))(inputs)
            x = BatchNormalization(axis=3, name=bn_name+'1a_split'+str(i))(x)
            x = Activation('relu')(x)

            # second conv of the transformation phase
            x = Conv2D(filters=f2, kernel_size=(3,3), strides=(1,1), padding="same", 
                       name=conv_name+'1b_split'+str(i), kernel_initializer=glorot_uniform(seed=0))(x)
            x = BatchNormalization(axis=3, name=bn_name+'1b_split'+str(i))(x)
            x = Activation('relu')(x)

            # Add x to transformed tensor list
            transformed_tensor.append(x)
            i+=1

        # Concatenate all tensor from each group
        x = Concatenate(name='concat'+str(stage)+''+block)(transformed_tensor)

        return x
    
    def transition(self, inputs, filters, stage, block):
        x = Conv2D(filters=filters, kernel_size=(1,1), strides=(1,1), padding="valid", 
                       name='conv2d-trans'+str(stage)+''+block, kernel_initializer=glorot_uniform(seed=0))(inputs)
        x = BatchNormalization(axis=3, name='batchnorm-trans'+str(stage)+''+block)(x)
        x = Activation('relu')(x)

        return x
    
    def identity_block(self, inputs, filters, cardinality, stage, block, strides=(1,1)):
    
        conv_name = "conv2d-{stage}{block}-branch".format(stage=str(stage),block=str(block))
        bn_name = "batchnorm-{stage}{block}-branch".format(stage=str(stage),block=str(block))

        #save the input tensor value
        x_shortcut = inputs
        x = inputs

        f1, f2, f3 = filters

        # divide input channels into groups. The number of groups is define by cardinality param
        groups = self.split(inputs=x, cardinality=cardinality)

        # transform each group by doing a set of convolutions and concat the results
        f1 = f1 // cardinality
        f2 = f2 // cardinality
        x = self.transform(groups=groups, filters=(f1, f2), strides=strides, stage=stage, block=block)

        # make a transition by doing 1x1 conv
        x = self.transition(inputs=x, filters=f3, stage=stage, block=block)

        # Last step of the identity block, shortcut concatenation
        x = Add()([x,x_shortcut])
        x = Activation('relu')(x)

        return x
    
    def downsampling(self, inputs, filters, cardinality, strides, stage, block):
    
        conv_name = "conv2d-{stage}{block}-branch".format(stage=str(stage), block=str(block))
        bn_name = "batchnorm-{stage}{block}-branch".format(stage=str(stage), block=str(block))

        # Retrieve filters for each layer
        f1, f2, f3 = filters

        # save the input tensor value
        x_shortcut = inputs
        x = inputs

        # divide input channels into groups. The number of groups is define by cardinality param
        groups = self.split(inputs=x, cardinality=cardinality)

        # transform each group by doing a set of convolutions and concat the results
        f1 = f1 // cardinality
        f2 = f2 // cardinality
        x = self.transform(groups=groups, filters=(f1, f2), strides=strides, stage=stage, block=block)

        # make a transition by doing 1x1 conv
        x = self.transition(inputs=x, filters=f3, stage=stage, block=block)

        # Projection Shortcut to match dimensions 
        x_shortcut = Conv2D(filters=f3, kernel_size=(1,1), strides=strides, padding="valid", 
                   name='{base}2'.format(base=conv_name), kernel_initializer=glorot_uniform(seed=0))(x_shortcut)
        x_shortcut = BatchNormalization(axis=3, name='{base}2'.format(base=bn_name))(x_shortcut)

        # Add x and x_shortcut
        x = Add()([x,x_shortcut])
        x = Activation('relu')(x)

        return x
    
    def build(self):
    
        # Transform input to a tensor of shape input_shape 
        x_input = Input(self.input_shape)

        # Add zero padding
        x = ZeroPadding2D((3,3))(x_input)

        # Initial Stage. Let's say stage 1
        x = Conv2D(filters=64, kernel_size=(7,7), strides=(2,2), 
                   name='conv2d_1', kernel_initializer=glorot_uniform(seed=0))(x)
        x = BatchNormalization(axis=3, name='batchnorm_1')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((3,3), strides=(2,2))(x)

        # Stage 2
        x = self.downsampling(inputs=x, filters=(128,128,256), cardinality=4, strides=(2,2), stage=2, block="a")
        x = self.identity_block(inputs=x, filters=(128,128,256), cardinality=4, stage=2, block="b")
        x = self.identity_block(inputs=x, filters=(128,128,256), cardinality=4, stage=2, block="c")


        # Stage 3
        x = self.downsampling(inputs=x, filters=(256,256,512), cardinality=4, strides=(2,2), stage=3, block="a")
        x = self.identity_block(inputs=x, filters=(256,256,512), cardinality=4, stage=3, block="b")
        x = self.identity_block(inputs=x, filters=(256,256,512), cardinality=4, stage=3, block="c")
        x = self.identity_block(inputs=x, filters=(256,256,512), cardinality=4, stage=3, block="d")


        # Stage 4
        x = self.downsampling(inputs=x, filters=(512,512,1024), cardinality=4, strides=(2,2), stage=4, block="a")
        x = self.identity_block(inputs=x, filters=(512,512,1024), cardinality=4, stage=4, block="b")
        x = self.identity_block(inputs=x, filters=(512,512,1024), cardinality=4, stage=4, block="c")
        x = self.identity_block(inputs=x, filters=(512,512,1024), cardinality=4, stage=4, block="d")
        x = self.identity_block(inputs=x, filters=(512,512,1024), cardinality=4, stage=4, block="e")
        x = self.identity_block(inputs=x, filters=(512,512,1024), cardinality=4, stage=4, block="f")


        # Stage 5
        x = self.downsampling(inputs=x, filters=(1024,1024,2048), cardinality=4, strides=(2,2), stage=5, block="a")
        x = self.identity_block(inputs=x, filters=(1024,1024,2048), cardinality=4, stage=5, block="b")
        x = self.identity_block(inputs=x, filters=(1024,1024,2048), cardinality=4, stage=5, block="c")


        # Average pooling
        x = AveragePooling2D(pool_size=(2,2), padding="same")(x)

        # Output layer
        x = Flatten()(x)
        x = Dense(self.num_classes, activation="softmax", kernel_initializer=glorot_uniform(seed=0), 
                  name="fc{cls}".format(cls=str(self.num_classes)))(x)

        # Create the model
        model = Model(inputs=x_input, outputs=x, name="resnet50")

        return model