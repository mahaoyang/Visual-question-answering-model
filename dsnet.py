from keras import layers
from keras import models
from keras import backend


def dense_block(x, blocks):
    """A dense block.

    # Arguments
        x: input tensor.
        blocks: integer, the number of building blocks.
        name: string, block label.

    # Returns
        output tensor for the block.
    """
    for i in range(blocks):
        x = conv_block(x, 32)
    return x


def transition_block(x, reduction):
    """A transition block.

    # Arguments
        x: input tensor.
        reduction: float, compression rate at transition layers.
        name: string, block label.

    # Returns
        output tensor for the block.
    """
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(int(backend.int_shape(x)[bn_axis] * reduction), 1, use_bias=False)(x)
    x = layers.AveragePooling2D(2, strides=2)(x)
    return x


def conv_block(x, growth_rate):
    """A building block for a dense block.

    # Arguments
        x: input tensor.
        growth_rate: float, growth rate at dense layers.
        name: string, block label.

    # Returns
        Output tensor for the block.
    """
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
    x1 = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x)
    x1 = layers.Activation('relu')(x1)
    x1 = layers.Conv2D(4 * growth_rate, 1, use_bias=False)(x1)
    x1 = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x1)
    x1 = layers.Activation('relu')(x1)
    x1 = layers.Conv2D(growth_rate, 3, padding='same', use_bias=False)(x1)
    x = layers.Concatenate(axis=bn_axis)([x, x1])
    return x


def dense_net(blocks):
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
    x = models.Sequential()
    x.add(layers.Conv2D(64, 7, strides=2, use_bias=False))
    x.add(layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5))
    x.add(layers.Activation('relu'))
    x.add(layers.ZeroPadding2D(padding=((1, 1), (1, 1))))
    x.add(layers.MaxPooling2D(3, strides=2))
    x.add(dense_block(x, blocks[0]))
    x.add(transition_block(x, 0.5))
    x.add(dense_block(x, blocks[1]))
    x.add(transition_block(x, 0.5))
    x.add(dense_block(x, blocks[2]))
    x.add(transition_block(x, 0.5))
    x.add(dense_block(x, blocks[3]))
    x.add(layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5)(x))
    x.add(layers.GlobalMaxPooling2D()(x))
    return x
