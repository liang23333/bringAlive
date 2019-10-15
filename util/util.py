import sys
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import math
if sys.version_info.major == 3:
    xrange = range


def im2uint8(x):
    if x.__class__ == tf.Tensor:
        return tf.cast(tf.clip_by_value(x, 0.0, 1.0) * 255.0, tf.uint8)
    else:
        t = np.clip(x, 0.0, 1.0) * 255.0
        return t.astype(np.uint8)









def blockLayer(x, channels, r, kernel_size=[3,3]):
    output = tf.layers.conv2d(x, channels, (3, 3), padding='same', dilation_rate=(r, r))
    return tf.nn.relu(output)

def resDenseBlock(x, out, channels=32, layers=4, kernel_size=[3,3], scale=0.1, name='rdb'):
    with tf.variable_scope(name):
        outputs = [x]
        rates = [1,1,1,1]
        for i in range(layers):
            output = blockLayer(tf.concat(outputs[:i],3) if i>=1 else x, channels, rates[i])
            outputs.append(output)

        output = tf.concat(outputs, 3)
        output = slim.conv2d(output, out, [3,3], activation_fn = None)
        output *= scale
        return x + output

        


def Bottleneck(x, out, name='bottleneck'):
    with tf.variable_scope(name):
        conv1 = slim.conv2d(x, out*2, [1,1], scope = 'conv1')
        conv2 = slim.conv2d(conv1, out, [3,3], scope = 'conv2')
        return conv2
