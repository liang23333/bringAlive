from __future__ import print_function
import os
import time
import random
import datetime
import scipy.misc
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from datetime import datetime
from util.util import *
from util.BasicConvLSTMCell import *
import cv2
from skimage.measure import compare_psnr

class DEBLUR(object):
    def __init__(self, args):
        self.args = args
        self.n_levels = 1
        self.scale = 0.5
        self.chns = 3 if self.args.model == 'color' else 1  # input / output channels

        # if args.phase == 'train':
        self.crop_size = 256
        self.train_dir = os.path.join('./checkpoints', args.model)
        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)



    def generator(self, inputs, reuse=False, scope='g_net'):
        n, h, w, c = inputs.get_shape().as_list()


        x_unwrap = []
        with tf.variable_scope(scope, reuse=reuse):
            with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                                activation_fn=tf.nn.relu, padding='SAME', normalizer_fn=None,
                                weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                                biases_initializer=tf.constant_initializer(0.0)):

                for i in xrange(self.n_levels):
                    
                    # encoder
                    
                    inp_1 = tf.space_to_depth(inputs,2,name='space_to_depth')
                    print('space_to_depths out :', inp_1)
                    
                    conv1 = slim.conv2d( inp_1, 64, [7,7], scope = 'enc_conv1')
                    print('encoder_conv1 out :', conv1)

                    rdb1_1 = resDenseBlock(conv1, 64, name='rdb1_1')
                    print('rdb1_1 out:', rdb1_1)

                    rdb1_2 = resDenseBlock(rdb1_1, 64, name='rdb1_2')
                    print('rdb1_2 out:', rdb1_2)

                    rdb1_3 = resDenseBlock(rdb1_2, 64, name='rdb1_3')
                    print('rdb1_3 out:', rdb1_3)

                    conv2 = slim.conv2d( rdb1_3, 96, [3,3], stride = 2, scope='enc_conv2')
                    print('encoder_conv2 out:', conv2)

                    rdb2_1 = resDenseBlock( conv2, 96, name='rdb2_1')
                    print('rdb2_1 out:', rdb2_1)

                    rdb2_2 = resDenseBlock(rdb2_1, 96, name='rdb2_2')
                    print('rdb2_2 out:', rdb2_2)

                    rdb2_3 = resDenseBlock(rdb2_2, 96, name='rdb2_3')
                    print('rdb2_3 out:', rdb2_3)

                    conv3 = slim.conv2d( rdb2_3, 128, [3,3], stride =2, scope='enc_conv3')
                    print('encoder_conv3 out:', conv3)

                    rdb3_1 = resDenseBlock( conv3, 128, name='rdb3_1')
                    print('rdb3_1 out:', rdb3_1)

                    rdb3_2 = resDenseBlock( rdb3_1, 128, name='rdb3_2')
                    print('rdb3_2 out:', rdb3_2)

                    rdb3_3 = resDenseBlock( rdb3_2, 128, name='rdb3_3')
                    print('rdb3_3 out:', rdb3_3)


                    # decoder

                    bottleneck1 = Bottleneck(rdb3_3, 128, name='bottleneck1')
                    print('bottleneck1 out: ', bottleneck1)

                    deconv1 = slim.conv2d_transpose(bottleneck1, 96, [4,4], stride=2, scope='deconv1')
                    print('deconv1 out:', deconv1)
                    
                    projection_conv1 = slim.conv2d(rdb2_3, 96, [3,3], scope='projection_conv1')
                    print('projection conv1 out: ', projection_conv1)

                    bottleneck2 = Bottleneck(deconv1+projection_conv1, 96, name='bottleneck2')
                    print('bottleneck2 out:', bottleneck2)

                    deconv2 = slim.conv2d_transpose(bottleneck2, 64, [4,4], stride=2, scope='deconv2')
                    print('deconv2 out:', deconv2)
                    
                    projection_conv2 = slim.conv2d(rdb1_3, 64, [3,3], scope='projection_conv2')
                    print('projection conv2 out:', projection_conv2)

                    bottleneck3 = Bottleneck(deconv2+projection_conv2, 64, name='bottleneck3')
                    print('bottleneck3 out:', bottleneck3)

                    deconv3 = slim.conv2d_transpose(bottleneck3, 32, [4,4], stride=2, scope='deconv3')
                    print('deconv3 out:', deconv3)

                    conv4 = slim.conv2d( tf.concat([deconv3,inputs],axis=-1), 16, [3,3], scope='conv4')
                    print('conv4 out:', conv4)

                    conv5 = slim.conv2d(conv4, 3, [3,3], activation_fn = None, scope='conv5')
                    print('conv5 out:', conv5)

                    x_unwrap.append(conv5)

            return x_unwrap


    def load(self, sess, checkpoint_dir, step=None):
        print(" [*] Reading checkpoints...")
        model_name = "deblur.model"
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

        if step is not None:
            ckpt_name = model_name + '-' + str(step)
            self.saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Reading intermediate checkpoints... Success")
            return str(step)
        elif ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            ckpt_iter = ckpt_name.split('-')[1]
            self.saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Reading updated checkpoints... Success")
            return ckpt_iter
        else:
            print(" [*] Reading checkpoints... ERROR")
            return False

    def test(self, height, width, input_path, output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        imgsName = sorted(os.listdir(input_path))

        H, W = height, width
        inp_chns = 3 if self.args.model == 'color' else 1
        self.batch_size = 1 if self.args.model == 'color' else 3
        inputs = tf.placeholder(shape=[self.batch_size, H, W, inp_chns], dtype=tf.float32)
        outputs = self.generator(inputs, reuse=False)

        sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

        self.saver = tf.train.Saver()
        self.load(sess, os.path.join(self.train_dir,'checkpoints'), step=1000)

        for imgName in imgsName:
            blur = scipy.misc.imread(os.path.join(input_path, imgName))
            h, w, c = blur.shape
            # make sure the width is larger than the height
            rot = False
            if h > w:
                blur = np.transpose(blur, [1, 0, 2])
                rot = True
            h = int(blur.shape[0])
            w = int(blur.shape[1])
            resize = False
            if h > H or w > W:
                scale = min(1.0 * H / h, 1.0 * W / w)
                new_h = int(h * scale)
                new_w = int(w * scale)
                blur = scipy.misc.imresize(blur, [new_h, new_w], 'bicubic')
                resize = True
                blurPad = np.pad(blur, ((0, H - new_h), (0, W - new_w), (0, 0)), 'edge')
            else:
                blurPad = np.pad(blur, ((0, H - h), (0, W - w), (0, 0)), 'edge')
            blurPad = np.expand_dims(blurPad, 0)
            if self.args.model != 'color':
                blurPad = np.transpose(blurPad, (3, 1, 2, 0))

            start = time.time()
            deblur = sess.run(outputs, feed_dict={inputs: blurPad / 255.0})
            duration = time.time() - start
            print('Saving results: %s ... %4.3fs' % (os.path.join(output_path, imgName), duration))
            res = deblur[-1]
            if self.args.model != 'color':
                res = np.transpose(res, (3, 1, 2, 0))
            res = im2uint8(res[0, :, :, :])
            # crop the image into original size
            if resize:
                res = res[:new_h, :new_w, :]
                res = scipy.misc.imresize(res, [h, w], 'bicubic')
            else:
                res = res[:h, :w, :]

            if rot:
                res = np.transpose(res, [1, 0, 2])
            scipy.misc.imsave(os.path.join(output_path, imgName), res)
