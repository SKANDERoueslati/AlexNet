import tensorflow as tf
import numpy as np

import AlexNetLayers  as layers





class AlexNet:
  
  def __init__(self, x, nbr_classes, keep_prob):
    self.X = x
    self.nbr_classes = nbr_classes
    self.keep_prob = keep_prob
  
  def construct(self):
    
    # CONV LAYER 1
    conv1 = layers.convLayer(self.X, 11, 11, [1, 4, 4, 1], 96, padding='VALID', groups=1, name='conv1')
    norm1 = layers.lrnLayer(conv1, 2, 1, 2e-05, 0.75, name='NORM1')
    pool1 = layers.maxPoolingLayer(norm1, 3, 3, [1, 2, 2, 1], padding='VALID', name='POOL1')
    
    # CONV LAYER 2
    conv2 = layers.convLayer(pool1, 5, 5, [1, 1, 1, 1], 256, padding='SAME', groups=2, name='conv2')
    norm2 = layers.lrnLayer(conv2, 2, 1, 2e-05, 0.75, name='NORM2')
    pool2 = layers.maxPoolingLayer(norm2, 3, 3, [1, 2, 2, 1], padding='VALID', name='POOL2')
    
    # CONV LAYER 3
    conv3 = layers.convLayer(pool2, 3, 3, [1, 1, 1, 1], 384, padding='SAME', groups=1, name='conv3')
    
    # CONV LAYER 4
    conv4 = layers.convLayer(conv3, 3, 3, [1, 1, 1, 1], 384, padding='SAME', groups=2, name='conv4')
    
    # CONV LAYER 5
    conv5 = layers.convLayer(conv4, 3, 3, [1, 1, 1, 1], 256, padding='SAME', groups=2, name='conv5')
    pool5 = layers.maxPoolingLayer(conv5, 3, 3, [1, 2, 2, 1], padding='VALID', name='POOL5')
    
    # FC LAYER 6
    input6_size = int(np.prod(pool5.get_shape()[1:]))
    flat6 = tf.reshape(pool5, [-1, input6_size])
    fc6 = layers.FcLayer(flat6, input6_size, 4096, relu=True, name='fc6')
    dropOut6 = layers.dropoutLayer(fc6, self.keep_prob)
    
    # FC LAYER 7
    fc7 = layers.FcLayer(dropOut6, 4096, 4096, relu=True, name='fc7')
    dropOut7 = layers.dropoutLayer(fc7, self.keep_prob)
    
    # FC LAYER 8
    fc8 = layers.FcLayer(dropOut7, 4096, self.nbr_classes, relu=False, name='fc8')
    self.cnn_output = tf.nn.softmax(fc8)

  def loadTrainedParams(self, session, path):
        # Load the weights into memory
        weights_dict = np.load(open("bvlc_alexnet.npy", "rb"), encoding="latin1").item()#np.load(path, encoding='bytes').item()

        # Loop over all layer names stored in the weights dict
        for op_name in weights_dict:

            # Check if layer should be trained from scratch
            if True: #op_name not in self.SKIP_LAYER:

                with tf.variable_scope(op_name, reuse=True):

                    # Assign weights/biases to their corresponding tf variable
                    for data in weights_dict[op_name]:

                        # Biases
                        if len(data.shape) == 1:
                            var = tf.get_variable('biases', trainable=False)
                            session.run(var.assign(data))

                        # Weights
                        else:
                            var = tf.get_variable('weights', trainable=False)
                            session.run(var.assign(data))
                            


                                                   