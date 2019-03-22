import tensorflow as tf



######## CONV LAYER ########
def convLayer(input, filter_H, filter_W, strides, nbr_filters, padding = 'SAME', groups=1, name=None):

  input_channels = int(input.get_shape()[-1])

  with tf.variable_scope(name) as scope :
    weights = tf.get_variable('weights', shape=[filter_H, filter_W, input_channels/groups, nbr_filters])
    biases    = tf.get_variable('biases', shape=[nbr_filters])


  if(groups == 1):
    conv_output = tf.nn.conv2d(input, weights, strides, padding, name=name)
  else:
    input_groups = tf.split(input, groups, axis=3)
    weights_groups = tf.split(weights, groups, axis=3)
  
    #The output groups
    conv_output_groups = [tf.nn.conv2d(i, k, strides, padding, name=name) for i, k in zip(input_groups, weights_groups)]
  
    #Concatinate the output groups
    conv_output = tf.concat(conv_output_groups, axis=3)

  
  #Add the Bias
  bias_output = tf.nn.bias_add(conv_output, biases)

  #Reshape
  bias_output = tf.reshape(bias_output, shape=tf.shape(conv_output))

  #Apply the ReLu
  relu_output = tf.nn.relu(bias_output, name=scope.name)

  return relu_output

######## FULLY CONNECTED LAYER ########
def FcLayer(input, nbr_inputs, nbr_outputs, relu=True, name=None):
  
  with tf.variable_scope(name) as scope:
    weights = tf.get_variable('weights', shape=[nbr_inputs, nbr_outputs], trainable=True)
    biases = tf.get_variable('biases', shape=[nbr_outputs], trainable=True)
  
  output = tf.nn.xw_plus_b(input, weights, biases)
  
  if(relu):
    output = tf.nn.relu(output, name = scope.name)
  
  return output

######## MAX-POOLING LAYER ########
def maxPoolingLayer(input, filter_H, filter_W, strides, padding = 'SAME', name=None):
  
  return tf.nn.max_pool(input, [1, filter_H, filter_W, 1], strides, padding, name=name)

  
######## Local Response Normalization LAYER ########
def lrnLayer(input, depth_radius=5, bias=1, alpha=1, beta=0.5, name=None):
  return tf.nn.local_response_normalization(input, depth_radius, bias, alpha, beta, name=name)


######## Local Response Normalization LAYER ########
def dropoutLayer(input, keep_prob, name=None):
  """Create a dropout layer."""
  return tf.nn.dropout(input, keep_prob, name=name)
    
     
    