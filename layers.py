import tensorflow as tf

def conv_layer(input, filters, name, size=3, stride=1 , padding='same'):
    with tf.variable_scope(name):
        kernel_init = tf.contrib.layers.xavier_initializer(seed=0)
        conv = tf.layers.conv2d(inputs=input,
                                filters=filters,
                                kernel_size=size,
                                strides=(stride,stride),
                                padding='same',
                                kernel_initializer=kernel_init,
                                name='conv')
        conv = tf.nn.relu(conv, name='relu')

    return conv

def fc_layer(input, input_size, output_size, name, relu=True):
	with tf.variable_scope(name):
        init_w = tf.random_normal_initializer(mean=0.0, stddev=0.01)
        init_b = tf.constant_initializer(0.0)

		W = tf.get_variable('weights',
                            shape=[input_size, output_size],
			                initializer=init_w)
		b = tf.get_variable('biases',
                            shape=[output_size],
                            initializer=init_b)

		fc = tf.nn.bias_add(tf.matmul(input, W), b, name='fc')

        if relu:
			fc = tf.nn.relu(fc, name='relu')

    return fc

def maxpool_layer(input, name, size=2, stride=2, padding='VALID'):
    with tf.variable_scope(name):
        ksize = [1, size, size, 1]
        strides = [1, stride, stride, 1]

        max_pooling = tf.nn.max_pool(input,
                                    ksize=ksize,
                                    strides=strides,
                                    padding=padding,
                                    name='max_pool')
    return max_pooling

def dropout_layer(x, keep_prob):
    return tf.nn.dropout(x, keep_prob = keep_prob)
