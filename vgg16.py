import tensorflow as tf
from layers import conv_layer, fc_layer, maxpool_layer, dropout_layer

def VGG16(x, num_classes, keep_prob):
	conv1_1 = conv_layer(x, 64, 'conv1_1')
	conv1_2 = conv_layer(conv1_1, 64, 'conv1_2')
	pool1 = max_pool_layer(conv1_2, 'pool1')

	conv2_1 = conv_layer(pool1, 128, 'conv2_1')
	conv2_2 = conv_layer(conv2_1, 128, 'conv2_2')
	pool2 = max_pool_layer(conv2_2, 'pool2')

	conv3_1 = conv_layer(pool2, 256, 'conv3_1')
	conv3_2 = conv_layer(conv3_1, 256, 'conv3_2')
	conv3_3 = conv_layer(conv3_2, 256, 'conv3_3')
	pool3 = max_pool_layer(conv3_3, 'pool3')

	conv4_1 = conv_layer(pool3, 512, 'conv4_1')
	conv4_2 = conv_layer(conv4_1, 512, 'conv4_2')
	conv4_3 = conv_layer(conv4_2, 512, 'conv4_3')
	pool4 = max_pool_layer(conv4_3, 'pool4')

	conv5_1 = conv_layer(pool4, 512, 'conv5_1')
	conv5_2 = conv_layer(conv5_1, 512, 'conv5_2')
	conv5_3 = conv_layer(conv5_2, 512, 'conv5_3')
	pool5 = max_pool_layer(conv5_3, 'pool5')

	flattened = tf.reshape(pool5, [-1, 1 * 1 * 512])
	fc6 = fc_layer(flattened, 1 * 1 * 512, 4096, name='fc6')
	dropout6 = dropout(fc6, keep_prob)

	fc7 = fc_layer(dropout6, 4096, 4096, name='fc7')
	dropout7 = dropout(fc7, keep_prob)

	fc8 = fc_layer(dropout7, 4096, num_classes, relu=False, name='fc8')

	return fc8
