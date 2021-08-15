import tensorflow as tf
import numpy as np
import math, os
from vgg16 import VGG16
import utils
from datetime import datetime

# # ============================================================================
# #                                   SETTINGS
# # ============================================================================

CLASSES = 3
selected_model = 'VGG-16'

print(100*'=','\n')
print('Selected model : {}'.format(selected_model))
print('\n')

# # -------------------------------- TRAINING ----------------------------------

batch_size = 1

template = '{}, Epoch {}, train_loss: {:.4f} - val_loss: {:.4f} - train_acc: {:.4f} - val_acc: {:.4f}'
AUTOTUNE = tf.data.experimental.AUTOTUNE

# # ------------------------------- OPTIMIZER ----------------------------------
learning_rate = 1e-5
# # -------------------------------- FOLDERS -----------------------------------
now = datetime.now()
timestamp = round(datetime.timestamp(now))

exp_name = ''
save_path = './tmp/'+exp_name+'/'
loss_path = './losses/'+exp_name+'/'

# # ============================================================================
# #                                   DATA
# # ============================================================================

path_npy = './files_npy/'
path_im = './data/'

# # ----------------------------- TF DATASET -----------------------------------

# # separar imagenes en folder y etiquetas en tf constant
X_test, y_test = utils.get_testfilenames(path=path_npy, path_im=path_im)

print(100*'=','\n')
print('Loading data ...')

test_len = len(X_test)
steps_test = (test_len // batch_size)

print('{} files for testing'.format(test_len))
print('\n')

test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_dataset = test_dataset.map(utils.load, num_parallel_calls=AUTOTUNE)
test_dataset = test_dataset.batch(batch_size)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

iterator = tf.data.Iterator.from_structure(test_dataset.output_types,
                                           test_dataset.output_shapes)

data_im, data_lbl = iterator.get_next()

# create the initialization operations
test_init_op = iterator.make_initializer(test_dataset)

# # ============================================================================
# #                                   MODEL
# # ============================================================================

print(100*'=','\n')
print('Training Settings \n')
print('Batch Size : {}'.format(batch_size))
print('Steps test : {}'.format(steps_test))
print('\n')

keep_prob = tf.placeholder(tf.float32)
net = VGG16(data_im, CLASSES, keep_prob)
net = tf.squeeze(net, axis=(1, 2))

net_softmax = tf.nn.softmax(net)

# # ============================================================================
# #                                 OPTIMIZATION
# # ============================================================================

# ----------------------------------- LOSS -------------------------------------

y = tf.one_hot(data_lbl, depth=CLASSES)   # one hot encoding
loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(logits=net, labels=y))

# --------------------------------- OPTIMIZER ----------------------------------

global_step = tf.train.create_global_step()
opt = tf.train.AdamOptimizer(learning_rate=learning_rate)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
train_opt = opt.minimize(loss, global_step=global_step)
train_opt = tf.group([train_opt, update_ops])

# ==============================================================================
#                              SAVER AND SUMMARY
# ==============================================================================

# --------------------------------- METRICS ------------------------------------

with tf.name_scope('train'):
   train_acc, train_acc_op = tf.metrics.accuracy(labels=tf.argmax(y, 1),
                                                 predictions=tf.argmax(net,1))
with tf.name_scope('valid'):
   val_acc, val_acc_op = tf.metrics.accuracy(labels=tf.argmax(y, 1),
                                                 predictions=tf.argmax(net,1))

# ==============================================================================
#                                   SESSION
# ==============================================================================

print(100*'=','\n')
print('Starting testing ... \n')

with tf.Session() as sess:

# ------------------------------ LOAD WEIGHTS ----------------------------------
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    print('Loading weights ... \n')
    saver.restore(sess,tf.train.latest_checkpoint(save_path))
    print('Weights successfully loaded\n')

    sess.run(test_init_op)

    try:
        output, y_labels = [],[]
        for step in range(steps_test):
            y_out, y_label = sess.run([net, y])

            output.append(y_out)
            y_labels.append(y_label)

            # # ------------------------------ GRAD-CAM -----------------------------------------------
            # imgs, lbls, out = sess.run([data_im, data_lbl, net])
            # cams = utils.gradcam(layer_name, loss, sess)
            # utils.save_cam(cams, imgs[0], lbls[0], out[0], output_path+'gradcams/', step)
            # # ---------------------------------------------------------------------------------------

        np.save(loss_path+'out_test.npy', output)
        np.save(loss_path+'labels_test.npy', y_labels)

    except tf.errors.OutOfRangeError:
        print('outOfRangeError, ', step)
        pass
