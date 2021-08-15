import tensorflow as tf
import numpy as np
import math, os
from vgg16 import VGG16
import utils

# # ============================================================================
# #                                   SETTINGS
# # ============================================================================

CLASSES = 3
selected_model = 'VGG-16'
ckpt_imagenet = './vgg_16_ckpt/vgg_16.ckpt'

print(100*'=','\n')
print('Selected model : ResNet-{}'.format(selected_model))
print('Pretrained model (ImageNet) weights : {}'.format(ckpt_imagenet))
print('\n')

# # -------------------------------- TRAINING ----------------------------------

batch_size = 32

best_val = math.inf
#best_val = 0
best_val_epoch = 0
patience = 0
stop = 200

epochs = 5000

template = '{}, Epoch {}, train_loss: {:.4f} - val_loss: {:.4f} - train_acc: {:.4f} - val_acc: {:.4f}'
AUTOTUNE = tf.data.experimental.AUTOTUNE

repeat = 1

# # ------------------------------- OPTIMIZER ----------------------------------
learning_rate = 1e-2 #1e-4 #1e-3
# # -------------------------------- FOLDERS -----------------------------------
now = datetime.now()
timestamp = round(datetime.timestamp(now))

exp_name = ''
save_path = './tmp/'+exp_name+'/'
loss_path = './losses/'+exp_name+'/'

print(100*'=','\n')
print('Creating logs paths ...')

utils.check_dir(save_path)
utils.check_dir(loss_path)

print('\n')

# # ============================================================================
# #                                   DATA
# # ============================================================================

path_npy = './files_npy/'
path_im = './data/'

# # ----------------------------- TF DATASET -----------------------------------

X_train, y_train = utils.get_filenames(path=path_npy, path_im=path_im)
X_val, y_val = utils.get_filenames(path=path_npy, path_im=path_im, train=False)

print(100*'=','\n')
print('Loading data ...')

train_len = len(X_train)
val_len = len(X_val)
steps_train = math.ceil((train_len / batch_size)) #*repeat
#steps_train = (train_len//batch_size)*repeat
steps_val = math.ceil(val_len / batch_size)

print('{} files for training'.format(train_len))
print('{} files for validation'.format(val_len))
print('\n')

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=train_len)
train_dataset = train_dataset.repeat(repeat)
train_dataset = train_dataset.map(utils.load, num_parallel_calls=AUTOTUNE)
train_dataset = train_dataset.map(utils.augment, num_parallel_calls=AUTOTUNE)
train_dataset = train_dataset.batch(batch_size)
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_dataset = val_dataset.shuffle(buffer_size=val_len)
val_dataset = val_dataset.map(utils.load, num_parallel_calls=AUTOTUNE)
val_dataset = val_dataset.batch(batch_size)
val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)

iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                           train_dataset.output_shapes)

data_im, data_lbl = iterator.get_next()

# # create the initialization operations
train_init_op = iterator.make_initializer(train_dataset)
val_init_op = iterator.make_initializer(val_dataset)

# # ============================================================================
# #                                   MODEL
# # ============================================================================

print(100*'=','\n')
print('Training Settings \n')
print('Epochs : {}'.format(epochs))
print('Batch Size : {}'.format(batch_size))
print('Steps train : {}'.format(steps_train))
print('Steps validation : {}'.format(steps_val))
print('Early Stopping patience : {}'.format(stop))
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

#decayed_lr = tf.train.exponential_decay(learning_rate,
#                                        global_step, 10000,
#                                        0.95, staircase=True)

opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
#opt = tf.train.AdamOptimizer(decayed_lr)

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

# --------------------------------- VARS ---------------------------------------
## Get layers to restore, except last layer of logits
vars = []
for variable in tf.all_variables():
    stat_1 = 'logits' not in (variable.name)
    stat_2 = 'Adam' not in (variable.name)
    stat_3 = 'power' not in (variable.name)
    stat_4 = 'global_step' not in (variable.name)

    if  stat_1 and stat_2 and stat_3 and stat_4:
        vars.append(variable)

saver_restore = tf.train.Saver(var_list=vars)
saver = tf.train.Saver()

# ==============================================================================
#                                   SESSION
# ==============================================================================

print(100*'=','\n')
print('Starting training ... \n')

with tf.Session() as sess:

# ------------------------------ LOAD WEIGHTS ----------------------------------
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    print('Loading pretrained weights ... \n')
    saver_restore.restore(sess, ckpt_imagenet)

    #print('Weights successfully loaded\n')

# ------------------------------ TRAINING --------------------------------------
    train_loss_, val_loss_ = [], []
    train_acc_, val_acc_ = [], []

    for epoch in range(epochs):
        pred_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        tl, vl = [], []

        sess.run(train_init_op)
        try:
            for step in range (steps_train):
                _, train_loss, acc_tr, acc_op_tr = sess.run([train_opt, loss, train_acc, train_acc_op])
                tl.append(train_loss)

            mean_train = utils.list_mean(tl)
            train_loss_.append(mean_train)

        except tf.errors.OutOfRangeError:
            print(epoch, step)
            pass

        if (epoch+1) % 1 == 0:
            sess.run(val_init_op)
            try:
                for step in range (steps_val):
                    val_loss, acc_v, acc_op_v = sess.run([loss, val_acc, val_acc_op])
                    vl.append(val_loss)
                mean_val = utils.list_mean(vl)
                val_loss_.append(mean_val)

            except tf.errors.OutOfRangeError:
                pass

        print(template.format(pred_time, epoch, mean_train, mean_val, acc_op_tr, acc_op_v))

        train_acc_.append(acc_op_tr)
        val_acc_.append(acc_op_v)

        # early stopping
        if mean_val < best_val:
        #if acc_op_v > best_val:
            print('Saving on epoch {0}'.format(epoch))
            best_val = mean_val
            #best_val = acc_op_v
            patience = 0

            best_val_epoch = epoch
            saver.save(sess, save_path+'best_model_epoch_{}'.format(best_val_epoch), global_step=global_step)

            np.save(loss_path+'train_loss.npy', train_loss_)
            np.save(loss_path+'val_loss.npy', val_loss_)
            np.save(loss_path+'train_acc.npy', train_acc_)
            np.save(loss_path+'val_acc.npy', val_acc_)
        else:
            patience += 1

        if patience == stop:
            # training stops if there's no improvement after certain epochs
            np.save(loss_path+'train_loss.npy', train_loss_)
            np.save(loss_path+'val_loss.npy', val_loss_)
            np.save(loss_path+'train_acc.npy', train_acc_)
            np.save(loss_path+'val_acc.npy', val_acc_)
            print('Early stopping at epoch: {}'.format(best_val_epoch))
            break

print('Training ended . \n')
np.save(loss_path+'train_loss.npy', train_loss_)
np.save(loss_path+'val_loss.npy', val_loss_)
np.save(loss_path+'train_acc.npy', train_acc_)
np.save(loss_path+'val_acc.npy', val_acc_)
