import tensorflow as tf
import numpy as np

def list_mean(l):
    l_np = np.asarray(l)
    return np.mean(l_np)

def check_dir(MYDIR):
    CHECK_FOLDER = os.path.isdir(MYDIR)
    if not CHECK_FOLDER:
        os.makedirs(MYDIR)
        print("Created Folder : ", MYDIR)
    else:
        print(MYDIR, "Folder already exists!")

def get_filenames(path, path_im, train=True):
    if train:
        files = np.load(path+'train_files.npy', allow_pickle=True)
        labels = np.load(path+'train_labels.npy', allow_pickle=True).astype(np.uint8)

        files_im = []
        for x in files:
            if x.split('.')[-1] == 'png':
               files_im.append(path_im+x)
    else:
        files = np.load(path+'val_files.npy', allow_pickle=True)
        labels = np.load(path+'val_labels.npy', allow_pickle=True).astype(np.uint8)

        files_im = []
        for x in files:
            if x.split('.')[-1] == 'png':
               files_im.append(path_im+x)

    return files_im, labels

def load_image(path):
    image_string = tf.read_file(path)
    # Don't use tf.image.decode_image, or the output shape will be undefined
    image = tf.image.decode_png(image_string, channels=3)
    # This will convert to float values in [0, 1]
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.expand_dims(image, 0)

    image_size = 224
    image = tf.image.resize_bilinear(image,
                                    [image_size, image_size],
                                    align_corners=False)

    image = tf.squeeze(image, [0])

    return image

def load(im_file, label):
    image = load_image(im_file)
    return image, label

def augment(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32) # Cast and normalize the$
    #image = tf.image.rot90(image, tf.random_uniform(shape=[], minval=0, maxval=4, d$

    #image = tf.image.random_flip_up_down(image)
    image = tf.image.random_flip_left_right(image)

    #image = tf.image.random_brightness(image, max_delta=0.4) # Random brightness

    random_angles = tf.random.uniform(shape=[] , minval = -np.pi/4, maxval = np.pi/4)
    image = tf.contrib.image.rotate(image, random_angles, interpolation='BILINEAR')

    random_trans1 = tf.random.uniform(shape=[] , minval = 0, maxval = 10)
    random_trans2 = tf.random.uniform(shape=[] , minval = -10, maxval = 0)
    image = tf.contrib.image.translate(image, [random_trans1, random_trans2])

    image = tf.clip_by_value(image, 0.0, 1.0)

    return image,label
