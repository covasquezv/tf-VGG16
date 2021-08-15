# Tensorflow 1.15 implementation of VGG-16.

Download Imagenet weights [here](https://drive.google.com/drive/folders/1kP66oCFU8XTsT2XhBPSZM0txEv6WmIzQ?usp=sharing).

Requirements:
- Python 3.7 
- Tensorflow 1.15

Usage:

- training:
```
python main.py
```

- predict:
```
python predict.py
```


Notes:

1) **./files_npy/** : folder contatining the following npy files

- train_files.npy
- trains_labels.npy
- val_files.npy
- val_labels.npy
- test_files.npy
- test_labels.npy

*_files.npy contains filenames of images randomly chosen for each set.

*_labels.npy contains integer labels for the images setted in *_files.npy (must be in the same order).


2) **./data/** contains the images.
3) **./tmp/** where the weights are saved during training.
4) **./losses/** where the loss and accuracy metrics are saved during training.
