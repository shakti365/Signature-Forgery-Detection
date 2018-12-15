import os
import sys
import glob
from PIL import Image
import numpy as np
from scipy.misc import imresize
from collections import defaultdict
import itertools
from utils import process

real_images = glob.glob('../../data/interim/real/*.png')
forged_images = glob.glob('../../data/interim/forged/*.png')

def get_image_id(image_path):
    """returns image ID from the image path"""
    image_id = image_path.split('/')[-1].split('_')[0]
    return image_id

# Create a dictionary to store all images.
real_images_dict = defaultdict(list)
forged_images_dict = defaultdict(list)

# Iterate over real images and put them in dictionary values for same image_id key.
for real_image, forged_image in zip(real_images, forged_images):
    
    # add image to dictionary
    real_image_id = get_image_id(real_image)
    real_images_dict[real_image_id].append(real_image)
    
    forged_image_id = get_image_id(forged_image)
    forged_images_dict[forged_image_id].append(forged_image)
    
# create tuples of image for training
negative_image_tuples = list()
# positive_image_tuples = list()

for image_id in real_images_dict.keys():
    real = real_images_dict[image_id]
    forged = forged_images_dict[image_id]
    
    negative_image_tuples.extend(list(itertools.product(real, real, forged)))
#     positive_image_tuples.extend(list(itertools.product(real, real)))


# pre-process data
image_1 = []
image_2 = []
image_3 = []
labels = []

for anchor, positive, negative in negative_image_tuples:
    image_1.append(process(anchor))
    image_2.append(process(positive))
    image_3.append(process(negative))
    labels.append(0)
    
# for real1, real2 in positive_image_tuples:
#     image_1.append(process(real1))
#     image_2.append(process(real2))
#     labels.append(1)
    
# Convert to numpy arrays
image_1_array = np.asarray(image_1)
image_2_array = np.asarray(image_2)
image_3_array = np.asarray(image_3)
labels_array = np.array(labels)

# shuffle numpy arrays
# idx = np.random.choice(range(len(image_1)), size=len(image_1), replace=False)
#
# X_1 = image_1_array[idx]
# X_2 = image_2_array[idx]
# X_3 = image_3_array[idx]
# y = labels_array[idx]

# split data into train-valid-test set.
train_split = 0.8
valid_split = 0.9
train_offset = int(train_split * len(image_1_array))
valid_offset = int(valid_split * len(image_1_array))

X_1_train = image_1_array[:train_offset]
X_2_train = image_2_array[:train_offset]
X_3_train = image_3_array[:train_offset]
y_train = labels_array[:train_offset]

X_1_valid = image_1_array[train_offset:valid_offset]
X_2_valid = image_2_array[train_offset:valid_offset]
X_3_valid = image_3_array[train_offset:valid_offset]
y_valid = labels_array[train_offset:valid_offset]

X_1_test = image_1_array[valid_offset:]
X_2_test = image_2_array[valid_offset:]
X_3_test = image_3_array[valid_offset:]
y_test = labels_array[valid_offset:]

idx_train = np.random.choice(range(len(X_1_train)), size=len(X_1_train), replace=False)

X_1_train_idx = X_1_train[idx_train]
X_2_train_idx = X_2_train[idx_train]
X_3_train_idx = X_3_train[idx_train]
y_train_idx = y_train[idx_train]

print "shape of train", X_1_train_idx.shape, X_3_train_idx.shape,X_2_train_idx.shape, y_train_idx.shape
idx_val = np.random.choice(range(len(X_1_valid)), size=len(X_1_valid), replace=False)


X_1_valid_idx = X_1_valid[idx_val]
X_2_valid_idx = X_2_valid[idx_val]
X_3_valid_idx = X_3_valid[idx_val]
y_valid_idx = y_valid[idx_val]
print "shape of val", X_1_valid_idx.shape, X_3_valid_idx.shape,X_2_valid_idx.shape, y_valid_idx

idx_test = np.random.choice(range(len(X_1_test)), size=len(X_1_test), replace=False)

X_1_test_idx = X_1_valid[idx_test]
X_2_test_idx = X_2_valid[idx_test]
X_3_test_idx = X_3_valid[idx_test]
y_test_idx = y_valid[idx_test]

print "shape of test", X_1_test_idx.shape, X_3_test_idx.shape,X_2_test_idx.shape, y_test_idx

destn_dir = '../../data/processed/'
np.savez(os.path.join(destn_dir, 'train.npz'), X_1_train=X_1_train_idx, X_2_train=X_2_train_idx, X_3_train=X_3_train_idx, y_train=y_train_idx)
np.savez(os.path.join(destn_dir, 'valid.npz'), X_1_valid=X_1_valid_idx, X_2_valid=X_2_valid_idx, X_3_valid=X_3_valid_idx, y_valid=y_valid_idx)
np.savez(os.path.join(destn_dir, 'test.npz'), X_1_test=X_1_test_idx, X_2_test=X_2_test_idx, X_3_test=X_3_test_idx, y_test=y_test_idx)
