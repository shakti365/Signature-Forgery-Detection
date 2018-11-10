import os
import sys
import glob
from PIL import Image
import numpy as np
from scipy.misc import imresize
from collections import defaultdict
import itertools

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
positive_image_tuples = list()

for image_id in real_images_dict.keys():
    real = real_images_dict[image_id]
    forged = forged_images_dict[image_id]
    
    negative_image_tuples.extend(list(itertools.product(real, forged)))
    positive_image_tuples.extend(list(itertools.product(real, real)))
    
def process(image_path, size=(128, 256)):
    """returns processed images"""
    # Open image and convert to grayscale.
    image = Image.open(image_path)
    image = image.convert("L")
    
    image_array = np.array(image)
    
    # Resize image to 128, 256 using bilinear interpolation.
    image_array_processed = imresize(image_array, size=size, interp='bilinear')
    
    # Invert pixel values.
    image_array_processed = 1 - image_array_processed
    
    # Normalize by dividing pixel values with standard deviation.
    image_array_processed = image_array_processed / np.std(image_array_processed)
    return image_array_processed

# pre-process data
image_1 = []
image_2 = []
labels = []

for real, forged in negative_image_tuples:
    image_1.append(process(real))
    image_2.append(process(forged))
    labels.append(0)
    
for real1, real2 in positive_image_tuples:
    image_1.append(process(real1))
    image_2.append(process(real2))
    labels.append(1)
    
# Convert to numpy arrays
image_1_array = np.asarray(image_1)
image_2_array = np.asarray(image_2)
labels_array = np.array(labels)
labels_array = np.stack((labels_array, 1 - labels_array), axis=1)

# shuffle numpy arrays
idx = np.random.choice(range(len(image_1)), size=len(image_1), replace=False)

X_1 = image_1_array[idx]
X_2 = image_2_array[idx]
y = labels_array[idx]

# split data into train-valid-test set.
train_split = 0.8
valid_split = 0.9
train_offset = int(train_split * len(X_1))
valid_offset = int(valid_split * len(X_1))

X_1_train = X_1[:train_offset]
X_2_train = X_2[:train_offset]
y_train = y[:train_offset]

X_1_valid = X_1[train_offset:valid_offset]
X_2_valid = X_2[train_offset:valid_offset]
y_valid = y[train_offset:valid_offset]

X_1_test = X_1[valid_offset:]
X_2_test = X_2[valid_offset:]
y_test = y[valid_offset:]

destn_dir = '../../data/processed/'
np.savez(os.path.join(destn_dir, 'train.npz'), X_1_train=X_1_train, X_2_train=X_2_train, y_train=y_train)
np.savez(os.path.join(destn_dir, 'valid.npz'), X_1_valid=X_1_valid, X_2_valid=X_2_valid, y_valid=y_valid)
np.savez(os.path.join(destn_dir, 'test.npz'), X_1_test=X_1_test, X_2_test=X_2_test, y_test=y_test)