from PIL import Image
import numpy as np
from scipy.misc import imresize


def process(image_path, size=(155, 220)):
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

    # Expand dimension to (155, 220, 1)
    image_array_processed = np.expand_dims(image_array_processed, axis=2)

    return image_array_processed


def process_two(image_path1, image_path2):
    x1 = process(image_path1)
    x2 = process(image_path2)

    x = np.stack([x1, x2], axis=0)
    return x

