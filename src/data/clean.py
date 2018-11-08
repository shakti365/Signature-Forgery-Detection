import os
import glob
import shutil

# Minor correction in directory structure of extracted file.
# Move files from `dataset4/real1` to `dataset/real`
os.rename('../../data/raw/Dataset/dataset4/real1/', '../../data/raw/Dataset/dataset4/real/')

# Paths for all the dataset.
data_path = ['../../data/raw/Dataset/dataset1/', '../../data/raw/Dataset/dataset2/', 
             '../../data/raw/Dataset/dataset3/', '../../data/raw/Dataset/dataset4/']

# Get the forged and real images from all directories.
forged_images = [glob.glob(os.path.join(path, 'forge/*.png')) for path in data_path]
real_images = [glob.glob(os.path.join(path, 'real/*.png')) for path in data_path]

cleaned_files = []

# Loop over each directory of real and forged images.
for real, forged in zip(real_images, forged_images):
    
    while True:
        try:
            # Get the first file in real directory.
            file_name = real.pop(0)
        except IndexError:
            break

        # Get all its duplicates in real directory.
        real_file_names = [img for img in real if img[-7:] == file_name[-7:]]

        # Remove files from real data
        for file in real_file_names:
            real.remove(file)

        # Get the same images in forged directory.
        forged_file_names = [img for img in forged if img[-7:] == file_name[-7:]]

        # Remove files from forged data
        for file in forged_file_names:
            forged.remove(file)

        cleaned_files.append((real_file_names, forged_file_names))

def copy_file(src_file, destn_dir, idx, i):
    """copies files from source to destination"""
    shutil.copyfile(src_file, os.path.join(destn_dir, '{}_{}.png'.format(idx, i))) 

# Create interim directory with real and forged.
destn_dir = '../../data/interim'
real_destn_dir = os.path.join(destn_dir, 'real')
forged_destn_dir = os.path.join(destn_dir, 'forged')

# Create directories for real and forged images.
os.makedirs(real_destn_dir)
os.makedirs(forged_destn_dir)

for idx, (real, forged) in enumerate(cleaned_files):
    # Copy files to `interim/real`.
    [copy_file(real_file, real_destn_dir, idx, i) for i, real_file in enumerate(real)]
    # Copy files to `interim/forged`.
    [copy_file(forged_file, forged_destn_dir, idx, i) for i, forged_file in enumerate(forged)]
