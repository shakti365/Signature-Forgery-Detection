import os, sys
DIR = os.path.abspath(os.path.dirname(__file__))
SRC_DIR = os.path.join(DIR, os.pardir, os.pardir, 'src')
sys.path.append(SRC_DIR)

from data import process_two
from model import SiameseCNN

import glob
from collections import defaultdict
import pandas as pd

config = dict()

config['data_path'] = '../../data/processed'
config['valid_batch_size'] = 16
config['train_batch_size'] = 16
config['seed'] = 42
config['learning_rate'] = 0.0001
config['epochs'] = 40
config['export_dir'] = '../../data/models'
config['model_name'] = 'exp_2'
config['log_step'] = 1

siamese_model = SiameseCNN(config)

def infer(image_path1, image_path2):
    x = process_two(image_path1, image_path2)

    x1 = x[:1]
    x2 = x[1:2]

    distance = siamese_model.predict(x1, x2)

    return distance


def get_image_id(image_path):
    """returns image ID from the image path"""
    image_id = image_path.split('/')[-1].split('_')[0]
    return image_id

def dir_infer(dir_path_real, dir_path_test):
    real_images = glob.glob(os.path.join(dir_path_real, '*.jpg'))
    test_images = glob.glob(os.path.join(dir_path_test, '*.jpg'))

    print (real_images, test_images)

    real_images_dict = defaultdict(list)

    for image in real_images:
        image_id = get_image_id(image)
        real_images_dict[image_id].append(image)

    predictions = []
    file_names = []
    for test_image in test_images:
        preds = []
        test_image_id = get_image_id(test_image)
        for real_image in real_images_dict[test_image_id]:
            dist = infer(real_image, test_image)
            preds.append(dist)

        pred = sum(preds)/ len(preds) 
        if pred > 0.5: 
            predictions.append('Forgery')
        else:
            predictions.append('Original')
        file_names.append(test_image.split('/')[-1])
    predictions_df = pd.DataFrame({'Filename': file_names, 'Decision': predictions})
    
    return predictions_df



if __name__=='__main__':
    path1 = '../../data/external/Registration-signature-data/real'
    path2 = '../../data/external/Registration-signature-data/test'
    #output = infer(path1, path2)
    #print (output)

    output = dir_infer(dir_path_real=path1, dir_path_test=path2)
    print (output)
