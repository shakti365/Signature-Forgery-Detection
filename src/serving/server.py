import os
import sys

DIR = os.path.abspath(os.path.dirname(__file__))
SRC_DIR = os.path.join(DIR, os.pardir, os.pardir, 'src')
sys.path.append(SRC_DIR)

from flask import Flask, request, render_template, url_for
import numpy as np
from model.tf_model import SiameseCNN

from data import process_two

app = Flask(__name__)

config = dict()

config['data_path'] = '../../data/processed'
config['valid_batch_size'] = 16
config['train_batch_size'] = 16
config['seed'] = 42
config['learning_rate'] = 0.001
config['epochs'] = 1
config['export_dir'] = '../../data/models'
config['model_name'] = 'exp_1'
config['log_step'] = 1

siamese_model = SiameseCNN(config)


upload_folder = os.path.basename('static')
app.config['UPLOAD_FOLDER'] = upload_folder

@app.route("/")
def index():
    return "Server runnning!"


@app.route("/infer", methods=['POST', 'GET'])
def infer():

    if request.method == 'POST':

        input_1 = request.files['input1']
        f1 = os.path.join(app.config['UPLOAD_FOLDER'], input_1.filename)
        input_1.save(f1)

        input_2 = request.files['input2']
        f2 = os.path.join(app.config['UPLOAD_FOLDER'], input_2.filename)
        input_2.save(f2)

        x = process_two(f1, f2)

        distance = siamese_model.predict(x[:1], x[1:2])
        threshold = 1.0
        if distance > threshold:
            result='forged'
        else:
            result='real'

        output = "similarity score: {}, Image is {}".format(distance[0], result)
        
        return render_template('index.html', output=output, filename1=input_1.filename, filename2=input_2.filename)

    elif request.method == 'GET':
        return render_template('index.html', output='', filename1=False, filename2=False)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8005)

