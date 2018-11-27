import functools
import os, sys
DIR = os.path.abspath(os.path.dirname(__file__))
SRC_DIR = os.path.join(DIR, os.pardir, os.pardir, 'src')
sys.path.append(SRC_DIR)

from data import process_two

import requests
import json
from flask import (
    Blueprint, flash, g, redirect, render_template, request, Flask, url_for
)

from db import get_db

bp = Blueprint('signature', __name__, url_prefix='/signature')

UPLOAD_PATH='uploads'
upload_folder = os.path.abspath('serving/static')

@bp.route('/register', methods=('GET', 'POST'))
def register():
    if request.method == 'POST':
        username = request.form['username']
        image = request.files['input1']
        db = get_db()
        error = None

        if not username:
            error = 'Username is required.'
        elif db.execute(
            'SELECT id FROM user WHERE username = ?', (username,)
        ).fetchone() is not None:
            error = 'User {} is already registered.'.format(username)

        if error is None:

            # Set status of signature as 'real'
            status = 'real'

            # Create a signature path
            signature_path = os.path.join(UPLOAD_PATH, image.filename)

            # Save signature to the path
            image.save(os.path.join(upload_folder, signature_path))

            # Create user in the DB
            db.execute('INSERT INTO user (username) VALUES (?)', (username,))

            # Save signature in the DB
            db.execute(
                'INSERT INTO signature (user_id, status, signature_path) VALUES (?, ?, ?)',
                (username, status, signature_path)
            )
            db.commit()
            return redirect(url_for('signature.verify'))

        print (error)

    return render_template('register.html')

@bp.route('/verify', methods=('GET', 'POST'))
def verify():
    if request.method == 'POST':
        username = request.form['username']
        test_image = request.files['input1']
        db = get_db()
        error = None

        # Get user_id and username
        user_id, username = db.execute(
            'SELECT id, username FROM user WHERE username = ?', (username, )
        ).fetchone()

        if user_id is None:
            error = 'User not present'
        
        if error is None:

            # Create a signature path
            test_signature_path = os.path.join(UPLOAD_PATH, test_image.filename)

            # Save signature to the path
            test_image.save(os.path.join(upload_folder, test_signature_path))

            # Fetch real signature path for the user_id
            real_signature_path, = db.execute(
                "SELECT signature_path FROM signature WHERE user_id = ? AND status = 'real'", (username, )
            ).fetchone()

            x = process_two(os.path.join(upload_folder, real_signature_path), 
                            os.path.join(upload_folder, test_signature_path))

            x1 = x[0].tolist()
            x2 = x[1].tolist()

            payload = {"signature_name":"predictions", "instances": [{"x1": x1, "x2":x2}]}

            r = requests.post('http://localhost:8501/v1/models/my_model:predict', json=payload)

            if r.status_code == 200:
                output = r.json()
            else:
                output = "Error!"

            return render_template('result.html', 
                                    output=output['predictions'][0], 
                                    filename1=real_signature_path, 
                                    filename2=test_signature_path)

        flash(error)

    return render_template('verify.html')
