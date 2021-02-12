# Signature Forgery Detection

[One-Shot Signature Recognition Using Siamese Networks](https://github.com/shakti365/Signature-Forgery-Detection/blob/master/One-Shot%20Signature%20Recognition%20Using%20Siamese%20Networks.md)

## Getting Started

- create a `python 2.7` virtual environment and activate it

- install dependencies using `pip install -r requirements.txt`

- download the data from [here](https://drive.google.com/file/d/1HSGFgrS6592p4olpxWMRdRCHiOKgfaYB/view?usp=sharing) and store it in `data/raw`

- extract the `Dataset_Signature_Final.zip` file and store it in `data/raw`

- run the script to clean and store data in `data/interim`
```python
cd src/data
python clean.py
```
- run the script to preprocess and store data in `data/processed`
```python
cd src/data
python preprocess.py
```

- run the script to train model
```python
cd src/model
python model.py
```

- run tensorboard
```
tensorboard --logdir=<path/to/model/summary>
```

- run the flask server in `development` mode
```
cd src
export FLASK_APP=serving
export FLASK_ENV=development
flask run --host=0.0.0.0 --port=8000
```

- run tensorflow model serving
```
tensorflow_model_server --port=8500 --rest_api_port=8501 --model_name=my_model --model_base_path=<path/to/model/serving>
```