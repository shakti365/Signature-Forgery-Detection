from keras import layers, Input, Model, optimizers
import keras.backend as K
import numpy as np
import os

data_path = '../../data/processed/'

train = np.load(os.path.join(data_path, 'train.npz'))
X_1_train=train['X_1_train']
X_2_train=train['X_2_train']
X_3_train=train['X_3_train']
y_train=train['y_train']

valid = np.load(os.path.join(data_path, 'valid.npz'))
X_1_valid=valid['X_1_valid']
X_2_valid=valid['X_2_valid']
X_3_valid=valid['X_3_valid']
y_valid=valid['y_valid']

test = np.load(os.path.join(data_path, 'test.npz'))
X_1_test=test['X_1_test']
X_2_test=test['X_2_test']
X_3_test=test['X_3_test']
y_test=test['y_test']

inputTensor = Input((155,220,1))

conv1 = layers.Conv2D(filters=96, 
                      kernel_size=(11,11), 
                      strides=1, 
                      activation='relu', 
                      input_shape=(155, 220, 1), 
                      data_format="channels_last")(inputTensor)

conv1_norm = layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001,center=True,
            scale=True, beta_initializer='zeros', gamma_initializer='ones',
            moving_mean_initializer='zeros',moving_variance_initializer='ones')(conv1)

conv1_pool = layers.MaxPooling2D(pool_size=(3,3), 
                                 strides=2)(conv1_norm)

conv2_padding = layers.ZeroPadding2D((2, 2))(conv1_pool)

conv2 = layers.Conv2D(filters=256, 
                      kernel_size=(5,5), 
                      strides=1, 
                      activation='relu')(conv2_padding)

conv2_norm = layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001,center=True,
            scale=True, beta_initializer='zeros', gamma_initializer='ones',
            moving_mean_initializer='zeros',moving_variance_initializer='ones')(conv2)

conv2_pool = layers.MaxPooling2D(pool_size=(3,3), 
                                 strides=2)(conv2_norm)

conv2_dropout = layers.Dropout(0.3, seed=1)(conv2_pool)

conv3_padding = layers.ZeroPadding2D((1,1))(conv2_dropout)

conv3 = layers.Conv2D(filters=384, 
                      kernel_size=(3,3), 
                      strides=1, 
                      activation = 'relu')(conv3_padding)

conv4_padding = layers.ZeroPadding2D((1,1))(conv3)

conv4 = layers.Conv2D(filters=256, 
                      kernel_size=(3,3), 
                      strides=1, 
                      activation='relu')(conv4_padding)
                                                                    
conv4_pool = layers.MaxPooling2D(pool_size=(3,3), 
                                 strides=2)(conv4)
                                                                    
conv4_dropout = layers.Dropout(0.3, seed=1)(conv4_pool)
                                                                    
flatten_layer = layers.Flatten()(conv4_dropout)
                                                                    
fully_connected1 = layers.Dense(1024)(flatten_layer)

fc1_dropout = layers.Dropout(0.5, seed=1)(fully_connected1)
                                                                    
embedding = layers.Dense(128)(fc1_dropout)
                                                                    
embedding_model = Model(inputs=[inputTensor], 
                         outputs=embedding, 
                         name='embedding_model')
                                                                    

def euclidean_distance_loss(y_true, y_pred):
    """
    :param y_true: TensorFlow/Theano tensor
    :param y_pred: TensorFlow/Theano tensor of the same shape as y_true
    :return: float
    """
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))
    
def triplet_loss(embeddings):
    """
    calculates triplet loss over inputs.
    """
    
    processed_a, processed_p, processed_n = embeddings[0], embeddings[1], embeddings[2]
    
    positive_dist= euclidean_distance_loss(processed_a, processed_p)
    negative_dist = euclidean_distance_loss(processed_a, processed_n)
       
    margin = 0.0
    loss = K.maximum(margin, positive_dist - negative_dist)
    
    return K.mean(loss)
    
def identity_loss(y_true, y_pred):
    """
    Fake loss function for Keras.
    """
    return y_pred - 0 * y_true
    
# Siamese model

in_dim=(155,220,1)
input_anchor = Input(shape=(in_dim))
input_positive = Input(shape=(in_dim))
input_negative = Input(shape=(in_dim))
embedding_a=embedding_model(input_anchor)
embedding_p=embedding_model(input_positive)
embedding_n=embedding_model(input_negative)

embedding_concat = layers.concatenate(inputs=[embedding_a, 
                                    embedding_p, 
                                    embedding_n], axis=-1)

loss_layer = layers.Lambda(function=triplet_loss, 
                     output_shape=(1,))

loss = loss_layer(embedding_concat)

siamese_model = Model(input=[input_anchor, input_positive, input_negative], 
                      output=loss)

siamese_model.compile(loss=identity_loss, optimizer=optimizers.Adam())

siamese_model.fit(x=[X_1_train, X_2_train, X_3_train], y=y_train, batch_size=64)