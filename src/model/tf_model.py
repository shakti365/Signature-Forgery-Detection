import tensorflow as tf
import numpy as np
import os


class SiameseCNN:

    def __init__(self):

        self.data_path = '../../data/processed/'
        self.train_batch_size = 8
        self.valid_batch_size = 8
        self.seed = 42
        self.learning_rate=0.001
        self.epochs = 1

    def numpy_input_fn(self, train_X_1, train_X_2, train_X_3, train_y, valid_X_1, valid_X_2, valid_X_3, valid_y):
        """

        """
    
        # Create Dataset object from input.
        train_dataset = tf.data.Dataset.from_tensor_slices((train_X_1, train_X_2, train_X_3, train_y)).batch(self.train_batch_size)
        valid_dataset = tf.data.Dataset.from_tensor_slices((valid_X_1, valid_X_2, valid_X_3, valid_y)).batch(self.valid_batch_size)
        
        # Create generic iterator.
        data_iter = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)

        # Create initialisation operations.
        train_init_op = data_iter.make_initializer(train_dataset)
        valid_init_op = data_iter.make_initializer(valid_dataset)

        return train_init_op, valid_init_op, data_iter


    def model(self, x, reuse=False):
        """

        """

        with tf.variable_scope("siamese_cnn", reuse=reuse):
            conv1 = tf.layers.conv2d(inputs=x, filters=96, kernel_size=[11,11], activation=tf.nn.relu) 
            norm1 = tf.nn.local_response_normalization(conv1)
            pool1 = tf.layers.max_pooling2d(inputs=norm1, pool_size=[3,3], strides=2)
            pad1 = tf.pad(pool1, paddings=tf.constant([[0, 0], [2,2], [2,2], [0,0]]))

            conv2 = tf.layers.conv2d(inputs=pad1, filters=256, kernel_size=[5,5]) 
            norm2 = tf.nn.local_response_normalization(conv2) 
            pool2 = tf.layers.max_pooling2d(inputs=norm2, pool_size=[3,3], strides=2)


            conv3 = tf.layers.conv2d(inputs=pool2, filters=384, kernel_size=[3,3]) 
            conv4 = tf.layers.conv2d(inputs=conv3, filters=256, kernel_size=[3,3]) 
            pool3 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[3,3], strides=2)

            shapes = pool3.get_shape()
            pool3_flattened = tf.reshape(pool3, [-1, shapes[1] * shapes[2] * shapes[3]])
            dense1 = tf.layers.dense(inputs=pool3_flattened, units=1024)
            logits = tf.layers.dense(inputs=dense1, units=128)

        return logits


    def euclidean_distance(self, y_true, y_pred):
        """
        """
        euclidean_norm = tf.math.sqrt(tf.math.reduce_sum(tf.math.squared_difference(y_pred, y_true), axis=-1))
        return euclidean_norm

    def triplet_loss(self, embeddings):
        """
        """

        processed_a, processed_p, processed_n = embeddings[0], embeddings[1], embeddings[2]

        positive_dist= self.euclidean_distance(processed_a, processed_p)
        negative_dist = self.euclidean_distance(processed_a, processed_n)

        margin = 0.0
        triplet_loss_op = tf.math.maximum(margin, positive_dist - negative_dist)

        loss =  tf.math.reduce_mean(triplet_loss_op)
        return loss


    def train(self, x1, x2, x3, y):
        """
        
        """
        embedding_x1 = self.model(x1) 
        embedding_x2 = self.model(x2, reuse=True) 
        embedding_x3 = self.model(x3, reuse=True)
        loss_op = self.triplet_loss([embedding_x1, embedding_x2, embedding_x3])
        train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss_op) 
        return loss_op, train_op



    def fit(self):

        tf.reset_default_graph()
        tf.set_random_seed(self.seed)

        # Get train set
        train = np.load(os.path.join(self.data_path, 'train.npz'))
        X_1_train=train['X_1_train'].astype(np.float32)
        X_2_train=train['X_2_train'].astype(np.float32)
        X_3_train=train['X_3_train'].astype(np.float32)
        y_train=train['y_train'].astype(np.float32)

        # Get valid set
        valid = np.load(os.path.join(self.data_path, 'valid.npz'))
        X_1_valid=valid['X_1_valid'].astype(np.float32)
        X_2_valid=valid['X_2_valid'].astype(np.float32)
        X_3_valid=valid['X_3_valid'].astype(np.float32)
        y_valid=valid['y_valid'].astype(np.float32)

        train_init_op, valid_init_op, data_iter = self.numpy_input_fn(X_1_train, X_2_train, X_3_train, y_train, X_1_valid, X_2_valid, X_3_valid, y_valid)
        X1, X2, X3, Y = data_iter.get_next()

        loss_op, train_op = self.train(X1, X2, X3, Y)
        
        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            
            for epoch in range(self.epochs):

                sess.run(train_init_op)

                while True:
                    try:
                        loss, _ = sess.run([loss_op, train_op])
                        print (epoch, loss)
                    except tf.errors.OutOfRangeError:
                        break




if __name__=="__main__":
    
    siamese_model = SiameseCNN()
    siamese_model.fit()
