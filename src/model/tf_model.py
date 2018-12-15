
import utils
import tensorflow as tf
import numpy as np
import os


class SiameseCNN:

    def __init__(self, config):

        self.data_path = config['data_path']
        self.train_batch_size = config['valid_batch_size']
        self.valid_batch_size = config['train_batch_size']
        self.seed = config['seed']
        self.learning_rate = config['learning_rate']
        self.epochs = config['epochs']
        self.export_dir = os.path.join(config['export_dir'], config['model_name'])
        self.model_name = config['model_name']
        self.SERVING_DIR = os.path.join(self.export_dir, self.model_name + '_serving', '1')
        self.TF_SUMMARY_DIR = os.path.join(self.export_dir, self.model_name + '_summary')
        self.CKPT_DIR = os.path.join(self.export_dir, self.model_name + '_checkpoint')
        self.log_step = config['log_step']

    def numpy_input_fn(self):
        """
        Returns tf.data API ops for train and valid set.

        Returns:
        --------
            train_init_op: tensorflow op to initialize data_iter with train set data
            valid_init_op: tensorflow op to initialize data_iter with test set data
            data_iter: iterable over dataset
        """
    
        # Get train set
        train = np.load(os.path.join(self.data_path, 'train.npz'))
        X_1_train = train['X_1_train'].astype(np.float32)
        X_2_train = train['X_2_train'].astype(np.float32)
        X_3_train = train['X_3_train'].astype(np.float32)

        # Get valid set
        valid = np.load(os.path.join(self.data_path, 'valid.npz'))
        X_1_valid = valid['X_1_valid'].astype(np.float32)
        X_2_valid = valid['X_2_valid'].astype(np.float32)
        X_3_valid = valid['X_3_valid'].astype(np.float32)

        # Get test set
        test = np.load('/content/gdrive/My Drive/processed/test.npz')
        X_1_test = test['X_1_test'].astype(np.float32)
        X_2_test = test['X_2_test'].astype(np.float32)
        X_3_test = test['X_3_test'].astype(np.float32)

        X_1_valid = np.concatenate([X_1_valid, X_1_test], axis=0)
        X_2_valid = np.concatenate([X_2_valid, X_2_test], axis=0)
        X_3_valid = np.concatenate([X_3_valid, X_3_test], axis=0)

        # Create Dataset object from input.
        train_dataset = tf.data.Dataset.from_tensor_slices((X_1_train, X_2_train, X_3_train)).batch(
            self.train_batch_size)
        valid_dataset = tf.data.Dataset.from_tensor_slices((X_1_valid, X_2_valid, X_3_valid)).batch(
            self.valid_batch_size)

        # Create generic iterator.
        data_iter = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)

        # Create initialisation operations.
        train_init_op = data_iter.make_initializer(train_dataset)
        valid_init_op = data_iter.make_initializer(valid_dataset)

        return train_init_op, valid_init_op, data_iter

    def model(self, x):
        """
        Model architecture for SigNet returns embedding representation of input image.

        Parameter:
        ----------
            x: batch of images in NHWC format [num_samples, height, width, channel]

        Returns:
        --------
            logits: emebedding representation of signature from last layer.
        """

        with tf.variable_scope("signet", reuse=tf.AUTO_REUSE):
            conv1 = tf.layers.conv2d(inputs=x, filters=96, kernel_size=[11, 11], activation=tf.nn.relu)
            norm1 = tf.nn.local_response_normalization(conv1)
            pool1 = tf.layers.max_pooling2d(inputs=norm1, pool_size=[3, 3], strides=2)
            pad1 = tf.pad(pool1, paddings=tf.constant([[0, 0], [2, 2], [2, 2], [0, 0]]))

            conv2 = tf.layers.conv2d(inputs=pad1, filters=256, kernel_size=[5, 5], activation=tf.nn.relu)
            norm2 = tf.nn.local_response_normalization(conv2)
            pool2 = tf.layers.max_pooling2d(inputs=norm2, pool_size=[3, 3], strides=2)
            drop1 = tf.layers.dropout(pool2, 0.3)

            conv3 = tf.layers.conv2d(inputs=drop1, filters=384, kernel_size=[3, 3], activation=tf.nn.relu)
            conv4 = tf.layers.conv2d(inputs=conv3, filters=256, kernel_size=[3, 3], activation=tf.nn.relu)
            pool3 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[3, 3], strides=2)
            drop2 = tf.layers.dropout(pool3, 0.3)

            shapes = drop2.get_shape()
            pool3_flattened = tf.reshape(drop2, [-1, shapes[1] * shapes[2] * shapes[3]])
            dense1 = tf.layers.dense(inputs=pool3_flattened, units=1024, activation=tf.nn.relu)
            logits = tf.layers.dense(inputs=dense1, units=128, activation=None)
            normalized_logits = tf.math.l2_normalize(logits)

        return normalized_logits

    def euclidean_distance(self, y_true, y_pred):
        """
        Finds the euclidean distance between the two given inputs received.
        :param y_true:
        :param y_pred:
        :return: euclidean distance
        """
        with tf.name_scope("euclidean_norm"):
            euclidean_norm = tf.reduce_sum(tf.squared_difference(y_pred, y_true), axis=-1)
        return euclidean_norm

    def triplet_loss(self, anchor, positive, negative):
        """
        Computs the embedding distance for anchor-positive and anchor-negative and accordingly calculate the loss given
         by the below formula used in the code
        :param anchor:
        :param positive:
        :param negative:
        :return: loss
        """
        with tf.name_scope("triplet_loss"):
            positive_dist= self.infer(anchor, positive)
            negative_dist = self.infer(anchor, negative)

            margin = 0.5
            triplet_loss_op = tf.maximum(0.0, margin + positive_dist - negative_dist)

            loss = tf.reduce_mean(triplet_loss_op)
        return loss

    def train(self, x1, x2, x3):

        """
        The triplet loss is calculated by and is minimized using Optimizer.Returns the metrics and optimizer operation.
        :param x1:
        :param x2:
        :param x3:
        :return: train_op, summary_op, metrics_update_op, loss_op
        """
        with tf.name_scope("train"):

            loss_op = self.triplet_loss(x1, x2, x3)

            train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss_op)

            summary_op, mean_loss_op, metrics_update_op = utils.get_metrics(loss_op)

            tf.summary.scalar('loss', loss_op)

        return train_op, summary_op, metrics_update_op, loss_op, mean_loss_op

    def infer(self, x1, x2, reuse=tf.AUTO_REUSE):
        """
        returns embedding distance between x1 and x2
        :param x1:
        :param x2:
        :param reuse:
        :return: embedding_dist
        """
        with tf.variable_scope("siamese_network", reuse=reuse) as scope:
            embedding_x1 = self.model(x1)
            scope.reuse_variables()
            embedding_x2 = self.model(x2)

        embedding_dist = self.euclidean_distance(embedding_x1, embedding_x2)

        return embedding_dist

    def fit(self):
        """
        The end to end process of training is completed from data generation to model training to summarizing the loss
        and metrics.  The trained model is saved as a checkpoint for further usage.
        :return:
        """
        # Check if the export directory is present,
        # if not present create new directory.
        # if os.path.exists(self.export_dir):
        #     raise ValueError("Export directory already exists. Please specify different export directory.")
        # else:
        #     os.mkdir(self.export_dir)

        self.builder = tf.saved_model.builder.SavedModelBuilder(self.SERVING_DIR)

        # Clear default graph stack and set random seed.
        tf.reset_default_graph()
        tf.set_random_seed(self.seed)

        x1_ = tf.placeholder(shape=[None, 155, 220, 1], dtype=tf.float32, name='input')
        x2_ = tf.placeholder(shape=[None, 155, 220, 1], dtype=tf.float32, name='input')

        embedding_dist = self.infer(x1_, x2_, reuse=True)

        # Create tensorflow ops to iterate over train and valid dataset.
        train_init_op, valid_init_op, data_iter = self.numpy_input_fn()
        x1, x2, x3 = data_iter.get_next()

        # Get trainining, tensorboard summary, loss metrics updation and loss op.
        train_op, summary_op, metrics_update_op, loss_op, mean_loss_op = self.train(x1, x2, x3)

        # Get euclidean distance of embedding representations.
        prediction = self.infer(x1, x2)

        x1_ = tf.placeholder(shape=[None, 155, 220, 1], dtype=tf.float32, name='input')
        x2_ = tf.placeholder(shape=[None, 155, 220, 1], dtype=tf.float32, name='input')

        embedding_dist = self.infer(x1_, x2_, reuse=True)

        # Ops to reset metrics at the end of every epoch
        metrics_global_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='metrics')
        metrics_local_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='metrics')
        metrics_vars = metrics_global_vars + metrics_local_vars
        metrics_init_op = tf.variables_initializer(var_list=metrics_vars)

        # Object to saver model checkpoints
        self.saver = tf.train.Saver()

        with tf.Session() as sess:

            # Initialize local and global variables.
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            # Create file writer directory to store summary and events.
            train_writer = tf.summary.FileWriter(self.TF_SUMMARY_DIR + '/train', sess.graph)
            valid_writer = tf.summary.FileWriter(self.TF_SUMMARY_DIR + '/valid')

            ckpt = tf.train.get_checkpoint_state(self.export_dir)
            if ckpt and ckpt.model_checkpoint_path:
                self.saver.restore(sess, ckpt.model_checkpoint_path)

            # Run epochs
            for epoch in range(self.epochs):

                # Initialize data iterator with train set.
                sess.run(train_init_op)

                # Initialize metrics update op for train set steps over one epoch.
                sess.run(metrics_init_op)

                # Step over all the batches in train set
                while True:
                    try:
                        train_summary, _, _, train_loss, mean_train_loss = sess.run(
                            [summary_op, train_op, metrics_update_op, loss_op, mean_loss_op])

                    except tf.errors.OutOfRangeError:
                        break

                if epoch % self.log_step == 0:

                    # Log metrics for train set epoch
                    train_writer.add_summary(train_summary, epoch)
                    print("train loss for {} epoch is {}".format(epoch, mean_train_loss))

                    # Save model checkpoint.
                    #                     self.saver.save(sess, self.CKPT_DIR+"{}.ckpt".format(self.model_name))

                    # Initialize data iterator with valid set.
                    sess.run(valid_init_op)

                    # Initialize metrics update op for valid set steps over one epoch.
                    sess.run(metrics_init_op)

                    # Step over all the batches of valid set
                    while True:
                        try:
                            valid_summary, _, valid_loss, mean_valid_loss = sess.run(
                                [summary_op, metrics_update_op, loss_op, mean_loss_op])
                        #                             print (valid_loss, mean_valid_loss)
                        except tf.errors.OutOfRangeError:
                            break

                    # Log metrics for valid set epoch
                    valid_writer.add_summary(valid_summary, epoch)
                    print("valid loss for {} epoch is {}".format(epoch, mean_valid_loss))
                if epoch % 4 == 0:
                    self.saver.save(sess, self.CKPT_DIR + "{}.ckpt".format(self.model_name))

            # Create model serving at the end of all epochs and save it.

            prediction_signature = self.create_prediction_signature(x1_, x2_, embedding_dist)
            self.save_servables(prediction_signature, signature_def_key='predictions')

    def predict(self, x1, x2):
        """
        Predict euclidean distance between set of images and display metrics.

        Parameters:
        -----------
            x1: Numpy array [num_samples, height, width, channels]
                NHWC representation of images

            x2: Numpy array [num_samples, height, width, channels]
                NHWC representation of images

        Returns:
        --------
            dist: Numpy array [num_samples, 1]
                Euclidean distance of images for all samples in batch
        """

        tf.reset_default_graph()

        x1_ = tf.placeholder(shape=[None, 155, 220, 1], dtype=tf.float32, name='input')
        x2_ = tf.placeholder(shape=[None, 155, 220, 1], dtype=tf.float32, name='input')

        embedding_dist = self.infer(x1_, x2_, reuse=True)

        self.saver = tf.train.Saver()

        with tf.Session() as sess:
            self.saver.restore(sess, self.CKPT_DIR + "{}.ckpt".format(self.model_name))

            dist = sess.run(embedding_dist, {x1_: x1, x2_: x2})

        return dist

    def create_prediction_signature(self, x1, x2, prediction):
        """
        Creates prediction signature for tensorflow serving
        
        Returns
        -------
            prediction_signature: 
                Prediction signature with input and output defined 
                for model for setting up tensorflow serving.
                
        """
        # Create input and output utils for prediction signature.
        tensor_info_x1 = tf.saved_model.utils.build_tensor_info(x1)
        tensor_info_x2 = tf.saved_model.utils.build_tensor_info(x2)

        tensor_info_prediction = tf.saved_model.utils.build_tensor_info(prediction)

        # Create prediction signature.
        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME,
                inputs={
                    'x1': tensor_info_x1,
                    'x2': tensor_info_x2
                },
                outputs={
                    'logits': tensor_info_prediction
                },
            )
        )

        return prediction_signature

    def save_servables(self, prediction_signature, signature_def_key):
        """
        Loads the latest model checkpoint and saves model as a servable for production
        
        Parameters
        ----------
            prediction_signature: 
                Prediction signature with input and output defined 
                for model for setting up tensorflow serving.
            
            signature_def_key: string
                Specify a user-generated-key for SignatureDef, 
                this key is used to map Signature Definition.
        """
        with tf.Session() as sess:
            # Restore model checkpoint.
            self.saver.restore(sess, self.CKPT_DIR + "{}.ckpt".format(self.model_name))

            # Map signature defnition with a specified key.
            signature_def_map = {
                signature_def_key: prediction_signature
            }

            # Tags for meta-graph.
            tags = [tf.saved_model.tag_constants.SERVING]

            # Saving tensorflow servables.
            self.builder.add_meta_graph_and_variables(
                sess=sess,
                tags=tags,
                signature_def_map=signature_def_map
            )
            self.builder.save()


if __name__=="__main__":

    config = dict()

    config['data_path'] = '../../data/processed'
    config['valid_batch_size'] = 8
    config['train_batch_size'] = 8
    config['seed'] = 42
    config['learning_rate'] = 0.001
    config['epochs'] = 40
    config['export_dir'] = '../../data/models'
    config['model_name'] = 'exp_2'
    config['log_step'] = 2

    siamese_model = SiameseCNN(config)
    siamese_model.fit()

    """
    # Get test set
    test = np.load(os.path.join(config['data_path'], 'test.npz'))
    X_1_test=test['X_1_test'].astype(np.float32)[:8]
    X_2_test=test['X_2_test'].astype(np.float32)[:8]
    X_3_test=test['X_3_test'].astype(np.float32)[:8]

    pos = siamese_model.predict(X_1_test, X_2_test)
    neg = siamese_model.predict(X_1_test, X_3_test)

    threshold, accuracy, tp, tn, fp, fn = utils.find_threshold(pos, neg)
    print (threshold)
    print ("accuracy: ", accuracy[threshold])
    print ("tp: ", tp[threshold])
    print ("tn: ", tn[threshold])
    print ("fp: ", fp[threshold])
    print ("fn: ", fn[threshold])
    print ("precision: ", tp[threshold] / float(tp[threshold] + fp[threshold]))
    print ("recall: ", tp[threshold] / float(tp[threshold] + fn[threshold]))
    """
