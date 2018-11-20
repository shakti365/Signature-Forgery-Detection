import tensorflow as tf
import numpy as np
import os
import utils


class SiameseCNN:

    def __init__(self, config):

        self.data_path = config['data_path']
        self.train_batch_size = config['valid_batch_size']
        self.valid_batch_size = config['train_batch_size']
        self.seed = config['seed']
        self.learning_rate=config['learning_rate']
        self.epochs = config['epochs']
        self.export_dir = os.path.join(config['export_dir'], config['model_name'])
        self.model_name = config['model_name']
        self.SERVING_DIR=os.path.join(self.export_dir, self.model_name+'_serving', '1')
        self.TF_SUMMARY_DIR=os.path.join(self.export_dir, self.model_name+'_summary')
        self.CKPT_DIR=os.path.join(self.export_dir, self.model_name+'_checkpoint')
        self.log_step=config['log_step']                


    def numpy_input_fn(self):
        """

        """
    
        # Get train set
        train = np.load(os.path.join(self.data_path, 'train.npz'))
        X_1_train=train['X_1_train'].astype(np.float32)
        X_2_train=train['X_2_train'].astype(np.float32)
        X_3_train=train['X_3_train'].astype(np.float32)

        # Get valid set
        valid = np.load(os.path.join(self.data_path, 'valid.npz'))
        X_1_valid=valid['X_1_valid'].astype(np.float32)
        X_2_valid=valid['X_2_valid'].astype(np.float32)
        X_3_valid=valid['X_3_valid'].astype(np.float32)

        # Create Dataset object from input.
        train_dataset = tf.data.Dataset.from_tensor_slices((X_1_train, X_2_train, X_3_train)).batch(self.train_batch_size)
        valid_dataset = tf.data.Dataset.from_tensor_slices((X_1_valid, X_2_valid, X_3_valid)).batch(self.valid_batch_size)
        
        # Create generic iterator.
        data_iter = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)

        # Create initialisation operations.
        train_init_op = data_iter.make_initializer(train_dataset)
        valid_init_op = data_iter.make_initializer(valid_dataset)

        return train_init_op, valid_init_op, data_iter


    def model(self, x):
        """

        """

        with tf.variable_scope("signet", reuse=tf.AUTO_REUSE):
            conv1 = tf.layers.conv2d(inputs=x, filters=96, kernel_size=[11,11], activation=tf.nn.relu) 
            norm1 = tf.nn.local_response_normalization(conv1)
            pool1 = tf.layers.max_pooling2d(inputs=norm1, pool_size=[3,3], strides=2)
            pad1 = tf.pad(pool1, paddings=tf.constant([[0, 0], [2,2], [2,2], [0,0]]))

            conv2 = tf.layers.conv2d(inputs=pad1, filters=256, kernel_size=[5,5], activation=tf.nn.relu) 
            norm2 = tf.nn.local_response_normalization(conv2) 
            pool2 = tf.layers.max_pooling2d(inputs=norm2, pool_size=[3,3], strides=2)


            conv3 = tf.layers.conv2d(inputs=pool2, filters=384, kernel_size=[3,3], activation=tf.nn.relu) 
            conv4 = tf.layers.conv2d(inputs=conv3, filters=256, kernel_size=[3,3], activation=tf.nn.relu) 
            pool3 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[3,3], strides=2)

            shapes = pool3.get_shape()
            pool3_flattened = tf.reshape(pool3, [-1, shapes[1] * shapes[2] * shapes[3]])
            dense1 = tf.layers.dense(inputs=pool3_flattened, units=1024, activation=tf.nn.relu)
            logits = tf.layers.dense(inputs=dense1, units=128, activation=tf.nn.relu)

        return logits


    def euclidean_distance(self, y_true, y_pred):
        """
        """
        with tf.name_scope("euclidean_norm"):
            euclidean_norm = tf.math.reduce_sum(tf.math.squared_difference(y_pred, y_true), axis=-1)
        return euclidean_norm

    def triplet_loss(self, anchor, positive, negative):
        """
        """
        with tf.name_scope("triplet_loss"):
            positive_dist= self.infer(anchor, positive)
            negative_dist = self.infer(anchor, negative)

            margin = 0.05
            triplet_loss_op = tf.math.maximum(0.0, margin + positive_dist - negative_dist)

            loss =  tf.math.reduce_mean(triplet_loss_op)
        return loss


    def train(self, x1, x2, x3):
        """
        
        """
        with tf.name_scope("train"):
            loss_op = self.triplet_loss(x1, x2, x3)
            train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss_op) 

            summary_op, metrics_update_op = utils.get_metrics(loss_op)

        return train_op, summary_op, metrics_update_op


    def infer(self, x1, x2, reuse=tf.AUTO_REUSE):
        """

        """
        with tf.variable_scope("siamese_network", reuse=reuse) as scope:
            embedding_x1 = self.model(x1) 
            scope.reuse_variables()
            embedding_x2 = self.model(x2) 

        embedding_dist = self.euclidean_distance(embedding_x1, embedding_x2)

        return embedding_dist

    def fit(self):

        # Check if the export directory is present,
        # if not present create new directory.
        if os.path.exists(self.export_dir):
            raise ValueError("Export directory already exists. Please specify different export directory.")
        else:
            os.mkdir(self.export_dir)


        self.builder=tf.saved_model.builder.SavedModelBuilder(self.SERVING_DIR)


        tf.reset_default_graph()
        tf.set_random_seed(self.seed)

        train_init_op, valid_init_op, data_iter = self.numpy_input_fn()
        x1, x2, x3 = data_iter.get_next()

        train_op, summary_op, metrics_update_op = self.train(x1, x2, x3)

        prediction = self.infer(x1, x2)

        # Ops to reset metrics at the end of every epoch
        metrics_global_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='metrics')
        metrics_local_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='metrics')
        metrics_vars = metrics_global_vars + metrics_local_vars
        metrics_init_op = tf.variables_initializer(var_list=metrics_vars)

        # Object to saver model checkpoints
        self.saver = tf.train.Saver()
        
        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            # Create file writer directory to store summary and events.
            train_writer = tf.summary.FileWriter(self.TF_SUMMARY_DIR+'/train', sess.graph)
            valid_writer = tf.summary.FileWriter(self.TF_SUMMARY_DIR+'/valid')

            for epoch in range(self.epochs):

                sess.run(train_init_op)

                while True:
                    try:
                        train_summary,_,_ = sess.run([summary_op, train_op, metrics_update_op])
                    except tf.errors.OutOfRangeError:
                        break

                if epoch % self.log_step == 0:

                    train_writer.add_summary(train_summary, epoch)

                    # Save model checkpoint.
                    self.saver.save(sess, self.CKPT_DIR+"{}.ckpt".format(self.model_name))
                    
                    sess.run(valid_init_op)

                    while True:
                        try:
                            valid_summary,_ = sess.run([summary_op, metrics_update_op])
                            valid_writer.add_summary(valid_summary, epoch)
                        except tf.errors.OutOfRangeError:
                            break
                    

            prediction_signature = self.create_prediction_signature(x1, x2, prediction)
            self.save_servables(prediction_signature, signature_def_key='predictions')


    def predict(self, x1, x2):
        """
        """ 

        tf.reset_default_graph()
        
        x1_ = tf.placeholder(shape=[None, 155, 220, 1], dtype=tf.float32, name='input')
        x2_ = tf.placeholder(shape=[None, 155, 220, 1], dtype=tf.float32, name='input')

        embedding_dist = self.infer(x1_, x2_, reuse=True)

        self.saver = tf.train.Saver()

        with tf.Session() as sess:

            self.saver.restore(sess, self.CKPT_DIR+"{}.ckpt".format(self.model_name))

            dist = sess.run(embedding_dist, {x1_:x1, x2_:x2})

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
        tensor_info_x2 = tf.saved_model.utils.build_tensor_info(x1)
        
        tensor_info_prediction = tf.saved_model.utils.build_tensor_info(prediction)

        # Create prediction signature.
        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                method_name = tf.saved_model.signature_constants.PREDICT_METHOD_NAME,
                inputs = {
                    'x1': tensor_info_x1,
                    'x2': tensor_info_x2
                },
                outputs = {
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
            self.saver.restore(sess, self.CKPT_DIR+"{}.ckpt".format(self.model_name))
            
            # Map signature defnition with a specified key.
            signature_def_map = {
                signature_def_key: prediction_signature
            }
            
            # Tags for meta-graph.
            tags = [tf.saved_model.tag_constants.SERVING]
            
            # Saving tensorflow servables.
            self.builder.add_meta_graph_and_variables(
                sess = sess,
                tags = tags,
                signature_def_map = signature_def_map
            )
            self.builder.save()


if __name__=="__main__":

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
    siamese_model.fit()

    # Get test set
    test = np.load(os.path.join(config['data_path'], 'test.npz'))
    X_1_test=test['X_1_test'].astype(np.float32)[:5]
    X_2_test=test['X_2_test'].astype(np.float32)[:5]
    X_3_test=test['X_3_test'].astype(np.float32)[:5]

    pos = siamese_model.predict(X_1_test, X_2_test)
    neg = siamese_model.predict(X_1_test, X_3_test)

    print (pos, neg)
