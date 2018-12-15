import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_curve, roc_auc_score, f1_score, accuracy_score
import operator

def get_metrics(loss):
    """
    Return accuracy, precision, recall and f1_score metrics

    Parameters
    ----------
        labels: Tensor shape=[batch_size, 1]
            true labels.
        predicted_proba: Tensor shape=[batch_size, 1]
            predicted probabilities.
        threshold: float
            threshold used for predicted probabilties.
        loss: Tensor shape=[1]
            Weighted mean of loss over every sample in batch.
    
    Returns
    -------
        merged: Tensor
            Tensor Op that merges all the tensorboard summaries for logging.
        grouped_update_op: Tensor
            Tensor Op which merges all the metrics updated ops into one.
    """

    with tf.variable_scope('metrics'):

        # create variable for storing sum and count of loss to calculate mean over batch. 
        total = tf.get_variable(initializer=0.0, dtype=tf.float32, name='total_loss', trainable=False)
        count = tf.get_variable(initializer=0.0, dtype=tf.float32, name='count_loss', trainable=False)
	
	
        with tf.name_scope('mean_loss'):
            update_total = tf.assign_add(total, loss)
            update_count = tf.assign_add(count, 1.0)
            mean_loss = tf.divide(update_total, update_count)
            mean_loss_update = tf.group([update_total, update_count])

        tf.summary.scalar('mean_loss', mean_loss)

        merged = tf.summary.merge_all()
			       
    return merged, mean_loss, mean_loss_update


def display_metrics(y, logits, threshold):
    """
    Log metrics for binary classification on console output.
    
    Parameters
    ----------
        y: numpy array shape=[num_samples, 1]
            True labels for the samples.
        
        logits: numpy array shape=[num_samples, 1]
            Predicted probabilities for the samples.
        
        threshold: float
            Threshold for classification.
    """
    # reshape results for train
    y = y.flatten()
    logits = logits.flatten()
    y_ = np.where(logits>threshold, 1, 0)
    
    confusion = confusion_matrix(y_true=y, y_pred=y_)
    accuracy = accuracy_score(y_true=y, y_pred=y_)
    precision = precision_score(y_true=y, y_pred=y_)
    recall = recall_score(y_true=y, y_pred=y_)
    auc = roc_auc_score(y, logits)
    f1 = f1_score(y, y_)
    
    print ("confusion matrix: \n", confusion)
    print ("accuracy score: ", accuracy) 
    print ("precision score: ", precision)
    print ("recall score: ", recall)
    print ("F1 score: ", f1)
    print ("AUC: ", auc)


def variable_summaries(var, var_name=None):
    """
    Creates summaries of a tensor variable.

    Parameters
    ----------
        var: Tensor
             A tensor variable for which the summaries have to be created.
        var_name: String (default None)
             Name with which the summaries are logged.
             If None, it takes the name of the variable.
    """
    if var_name is None:
        var_name= var.name

    mean = tf.reduce_mean(var)
    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('{}_mean'.format(var_name), mean)
    tf.summary.scalar('{}_stddev'.format(var_name), stddev)
    tf.summary.histogram('{}_histogram'.format(var_name), var)

def gradient_summaries(gradients):
    """
    Creates gradient summaries.

    Parameters
    ----------
        gradients: List of Tuple [(Tensor,Tensor)]
            A list of tuples of tensor varible and values.
    """
    with tf.name_scope('gradients'):
        # Save the gradients summary for tensorboard.
        for grad in gradients:
	    # Assign a name to identify gradients.
	    var_name = '{}-grad'.format(grad[1].name)
	    if 'bias' not in var_name:
	        variable_summaries(grad[0], var_name=var_name)

def parameter_summaries(variables):
    """
    Creates parameter summaries.

    Parameter
    ---------
        variables: List of Tensor
            A list of trainable parameters in the model.
    """
    with tf.name_scope('variables'):
        # Save the gradients summary for tensorboard.
        for var in variables:
	    # Assign a name to identify gradients.
	    var_name = '{}-variable'.format(var.name)
	    if 'bias' not in var_name:
	        variable_summaries(var, var_name=var_name)
    
def find_threshold(pos_dist, neg_dist):
    """
    receives the euclidean dist for similar(post_dist) and dissimilar(neg_dist) images,
    compares the accuracy for different values of these distances ranging from their min to max values.
    The dist which gives max accuracy is returned as threshold.

    :param pos_dist:
    :param neg_dist:
    :return:
    """
    min_dist = min([min(pos_dist), min(neg_dist)])

    max_dist = max([max(pos_dist), max(neg_dist)])
    print min_dist, max_dist
    accuracy = {}
    true_pos = {}
    true_neg = {}
    false_pos = {}
    false_neg = {}

    dist = min_dist
    while dist <= max_dist:
        tp, tn, fp, fn = 0, 0, 0, 0

        for pos in pos_dist:

            if pos < dist:
                tp = tp + 1
            else:
                fn = fn + 1

        for neg in neg_dist:
            if neg > dist:
                tn = tn + 1
            else:
                fp = fp + 1

        tpr = tp / float(len(pos_dist))
        tnr = tn / float(len(neg_dist))

        true_pos[dist] = tp
        true_neg[dist] = tn
        false_pos[dist] = fp
        false_neg[dist] = fn
        accuracy[dist] = (tpr + tnr) / 2.0
        dist = dist + 0.01

    print "accuracy", accuracy
    threshold = max(accuracy.iteritems(), key=operator.itemgetter(1))[0]
    return threshold, accuracy, true_pos, true_neg, false_pos, false_neg
