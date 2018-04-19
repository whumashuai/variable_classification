import tensorflow as tf
import numpy as np

class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes,class_weight,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.float32, [None, sequence_length,embedding_size], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        self.embedded_chars_expanded = tf.expand_dims(self.input_x, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)

            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            weights = tf.reduce_sum(class_weight * self.input_y, axis=1)
            self.loss = tf.reduce_mean(losses*weights) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        with tf.name_scope("evaluate_measures"):
            self.labels = tf.argmax(self.input_y, 1)
            p_cond = tf.equal(self.labels,
                              0)  # get positive answer position, return a list of True and False, the p position corresponds to True
            n_cond = tf.equal(self.labels, 1)  # get negative answer position
            p_prediction = tf.boolean_mask(self.predictions, p_cond)
            p_label = tf.boolean_mask(self.labels, p_cond)
            self.tp = tf.reduce_sum(tf.cast(tf.equal(p_prediction, p_label), "float"), name="tp")
            self.fp = tf.subtract(tf.reduce_sum(tf.cast(tf.equal(self.predictions, 0), "float")), self.tp)
            n_prediction = tf.boolean_mask(self.predictions, n_cond)
            n_label = tf.boolean_mask(self.labels, n_cond)
            self.tn = tf.reduce_sum(tf.cast(tf.equal(n_prediction, n_label), "int64"), name="tn")
            self.fn = tf.subtract(tf.reduce_sum(self.predictions), self.tn)
            # cast tp, fp, tn, fn to float64
            self.tp = tf.cast(self.tp, tf.float64)
            self.fp = tf.cast(self.fp, tf.float64)
            self.tn = tf.cast(self.tn, tf.float64)
            self.fn = tf.cast(self.fn, tf.float64)
            # Precision
            self.precision = tf.div(self.tp, tf.add(self.tp, self.fp), name="precision")
            # Recall
            self.recall = tf.div(self.tp, tf.add(self.tp, self.fn), name="recall")
            # F1-score
            self.product = tf.multiply(tf.cast(tf.constant(2.0), tf.float64),
                                   tf.multiply(self.precision, self.recall))
            self.f1_score = tf.div(self.product, tf.add(self.precision, self.recall), name="f1_score")

