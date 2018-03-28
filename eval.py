#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv

# Parameters
# ==================================================

# Data Parameters
tf.flags.DEFINE_string("data_file", "./cmm_path/dev.txt", "Data source for the positive data.")
tf.flags.DEFINE_string("max_data_length", 2000, "The max length of data.")
# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 1, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "./runs_cmm_path/", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# CHANGE THIS: Load data. Load your own data here
x_raw, y_test, category = data_helpers.load_data_and_labels(FLAGS.data_file)
pathdir = os.listdir(FLAGS.checkpoint_dir)
x_test = np.array(data_helpers.fit_transform(x_raw,FLAGS.max_data_length))
summary = [[0 for col in range(2)] for row in range(len(x_raw))]
for checkpointName in pathdir:

    checkpointPath = FLAGS.checkpoint_dir + checkpointName + '/checkpoints/'
    count = 0
    # Map data into vocabulary
    #vocab_path = os.path.join(checkpointPath, "..", "vocab")
    #vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)

    print("\nEvaluating...\n")

    # Evaluation
    # ==================================================
    checkpoint_file = tf.train.latest_checkpoint(checkpointPath)
    print(checkpoint_file)
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            # input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            scores = graph.get_operation_by_name("output/scores").outputs[0]

            # Generate batches for one epoch
            batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

            # Collect the predictions here
            all_predictions = []

            for x_test_batch in batches:
                batch_predictions = sess.run(tf.nn.softmax(scores), {input_x: x_test_batch, dropout_keep_prob: 1.0})
                if summary[count][0] == 0:
                    summary[count][0] = batch_predictions[0][0]
                    summary[count][1] = checkpointName
                else:
                    if batch_predictions[0][0] > summary[count][0]:
                        summary[count][0] = batch_predictions[0][0]
                        summary[count][1] = checkpointName
                count = count + 1

for k in  summary:
    print(k)
# Print accuracy
if y_test is not None:
    sums = 0
    for i in range(len(y_test)):
        if summary[i][1] == y_test[i]:
            sums = sums + 1
    correct_predictions = float(sums)
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))



