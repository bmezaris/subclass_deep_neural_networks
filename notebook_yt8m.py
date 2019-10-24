"""
Implementation in Tensorflow of the method and multilabel classification experiments
in the YouTube8M dataset using a rather shallow network, as described in the paper:
N. Gkalelis, V. Mezaris, "Subclass deep neural networks: re-enabling neglected classes
in deep network training for multimedia classification", Proc. 26th Int. Conf. on
Multimedia Modeling (MMM2020), Daejeon, Korea, Jan. 2020.

History
-------
DATE       | DESCRIPTION    | NAME              | Organization |
21/07/2019 | first creation | Nikolaos Gkalelis | CERTH-ITI    |
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from tensorflow import flags
import numpy as np

import pandas as pd
from model.subclasscnn import SubclassCnnTf

print("Tensorflow version: ", tf.VERSION)
print("Keras version: ", tf.keras.__version__)

FLAGS = flags.FLAGS

flags.DEFINE_string("train_data_pattern_glob", r'.\data\yt8m\video\train_subclass\train*.tfrecord', "File glob for the training dataset.") # 3
flags.DEFINE_string("subclass_labelling_data_save_fname", r'.\data\yt8m\video\train_subclass\subclassLabellingData.npz',
                    "Filename of file with the subclass labelling information.")
flags.DEFINE_string("eval_data_pattern_glob", r'.\data\yt8m\video\validate\validate*.tfrecord', "File glob for the validation dataset.")

flags.DEFINE_string("feature_names", "mean_rgb", "Name of the feature to use for training.")
flags.DEFINE_bool("shuffle_data", True, "Shuffle the data on read.")
flags.DEFINE_integer("num_parallel_calls", 4, "Number of threads to use in map function when processing the dataset.")
flags.DEFINE_integer("shuffle_buffer_size", 10000, "Buffer size for suffling batch observations (no more than 10000).")
flags.DEFINE_integer("num_subclasses", 3862+100, "Number of threads to use in map function when processing the dataset.") # 3862
flags.DEFINE_integer("num_classes", 3862, "Number of threads to use in map function when processing the dataset.") # 3862
flags.DEFINE_integer("num_train_observations", 3888919, "Number of training observations.")
flags.DEFINE_integer("top_k", 20, "How many predictions to output per video.")

# Model flags.
flags.DEFINE_string("dr_method", "subclassVgg16", "projection method.")
flags.DEFINE_bool("perform_training", True, "Perform training or use stored trained model?")
flags.DEFINE_integer("batch_size", 1024, "How many examples to process per batch for training.")
flags.DEFINE_float("regularization_penalty", 1e-3, "How much weight to give to the regularization loss (the label loss has a weight of 1).") # 1e-3, 1e-5
flags.DEFINE_float("base_learning_rate", 0.001, "Which learning rate to start with.")
flags.DEFINE_float("learning_rate_decay", 0.95, "Learning rate decay factor to be applied every learning_rate_decay_examples.")
flags.DEFINE_float("learning_rate_decay_examples", 4000000, "Multiply current learning rate by learning_rate_decay every learning_rate_decay_examples.") # 3888919 4000000
flags.DEFINE_integer("num_epochs", 1, "How many passes to make over the dataset before halting training.") # 5
flags.DEFINE_integer("max_steps", 90000000000, "The maximum number of iterations of the training loop.") # 200, 14000, 90000000000

# Other flags.
flags.DEFINE_string("optimizer", "AdamOptimizer", "What optimizer class to use.")


def main(unused_argv):

    train_data_pattern = os.path.join(FLAGS.train_data_pattern_glob) # absolute file glob for the dataset
    eval_data_pattern = os.path.join(FLAGS.eval_data_pattern_glob) # absolute file glob for the validation dataset
    model_save_path = '.\checkpoints' # Path to save trained models

    """ Initialize """
    aSubclassCnn = SubclassCnnTf(num_classes = FLAGS.num_classes,
                                 num_subclasses = FLAGS.num_subclasses,
                                 subclass_labelling_fname = FLAGS.subclass_labelling_data_save_fname,
                                 top_k= FLAGS.top_k,
                                 reg_pen = FLAGS.regularization_penalty,
                                 base_learning_rate= FLAGS.base_learning_rate,
                                 learning_rate_decay_examples= FLAGS.learning_rate_decay_examples,
                                 learning_rate_decay= FLAGS.learning_rate_decay, model_save_path= model_save_path,
                                 total_num_observations=FLAGS.num_train_observations) # initialize DR method

    """ Learn  """
    aSubclassCnn= aSubclassCnn.fit(
        trainDataPattern=train_data_pattern,
        numEpochs= FLAGS.num_epochs,
        batchSz= FLAGS.batch_size,
        shuffleBufSz= FLAGS.shuffle_buffer_size,
        numPrlCalls= FLAGS.num_parallel_calls,
        shuffleData= FLAGS.shuffle_data,
        max_steps=FLAGS.max_steps)

    """ predict class for validation data """
    epoch_info_dict = aSubclassCnn.predict(
        dataPattern=eval_data_pattern,
        numPrlCalls=FLAGS.num_parallel_calls,
        batchSz=2 ** 10)

    avg_hit_at_one = epoch_info_dict["avg_hit_at_one"]
    avg_perr = epoch_info_dict["avg_perr"]
    avg_loss = epoch_info_dict["avg_loss"]
    aps = epoch_info_dict["aps"]
    gap = epoch_info_dict["gap"]
    mean_ap = np.mean(aps)

    print("Avg_Hit@1: {0:02.3f} | Avg_PERR: {1:02.3f} | MAP: {2:02.3f} | GAP: {3:02.3f} | Avg_Loss: {4:02.3f}".format(
        avg_hit_at_one, avg_perr, mean_ap, gap, avg_loss))

if __name__ == "__main__":
    tf.app.run(main=main)