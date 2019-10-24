"""
Utilization functions for processing and learning from the tfrecord files of YouTube-8M dataset.

History
-------
DATE       | DESCRIPTION    | NAME              | Organization |
21/07/2019 | first creation | Nikolaos Gkalelis | CERTH-ITI    |
"""


import tensorflow as tf
import logging
from tensorflow import gfile

# local imports
from utils.youtubereader import YoutubeReader

def youTube8mAudioVisualIter(train_data_pattern= None,
                  shuffle_data= True,
                  num_classes= 3862,
                  num_parallel_calls= 4,
                  shuffle_buffer_size= 10000,
                  batch_size= 512,
                  num_epochs= 2,
                  ):
  """ youTube8mAudioVisualIter: iterator for parsing the tfrecords of YouTube-8M dataset.
      :param shuffle_data: boolean variable, to shuffle or not the data
      :param num_classes: number of classes
      :param num_parallel_calls: number of parallel calls for reading the data
      :param shuffle_buffer_size: buffer size for suffling batch observations
      :param batch_size: batch size
      :param num_epochs: number of epochs
  """

  files = tf.data.Dataset.list_files(file_pattern=train_data_pattern, shuffle=shuffle_data)
  dataset = tf.data.TFRecordDataset(files) # create dataset object from data in disk

  aYoutubeReader = YoutubeReader(num_classes=num_classes)

  dataset = dataset.map(map_func=aYoutubeReader.parse_function_all, num_parallel_calls=num_parallel_calls) # dataset transformations

  if shuffle_data is True: # if shuffle_data is not TRUE do not shuffle at all!!!
    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)

  dataset = dataset.batch(batch_size)
  dataset = dataset.repeat(num_epochs)

  batch_iter = dataset.make_initializable_iterator() # create iteration object for this dataset
  features_rgb, features_audio, y_batch, id_batch = batch_iter.get_next()


  features_rgb = tf.nn.l2_normalize(features_rgb, 1) # normalize
  features_audio = tf.nn.l2_normalize(features_audio, 1)
  x_batch = tf.concat([features_rgb, features_audio], axis=1) #concatenate

  return batch_iter, x_batch, y_batch, id_batch

def CrossEntropyLoss(predictions, labels):
  """ CrossEntropyLoss: Sigmoid cross-entropy loss.
      :param predictions: N x S array of predictions; N: number of training observations, S: number of subclasses
      :param labels: N x S array with one hot subclass vector at each row
      :return: cross entropy loss: scalar value
  """

  epsilon = 10e-6
  float_labels = tf.cast(labels, tf.float32)
  cross_entropy_loss = float_labels * tf.log(predictions + epsilon) + (
          1 - float_labels) * tf.log(1 - predictions + epsilon)
  cross_entropy_loss = tf.negative(cross_entropy_loss)
  return tf.reduce_sum(cross_entropy_loss, 1)
