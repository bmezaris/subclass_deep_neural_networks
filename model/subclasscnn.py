"""
Implementation of a rather shallow subclass CNN in Tensorflow.

History
-------
DATE       | DESCRIPTION    | NAME              | Organization |
21/07/2019 | first creation | Nikolaos Gkalelis | CERTH-ITI    |
"""



import tensorflow as tf
import tensorflow.contrib.slim as slim
from util.yt8m_utils import youTube8mAudioVisualIter
from util.yt8m_eval_util import EvaluationMetrics, CrossEntropyLoss
import numpy as np
import time
import os


class SubclassCnnTf:
    """ SubclassCnnTf: A Subclass CNN in Tensorflow.
    """

    def __init__(self,
                 num_classes= None,
                 num_subclasses=None,
                 subclass_labelling_fname=None,
                 top_k = 20,
                 reg_pen= 1e-3,
                 base_learning_rate= 0.01,
                 learning_rate_decay_examples= 4000000,
                 learning_rate_decay= 0.95,
                 model_save_path = None,
                 total_num_observations= None,
                 ):
        """ __init__: Initialization function for the subclass CNN.
            :param num_classes: target class label (not index)
            :param num_subclasses: batch size
            :param subclass_labelling_fname: filename of array containing the subclass information
            :param top_k: top k observations to be used in the GAP calculation
            :param reg_pen: regularization penalty
            :param base_learning_rate: initial learning rate
            :param learning_rate_decay_examples: number of observations for decaying the learning rate
            :param learning_rate_decay: multiplier for decaying the learning rate
            :param model_save_path: name of path to save the learned network
            :param total_num_observations: total number of observations
        """
        self.num_classes = num_classes # number of classes
        self.num_subclasses = num_subclasses # number of su
        self.subclass_labelling_fname = subclass_labelling_fname
        self.top_k = top_k
        self.reg_pen = reg_pen
        self.base_learning_rate = base_learning_rate
        self.learning_rate_decay_examples = learning_rate_decay_examples
        self.learning_rate_decay = learning_rate_decay
        self.total_num_observations = total_num_observations
        self.model_save_path = model_save_path
        self.model_save_pathname = None # to be defined when saving the model

    def fit(self,
            trainDataPattern=None,
            numEpochs=2,
            batchSz= 512,
            shuffleBufSz=10000,
            numPrlCalls=4,
            shuffleData= True,
            max_steps= 10000000):
        """ fit: Fit the optimization graph using a training dataset
            :param trainDataPattern: file glob for the training dataset
            :param numEpochs: number of epochs
            :param batchSz: batch size
            :param shuffleBufSz: buffer size for suffling batch observations
            :param numPrlCalls: number of parallel calls
            :param shuffleData: boolean, indicating to shuffle or not the data
            :param max_steps:
        """

        graph_learn_w = tf.Graph()
        with graph_learn_w.as_default():

            """
            Build graph for training
            """
            trainingIter, fea_batch, labels_batch, id_batch = youTube8mAudioVisualIter(
                train_data_pattern=trainDataPattern,
                shuffle_data= shuffleData,
                num_classes= self.num_subclasses,
                num_parallel_calls= numPrlCalls,
                shuffle_buffer_size= shuffleBufSz,
                batch_size= batchSz,
                num_epochs= numEpochs
            )

            num_batches_per_epoch = self.total_num_observations // batchSz

            print("Number of batches per epoch: {}".format(str(num_batches_per_epoch)) )

            reporting_step = num_batches_per_epoch // 100
            reporting_step = 100

            fea_batch_noise = tf.random.normal(shape=tf.shape(fea_batch), mean=0.0, stddev=0.03)
            fea_batch = tf.math.add(fea_batch, fea_batch_noise)
            fea_batch_normed = tf.nn.l2_normalize(fea_batch, 1)
            fea_batch_normed_expanded = tf.expand_dims(fea_batch_normed, 2)
            weight_decay = 0.005

            # define the layers
            Conv1D_1_1 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same',
                                                activation=tf.keras.layers.Activation('relu'),
                                                kernel_regularizer=tf.keras.regularizers.l2(weight_decay))

            LayerDo03 = tf.keras.layers.Dropout(0.3)
            MaxPooling1d_1 = tf.keras.layers.MaxPooling1D(pool_size=2, strides=None)

            LayerFlatten = tf.keras.layers.Flatten()
            LayerDense_C = tf.keras.layers.Dense(units=self.num_subclasses, activation='sigmoid',
                                                 kernel_regularizer=tf.keras.regularizers.l2(
                                                     weight_decay))

            # build the graph
            conv1_1 = Conv1D_1_1(fea_batch_normed_expanded)
            dout1 = LayerDo03(conv1_1)
            maxpooling1 = MaxPooling1d_1(dout1)

            flat6 = LayerFlatten(maxpooling1)
            pred_batch = LayerDense_C(flat6)

            labels_batch_loss = CrossEntropyLoss(pred_batch, labels_batch)
            total_batch_loss = tf.reduce_mean(labels_batch_loss)

            global_step = tf.Variable(0, trainable=False, name="global_step")
            learning_rate = tf.train.exponential_decay(
                self.base_learning_rate,
                global_step * batchSz,
                self.learning_rate_decay_examples,
                self.learning_rate_decay,
                staircase=True)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(total_batch_loss, global_step=global_step)

            """
            # optimization of graph variables #
            """
            var_init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            with tf.Session(graph=graph_learn_w) as sess:

                sess.run(var_init_op)
                sess.run(trainingIter.initializer)

                # initialize variables for housekeeping
                global_step_val = 0
                overall_start_time = time.time()
                period_loss_val = 0.

                reporting_start_time = overall_start_time

                while True:

                    try:
                        """
                        # using current batch, perform an optimization step to update graph variables #
                        """
                        period_start_time = time.time()  # timing

                        train_op_val, global_step_val, total_batch_loss_val, labels_batch_loss_val, pred_batch_val, labels_batch_val = sess.run(
                            [train_op, global_step, total_batch_loss, labels_batch_loss, pred_batch, labels_batch])

                        period_loss_val += total_batch_loss_val

                        if global_step_val % reporting_step == 0: # recording each epoch
                            print("Training period: step {} | Avg. Loss per batch: {:02.5f} | duration {}".format(
                                str(global_step_val), period_loss_val / reporting_step, time.time() - reporting_start_time))

                            reporting_start_time = time.time() # reset
                            period_loss_val = 0.

                        if global_step_val >= max_steps:
                            break

                    except tf.errors.OutOfRangeError:
                        break


                """
                # optimization finished - housekeeping #
                """
                print("Training finished: duration {} | training step {} | Loss: {:02.5f}".format(
                    time.time() - overall_start_time, str(global_step_val), period_loss_val))

                saver = tf.train.Saver()
                os.makedirs(self.model_save_path, exist_ok=True)
                save_path = saver.save(sess=sess, save_path=os.path.join(self.model_save_path, 'model.ckpt'),
                                       global_step=global_step_val)
                self.model_save_pathname = save_path
                print("Model saved in path: %s" % self.model_save_pathname)


        return self

    def predict(self, dataPattern=None, numPrlCalls= 4, batchSz = 2048):

        epoch_info_dict = None
        iteration_info_dict = None

        graph_transform = tf.Graph()

        with graph_transform.as_default():  # build model into the default Graph (tensorflow world)

            dataIter, fea_batch, labels_batch, id_batch = youTube8mAudioVisualIter(train_data_pattern=dataPattern,
                                                           shuffle_data=False,
                                                           num_classes=self.num_subclasses,
                                                           num_parallel_calls=numPrlCalls,
                                                           shuffle_buffer_size=1,
                                                           batch_size=batchSz,
                                                           num_epochs=1
                                                           )

            fea_batch_normed = tf.nn.l2_normalize(fea_batch, 1)
            fea_batch_normed_expanded = tf.expand_dims(fea_batch_normed, 2)
            weight_decay = 0.005

            # define the layers
            Conv1D_1_1 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same',
                                                activation=tf.keras.layers.Activation('relu'),
                                                kernel_regularizer=tf.keras.regularizers.l2(weight_decay))

            LayerDo03 = tf.keras.layers.Dropout(0.3)
            MaxPooling1d_1 = tf.keras.layers.MaxPooling1D(pool_size=2, strides=None)


            LayerFlatten = tf.keras.layers.Flatten()
            LayerDense_C = tf.keras.layers.Dense(units=self.num_subclasses, activation='sigmoid',
                                                 kernel_regularizer=tf.keras.regularizers.l2(
                                                     weight_decay))

            # build the graph
            conv1_1 = Conv1D_1_1(fea_batch_normed_expanded)
            dout1 = LayerDo03(conv1_1)
            maxpooling1 = MaxPooling1d_1(dout1)
            flat6 = LayerFlatten(maxpooling1)
            pred_batch = LayerDense_C(flat6)
            labels_batch_loss = CrossEntropyLoss(pred_batch, labels_batch)
            total_batch_loss = tf.reduce_mean(labels_batch_loss)
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

            saver = tf.train.Saver()

            with tf.Session(graph=graph_transform) as sess:
                sess.run(init_op)
                sess.run(dataIter.initializer)
                saver.restore(sess, self.model_save_pathname)
                print("Model restored from file: %s" % self.model_save_pathname)

                # restore subclass to class information
                subclassFnameNpz = np.load(self.subclass_labelling_fname, allow_pickle=True)
                class2subclassDict = subclassFnameNpz['class2subclassDict']
                subclass2classLbl = subclassFnameNpz['subclass2classLbl']
                class2subclassDict = class2subclassDict.item() # recover the dict

                start_time = time.time()
                step = 0

                evl_metrics = EvaluationMetrics(num_class=self.num_classes, top_k=self.top_k, top_n=None)
                evl_metrics.clear()

                while True:
                    try:  # gather and project data
                        step += 1

                        total_batch_loss_val, labels_batch_loss_val, pred_batch_val, labels_batch_val = sess.run(
                            [total_batch_loss, labels_batch_loss, pred_batch, labels_batch])

                        # project to class info
                        addedClasses = self.num_subclasses - self.num_classes
                        labels_batch_val = labels_batch_val[:, :-addedClasses]

                        for n in range(labels_batch_val.shape[0]):
                            for classLbl, subclassLbl in class2subclassDict.items():
                                pred_batch_val[n, classLbl] = np.max([pred_batch_val[n, classLbl] , pred_batch_val[n, subclassLbl]])

                        pred_batch_val = pred_batch_val[:, :-addedClasses]

                        iteration_info_dict = evl_metrics.accumulate(pred_batch_val, labels_batch_val, labels_batch_loss_val)

                        if step % 100 == 0:
                            print("Step {}".format(str(step)))

                    except tf.errors.OutOfRangeError:
                        duration_prj = time.time() - start_time
                        epoch_info_dict = evl_metrics.get()
                        evl_metrics.clear()
                        print("Duration: {} secs".format(duration_prj))

                        break

        return epoch_info_dict
