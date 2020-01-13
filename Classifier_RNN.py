import tensorflow as tf
import tensorboard as tboard
import warnings
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.python.framework import ops
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics.classification import accuracy_score


# ignore all of the tensorflow warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
ops.reset_default_graph()


class LSTM_RNN(object):
    """
    The double underscore __ before variables/functions show that they are private names for the class and aren't
    meant to be referenced outside of the class. This is standard form.
    """
    def __init__(self, hidden_size, vocab_size, embedding_size, n_classes, max_len=None, learning_rate=0.01, rand_state=24):
        self.input = self.__input(max_len)
        self.seq_len = self.__seq_len()
        self.target = self.__target(n_classes)
        self.dropout_keep_prob = self.__dropout_keep_prob()
        self.word_embeddings = self.__word_embeddings(self.input, vocab_size, embedding_size, random_state)
        self.scores = self.__scores(self.word_embeddings, self.seq_len, hidden_size, n_classes, self.dropout_keep_prob,
                                    random_state)

        self.predict = self.__predict(self.scores)
        self.losses = self.__losses(self.scores, self.target)
        self.loss = self.__loss(self.losses)
        self.train_step = self.__train_step(learning_rate, self.loss)
        self.accuracy = self.__accuracy(self.predict, self.target)
        self.merged = tf.summary.merge_all()

    def __input(self, max_length):
        """
        :param max_len: max length on input tensor
        :return: input placeholder [batch_size, max_len]
        """
        return tf.placeholder(tf.int32, [None, max_length], name='input')

    def __seq_len(self):
        """
        allows dynamic sequence length
        :return:
        """
        return tf.placeholder(tf.int32, [None], name='lengths')

    def __target(self, n_classes):
        return tf.placeholder(tf.float32, [None, n_classes], name='target')

    def __dropout_keep_prob(self):
        return tf.placeholder(tf.float32, name='dropout_keep_prob')

    def __cell(self, hidden_size, dropout_keep_prob, seed=24):
        lstm_cell = tf.nn.rnn_cell.LSTMCell(hidden_size, state_is_tuple=True)
        dropout_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, input_keep_prob=dropout_keep_prob,
                                                     output_keep_prob=dropout_keep_prob, seed=seed)
        return dropout_cell

    def __word_embeddings(self, x, vocab_size, embedding_size, seed=None):
        with tf.name_scope('word_embeddings'):
            embeddings = tf.get_variable("embeddings", shape=[vocab_size, embedding_size], dtype=tf.float32, initializer=None, regularizer=None, trainable=True, collections=None)
            embedded_words = tf.nn.embedding_lookup(embeddings, x)
        return embedded_words

    def __rnn_layer(self, hidden_size, x, seq_len, dropout_keep_prob, variable_scope=None, random_state=24):
        with tf.variable_scope(variable_scope, default_name='rnn_layer'):
            lstm_cell = self.__cell(hidden_size, dropout_keep_prob, random_state)
            outputs, _ = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32, sequence_length=seq_len)

        return outputs

    def __scores(self, embedded_words, seq_len, hidden_size, n_classes, dropout_kepp_prob, random_state=None):
        outputs = embedded_words
        for h in hidden_size:
            outputs = self.__rnn_layer(h, outputs, seq_len, dropout_kepp_prob)

        outputs = tf.reduce_mean(outputs, axis=[1])

        with tf.name_scope('final_layer/weights'):
            w = tf.get_variable("w", shape=[hidden_size[-1], n_classes], dtype=tf.float32, initializer=None,
                                regularizer=None, trainable=True, collections=None)
            self.variable_summaries(w, 'final_layer/weights')
        with tf.name_scope('final_layer/biases'):
            b = tf.get_variable("b", shape=[n_classes], dtype=tf.float32, initializer=None, regularizer=None,
                                trainable=True, collections=None)
            self.variable_summaries(b, 'final_layer/biases')
        with tf.name_scope('final_layer/wx_plus_b'):
            scores = tf.nn.xw_plus_b(outputs, w, b, name='scores')
            tf.summary.histogram('final_layer/wx_plus_b', scores)
        return scores

    def __predict(self, scores):
        """
        :param scores: Linear activation of each class with shape [batch_size, n_classes]
        :return: Softmax activations with shape [batch_size, n_classes]
        """
        with tf.name_scope('final_layer/softmax'):
            softmax = tf.nn.softmax(scores, name='predictions')
            tf.summary.histogram('final_layer/softmax', softmax)
        return softmax

    def __losses(self, scores, target):
        """
        :param scores: Linear activation of each class with shape [batch_size, n_classes]
        :param target: Target tensor with shape [batch_size, n_classes]
        :return: Cross entropy losses with shape [batch_size]
        """
        with tf.name_scope('cross_entropy'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=scores, labels=target, name='cross_entropy')
        return cross_entropy

    def __loss(self, losses):
        """
        :param losses: Cross entropy losses with shape [batch_size]
        :return: Cross entropy loss mean
        """
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(losses, name='loss')
            tf.summary.scalar('loss', loss)
        return loss

    def __train_step(self, learning_rate, loss):
        """
        :param learning_rate: Learning rate of RMSProp algorithm
        :param loss: Cross entropy loss mean
        :return: RMSProp train step operation
        """
        return tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

    def __accuracy(self, predict, target):
        """
        :param predict: Softmax activations with shape [batch_size, n_classes]
        :param target: Target tensor with shape [batch_size, n_classes]
        :return: Accuracy mean obtained in current batch
        """
        with tf.name_scope('accuracy'):
            correct_pred = tf.equal(tf.argmax(predict, 1), tf.argmax(target, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
            tf.summary.scalar('accuracy', accuracy)
        return accuracy

    def initialize_all_variables(self):
        """
        :return: Operation that initiallizes all variables
        """
        return tf.global_variables_initializer()

    @staticmethod
    def variable_summaries(var, name):
        """
        Attach a lot of summaries to a Tensor for Tensorboard visualization.
        Ref: https://www.tensorflow.org/versions/r0.11/how_tos/summaries_and_tensorboard/index.html
        :param var: Variable to summarize
        :param name: Summary name
        """
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean/' + name, mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev/' + name, stddev)
            tf.summary.scalar('max/' + name, tf.reduce_max(var))
            tf.summary.scalar('min/' + name, tf.reduce_min(var))
            tf.summary.histogram(name, var)