import os
import tensorflow as tf
import numpy as np
from cnn.alexnet import AlexNet
from tensorflow.contrib.data import Iterator
from tensorflow.python.client.session import BaseSession
from utils import make_list, get_init_op
from datagenerator import ImageDataGenerator
from scipy.spatial.distance import cdist


class NaiveOneNearestNeighborScorer:
    def __init__(self, folder_real, folder_generated, session: BaseSession, dir_for_list):
        self.folder0 = folder_real
        self.folder1 = folder_generated
        self.session = session
        self.dir_for_list = dir_for_list
        self._make_dir_for_list()
        self._latent = None
        self._pair_dist = None
        self._argmin = None
        self._score = None

    def _make_dir_for_list(self):
        try:
            os.makedirs(self.dir_for_list)
        except FileExistsError as e:
            print(e)
            print("abort making dir")

    def _set_latent(self):
        txt_path, length = make_list([self.folder1, self.folder0], [1, 0], [-1, -1], 'val', self.dir_for_list)
        print(txt_path, length)
        data = ImageDataGenerator(txt_path, 'inference', length, 2, shuffle=False)  # Do not shuffle the dataset
        iterator = Iterator.from_structure(data.data.output_types, data.data.output_shapes)  # type: Iterator
        next_batch = iterator.get_next()
        init_op = get_init_op(iterator, data)

        self.session.run(init_op)
        image_batch, label_batch = self.session.run(next_batch)
        # reshape the latent numpy array
        self._latent = np.reshape(image_batch, [image_batch.shape[0], -1])

    @property
    def latent(self):
        if self._latent is None:
            self._set_latent()
        return self._latent

    def _set_pair_dist(self):
        if self._latent is None:
            self._set_latent()
        self._pair_dist = cdist(self._latent, self._latent, metric="euclidean")
        np.fill_diagonal(self._pair_dist, np.inf)

    @property
    def pair_dist(self):
        if self._pair_dist is None:
            self._set_pair_dist()
        return self._pair_dist

    def _set_argmin(self):
        if self._pair_dist is None:
            self._set_pair_dist()
        self._argmin = self._pair_dist.argmin[0]

    @property
    def argmin(self):
        if self._argmin is None:
            self._set_argmin()
        return self._argmin

    def _set_score(self):
        if self._argmin is None:
            self._set_argmin()
        length = len(self._argmin)
        total = sum(1 for k in range(length) if (k < length / 2) == (self._argmin[k] < length / 2))
        self._score = total / length

    @property
    def score(self):
        if self._score is None:
            self._set_score()
        return self._score


class AlexNetOneNearestNeighborScorer(NaiveOneNearestNeighborScorer):
    def __int__(self, folder_real, folder_generated, session: BaseSession, dir_for_list, alexnet=None):
        NaiveOneNearestNeighborScorer.__init__(self, folder_real, folder_generated, session, dir_for_list)
        if alexnet is None:
            self._alexnet = None  # declare field in constructor to avoid warnings
            self._set_default_alexnet()
        else:
            self._alexnet = alexnet

    def _set_latent(self):
        txt_path, length = make_list([self.folder1, self.folder0], [1, 0], [-1, -1], 'val', self.dir_for_list)
        print(txt_path, length)
        data = ImageDataGenerator(txt_path, 'inference', length, 2, shuffle=False)  # Do not shuffle the dataset
        iterator = Iterator.from_structure(data.data.output_types, data.data.output_shapes)  # type: Iterator
        next_batch = iterator.get_next()
        init_op = get_init_op(iterator, data)

        # get the latent_tsr representation of each sample
        latent_tsr = alexnet.flattened
        keep_prob = 1.0

        self.session.run(init_op)
        image_batch, label_batch = self.session.run(next_batch)
        self._latent = self.session.run(latent_tsr, feed_dict={x_tsr: image_batch, keep_prob_tsr: keep_prob})

    def _set_default_alexnet(self):
        x_tsr = tf.placeholder(tf.float32, [None, 227, 227, 3])
        keep_prob_tsr = tf.placeholder(tf.float32, tuple())
        num_classes = 2
        train_layers = ['fc8']
        self._alexnet = AlexNet(x_tsr, keep_prob_tsr, num_classes, train_layers)
        # load model
        self.session.run(tf.global_variables_initializer())
        self._alexnet.load_model_pretrained(self.session)

    @property
    def alexnet(self):
        if self._alexnet is None:
            self._set_default_alexnet()
        return self._alexnet


if __name__ == '__main__':
    x_tsr = tf.placeholder(tf.float32, [None, 227, 227, 3])
    keep_prob_tsr = tf.placeholder(tf.float32, tuple())
    num_classes = 2
    train_layers = ['fc8']
    alexnet = AlexNet(x_tsr, keep_prob_tsr, num_classes, train_layers)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    alexnet.load_model_pretrained(sess)

    real_folder = "images/real-images/"
    reuse = True
