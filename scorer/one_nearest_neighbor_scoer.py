import tensorflow as tf
import numpy as np
from cnn.alexnet import AlexNet
from tensorflow.python.client.session import BaseSession
from scipy.spatial.distance import cdist
from .stats_scorer import StatsScorer


class NaiveOneNearestNeighborScorer(StatsScorer):
    def __init__(self, images_real, images_fake):
        super(NaiveOneNearestNeighborScorer, self).__init__(images_real, images_fake)
        self._latent = None
        self._pair_dist = None
        self._argmin = None
        self._score = None

    def _set_latent(self):
        latent0 = self._flatten(self._convert_to_array(self._images0))
        latent1 = self._flatten(self._convert_to_array(self._images1))
        if latent0.shape != latent1.shape:
            raise ValueError("real and fake latents differ in shape {} != {}".format(latent0.shape, latent1.shape))
        self._latent = np.concatenate([latent0, latent1])

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
    def __int__(self, images_real, images_fake, session: BaseSession, dir_for_list, alexnet=None):
        NaiveOneNearestNeighborScorer.__init__(self, images_real, images_fake)
        self.session = session
        if alexnet is None:
            self._alexnet = None  # declare field in constructor to avoid warnings
            self._set_default_alexnet()
        else:
            self._alexnet = alexnet
        self._tf_init()

    def _tf_init(self):
        self.session.run(tf.global_variables_initializer())

    def _set_latent(self):
        # equivalent to setting the initial values for self._latent
        NaiveOneNearestNeighborScorer._set_latent(self)
        # self._latent = some_input_images in np.array format

        # TODO allow for specifying which latent layer to use, default using `flattened`, i.e., the layer after conv5
        # grab the latent_tsr representation of each sample
        latent_tsr = self._alexnet.flattened

        self._latent = self.session.run(
            latent_tsr,
            feed_dict={
                self._alexnet.X: self._latent,
                self._alexnet.KEEP_PROB: 1.0,
            }
        )
        # in case latent_tsr is not flattened
        self._latent = self._flatten(self._latent)

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
