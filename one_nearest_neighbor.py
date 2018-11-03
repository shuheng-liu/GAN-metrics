import os
import tensorflow as tf
import numpy as np
import pickle as pkl
from alexnet import AlexNet
from tensorflow.contrib.data import Iterator
from tensorflow.python.client.session import BaseSession
from utils import make_list, get_init_op
from datagenerator import ImageDataGenerator
from scipy.spatial.distance import cdist


class OneNearestNeighborScorer:
    def __init__(self, folder_real, folder_generated, session: BaseSession, dump_dir):
        self.folder0 = folder_real
        self.folder1 = folder_generated
        self.session = session
        self.dump_dir = dump_dir
        self.latent_path = os.path.join(dump_dir, "latent.pkl")
        self._make_dump_dir()
        self._latent = None
        self._pair_dist = None
        self._argmin = None
        self._score = None

    def _make_dump_dir(self):
        try:
            os.makedirs(self.dump_dir)
        except FileExistsError as e:
            print(e)
            print("abort making dir")

    def _set_latent(self):
        txt_path, length = make_list([self.folder1, self.folder0], [1, 0], [-1, -1], 'val', self.dump_dir)
        print(txt_path, length)
        data = ImageDataGenerator(txt_path, 'inference', length, 2, shuffle=False)  # Do not shuffle the dataset
        iterator = Iterator.from_structure(data.data.output_types, data.data.output_shapes)  # type: Iterator
        next_batch = iterator.get_next()
        init_op = get_init_op(iterator, data)

        self.session.run(init_op)
        image_batch, label_batch = sess.run(next_batch)
        # reshape the latent numpy array
        self._latent = np.reshape(image_batch, [image_batch.shape[0], -1])

    def get_latent(self):
        if self._latent is None:
            self._set_latent()
        return self._latent

    latent = property(get_latent)

    def get_pair_dist(self):
        return self._pair_dist

    pair_dist = property(get_pair_dist)

    def get_argmin(self):
        return self._argmin

    argmin = property(get_argmin)


def get_score(self):
    return self._score


score = property(get_score)


def get_naive_latent_from_folders(real_folder, generated_folder, sess, dump_dir, reuse=False):
    latent_path = os.path.join(dump_dir, "latent.pkl")
    if reuse and os.path.isfile(latent_path):
        with open(latent_path, "rb") as f:
            latent = pkl.load(f)
    else:
        txt_path, length = make_list([real_folder, generated_folder], [1, 0], [-1, -1], 'val', dump_dir)
        print(txt_path, length)
        data = ImageDataGenerator(txt_path, 'inference', length, 2, shuffle=False)  # Do not shuffle the dataset
        iterator = Iterator.from_structure(data.data.output_types, data.data.output_shapes)  # type: Iterator
        next_batch = iterator.get_next()
        init_op = get_init_op(iterator, data)

        sess.run(init_op)
        image_batch, label_batch = sess.run(next_batch)
        # reshape the latent numpy array
        latent = np.reshape(image_batch, [image_batch.shape[0], -1])

        # dump the representation on disk
        with open(latent_path, "wb") as f:
            pkl.dump(latent, f)

    return latent


def get_latent_from_folders(real_folder, generated_folder, alexnet, sess, dump_dir, reuse=False):
    latent_path = os.path.join(dump_dir, "latent.pkl")
    if reuse and os.path.isfile(latent_path):
        with open(latent_path, "rb") as f:
            latent = pkl.load(f)
    else:
        # get the real samples and generated samples
        txt_path, length = make_list([real_folder, generated_folder], [1, 0], [-1, -1], 'val', dump_dir)
        print(txt_path, length)
        data = ImageDataGenerator(txt_path, 'inference', length, 2, shuffle=False)  # Do not shuffle the dataset
        iterator = Iterator.from_structure(data.data.output_types, data.data.output_shapes)  # type: Iterator
        next_batch = iterator.get_next()
        init_op = get_init_op(iterator, data)

        # get the latent_tsr representation of each sample
        latent_tsr = alexnet.flattened
        keep_prob = 1.0

        sess.run(init_op)
        image_batch, label_batch = sess.run(next_batch)
        latent = sess.run(latent_tsr, feed_dict={x_tsr: image_batch, keep_prob_tsr: keep_prob})

        # dump the representation on disk
        with open(latent_path, "wb") as f:
            pkl.dump(latent, f)

    return latent


def get_pair_dist_from_latent(latent, dump_dir, reuse=False):
    # get the pair-wise distance and dump it
    pair_path = os.path.join(dump_dir, "pair_dist.pkl")
    if reuse and os.path.isfile(pair_path):
        with open(pair_path, "rb") as f:
            pair_dist = pkl.load(f)
    else:
        pair_dist = cdist(latent, latent, metric="euclidean")
        with open(pair_path, "wb") as f:
            pkl.dump(pair_dist, f)

    return pair_dist


def get_argmin_from_pair_dist(pair_dist, dump_dir, reuse=False):
    # get the Leave One Out 1-NN result and dump it
    argmin_path = os.path.join(dump_dir, "argmin.pkl")
    if reuse and os.path.isfile(argmin_path):
        with open(argmin_path, "rb") as f:
            argmin = pkl.load(f)
    else:
        np.fill_diagonal(pair_dist, np.inf)
        argmin = pair_dist.argmin(0)
        with open(argmin_path, "wb") as f:
            pkl.dump(argmin, f)
    return argmin


def get_score_from_argmin(argmin):
    length = len(argmin)  # in case length is not defined
    total = sum(1 for k in range(length) if (k < length / 2) == (argmin[k] < length / 2))
    return total / length


def get_score_from_folder(real_folder, generated_folder, alexnet, sess, dump_dir, reuse=False, naive=False):
    try:
        os.makedirs(dump_dir)
    except FileExistsError:
        pass
    if naive:
        latent = get_naive_latent_from_folders(real_folder, generated_folder, sess, dump_dir, reuse=reuse)
    else:
        latent = get_latent_from_folders(real_folder, generated_folder, alexnet, sess, dump_dir, reuse=reuse)
    pair_dist = get_pair_dist_from_latent(latent, dump_dir, reuse=reuse)
    argmin = get_argmin_from_pair_dist(pair_dist, dump_dir, reuse=reuse)
    score = get_score_from_argmin(argmin)
    return score


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

    without_mute_layer = []
    for epoch in range(10, 1001, 10):
        print("%.1f percent completed" % (epoch / 20))
        generated_folder = "images/fake-images-97/%d" % epoch
        dump_dir = "naive-output-euclid/output-97/%d" % epoch
        score = get_score_from_folder(real_folder, generated_folder, alexnet,
                                      sess, dump_dir, reuse=reuse, naive=True)
        without_mute_layer.append(score)
    print(without_mute_layer)

    with_mute_layer = []
    for epoch in range(10, 1001, 10):
        print("%.1f percent completed" % (epoch / 20 + 50))
        generated_folder = "images/fake-images-120/%d" % epoch
        dump_dir = "naive-output-euclid/output-120/%d" % epoch
        score = get_score_from_folder(real_folder, generated_folder, alexnet,
                                      sess, dump_dir, reuse=reuse, naive=True)
        with_mute_layer.append(score)
    print(with_mute_layer)
