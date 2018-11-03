import tensorflow as tf
from .base import BaseModel
from .alexnet import AlexNet


class SiameseAlexNet(BaseModel):
    def __init__(self, x1, x2, keep_prob, num_classes, train_layers, name_scope="Siamese", proj="flattened",
                 falpha=2.0, margin00=3.5, margin01=7.0, margin11=8.0, weights_path='/pretrained/bvlc_alexnet.npy',
                 punish00=1.0, punish11=1.0, punish01=5.0):
        super(SiameseAlexNet, self).__init__()
        self.name_scope = name_scope
        self.margin00 = margin00
        self.margin01 = margin01
        self.margin11 = margin11
        self.punish00 = punish00
        self.punish11 = punish11
        self.punish01 = punish01
        self.proj = proj
        with tf.variable_scope(self.name_scope) as scope:
            self.net1 = AlexNet(x1, keep_prob, num_classes, train_layers, falpha=falpha, weights_path=weights_path)
            scope.reuse_variables()
            self.net2 = AlexNet(x2, keep_prob, num_classes, train_layers, falpha=falpha, weights_path=weights_path)
            # define a loss for Siamese Network
            self._create_loss(proj)

    def _create_loss(self, proj):
        # XXX punishing the Pos-Neg loss harder than Pos-Pos loss and Neg-Neg loss to avoid underfitting
        proj1, proj2 = self._get_projections(proj)
        eucd2 = tf.reduce_mean((proj1 - proj2) ** 2, axis=1, name="euclidean_dist_squared")
        eucd = tf.sqrt(eucd2, name="euclidean_dist")
        print('euclidean distances tensor', eucd)
        # y1, y2 and y_cmp should be wrapped instead of being a class member
        y1 = tf.cast(tf.argmax(self.net1.y, axis=1), tf.float32, name='siam-y1')
        y2 = tf.cast(tf.argmax(self.net2.y, axis=1), tf.float32, name='siam-y2')
        self.y1_label, self.y2_label = y1, y2
        y_diff = tf.cast(y1 - y2, tf.bool, name="comparison_label_in_tf.bool")
        y_diff = tf.cast(y_diff, tf.float32, name="comparison_label_in_tf.float32")
        self.count01 = tf.reduce_sum(y_diff, name='count01')
        self.count00 = tf.reduce_sum((1 - y1) * (1 - y2), name='count00')
        self.count11 = tf.reduce_sum(y1 * y2, name='count11')

        # if label1 and label2 are the same, y_diff = 0, punish the part where eucd exceeds margin
        loss00 = tf.reduce_mean(((1 - y1) * (1 - y2) * tf.nn.relu(eucd - self.margin00)) ** 2, axis=0, name='loss00')
        loss11 = tf.reduce_mean((y1 * y2 * tf.nn.relu(eucd - self.margin11)) ** 2, axis=0, name='loss11')
        self.mean_dist00 = tf.reduce_sum((1 - y1) * (1 - y2) * eucd) / self.count00
        self.mean_dist11 = tf.reduce_sum(y1 * y2 * eucd) / self.count11

        # if label1 and label2 are different, y_diff = 1, punish the part where eucd falls short of margin
        loss01 = tf.reduce_mean((y_diff * tf.nn.relu(self.margin01 - eucd)) ** 2, axis=0, name='loss01')
        self.mean_dist01 = tf.reduce_sum(y_diff * eucd) / self.count01

        self.loss00 = loss00 * self.punish00
        self.loss01 = loss01 * self.punish01
        self.loss11 = loss11 * self.punish11
        self.loss = tf.add(self.loss00 + self.loss11, self.loss01, name="siamese-loss")
        print(self.loss)

    def _get_projections(self, proj):
        print('projection =', proj, "type=", type(proj))
        projections = (self.net1.dropout6, self.net2.dropout6)
        try:
            if proj == "fc6":
                projections = (self.net1.fc6, self.net2.fc6)
            elif proj == "fc7":
                projections = (self.net1.fc7, self.net2.fc7)
            elif proj == "fc8":
                projections = (self.net1.fc8, self.net2.fc8)
            elif proj == "dropout6":
                projections = (self.net1.dropout6, self.net2.dropout6)
            elif proj == "dropout7":
                projections = (self.net1.dropout7, self.net2.dropout7)
            elif proj == "flattened":
                projections = (self.net1.flattened, self.net2.flattened)
            else:
                raise ValueError("Illegal Projection: " + proj)
        except ValueError as e:
            print("ValueError: encountered in _get_predictions")
            print(e)
        finally:
            print("projections of %s are " % self.name_scope, projections[0].name, projections[1].name)
            print("dimensions of projection is", projections[0].shape, projections[1].shape)
            return projections

    def load_model_pretrained(self, session):
        with tf.variable_scope(self.name_scope, reuse=True):
            self.net1.load_model_pretrained(session)

    def load_model_vars(self, path: str, session):
        with tf.variable_scope(self.name_scope, reuse=True):
            self.net1.load_model_vars(path, session)

    def save_model_vars(self, path: str, session, init=False):
        with tf.variable_scope(self.name_scope):
            self.net1.save_model_vars(path, session, init=init)

    def get_model_vars(self, session, init=False):
        with tf.variable_scope(self.name_scope):
            return self.net1.get_model_vars(session, init=init)

    def set_model_vars(self, variable_dict, session):
        with tf.variable_scope(self.name_scope):
            return self.net1.set_model_vars(variable_dict, session)

    # return a new instance of AlexNet with trainable variables
    def get_net_copy(self, session, x=None, keep_prob=None, num_classes=None, train_layers=None, falpha=None,
                     weights_path=None) -> AlexNet:
        if x is None:
            x = self.net1.X
            print("Warning: x_tsr should be specified as a new placeholder")
        if keep_prob is None:
            keep_prob = self.net1.KEEP_PROB
        if num_classes is None:
            num_classes = self.net1.NUM_CLASSES
        if train_layers is None:
            train_layers = self.net1.TRAIN_LAYERS
            print("Warning: train_layers should be specified as a new list of layer names")
        if falpha is None:
            falpha = self.net1.ALPHA
        if weights_path is None:
            weights_path = self.net1.WEIGHTS_PATH
        new_net = AlexNet(x, keep_prob, num_classes, train_layers, falpha=falpha, weights_path=weights_path)
        new_net.set_model_vars(self.get_model_vars(session), session)
        return new_net


if __name__ == "__main__":
    # how the two nets in Siamese Net share the keep_prob_tsr placeholder?
    # the keep_prob_tsr argument passed to constructor is an integer, instead of a placeholder
    keep_prob = tf.placeholder(tf.float32, [], name='keep_prob_tsr')
    x = tf.placeholder(tf.float32, [None, 227, 227, 3], name='x_tsr')
    x1 = tf.placeholder(tf.float32, [None, 227, 227, 3], name='x1')
    x2 = tf.placeholder(tf.float32, [None, 227, 227, 3], name='x2')
    # image_batch = np.random.rand(5, 227, 227, 3)
    # label_batch = np.random.rand(5, 1000)
    net = AlexNet(x, keep_prob, 2, ['fc6', 'fc7'])
    # net = SiameseAlexNet(x1, x2, 0.5, 3, ['fc6', 'fc7', 'fc8'], name_scope="SiameseA", proj="flattened")
    # netB = SiameseAlexNet(x1, x2, 0.5, 3, ['fc6', 'fc7', 'fc8'], name_scope="SiameseB")
    # check_path = "/Users/liushuheng/Desktop/vars.npy"
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     net.load_model_pretrained(sess)
    # y1 = sess.run(netA.net1.y, feed_dict={netA.net1.X: image_batch, netA.net1.y: label_batch})
    # y2 = sess.run(netB.net1.y, feed_dict={netB.net1.X: image_batch, netB.net1.y: label_batch})
    # netA.save_model_vars(check_path, sess)
    # netB.load_model_vars(check_path, sess)
    # y3 = sess.run(netB.net1.y, feed_dict={netB.net1.X: image_batch, netB.net1.y: label_batch})
    # assert (y1 == y2).all(), "assertion1 failed"
    # print("assertion1 passed")
    # assert (y1 == y3).all(), "assertion2 failed"
    # print("assertion2 passed")
    # d = net.get_model_vars(sess)
    # init_weights = np.load("/pretrained/bvlc_alexnet.npy", encoding="bytes").item()

    # for var in tf.global_variables():
    # # for var in tf.get_default_graph().get_operations():
    #     print(var.name, end=" ")
