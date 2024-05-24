import tensorflow.compat.v1 as tf
import numpy as np

np.set_printoptions(threshold=np.inf)
import matplotlib
import matplotlib.pyplot as plt
import time
import os
import scipy.io as sio

np.random.seed(1)
tf.set_random_seed(1)
tf.compat.v1.disable_eager_execution()
a = 0.000001
b = 1
c = 0.000001
d = 1


###########定义网络
class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, layers_U, layers_V1, layers_V2, layers_V3, layers_V4, layers_V5, layers_W1, layers_W2, layers_W3,
                 layers_W4, layers_W5, x_range, t_range, f, num_train_tps):

        # Initialize NNs
        self.layers_U = layers_U
        self.weights_U, self.biases_U, self.adaps_U = self.initialize_NN(layers_U)
        self.layers_V1 = layers_V1
        self.weights_V1, self.biases_V1, self.adaps_V1 = self.initialize_NN(layers_V1)
        self.layers_V2 = layers_V2
        self.weights_V2, self.biases_V2, self.adaps_V2 = self.initialize_NN(layers_V2)
        self.layers_V3 = layers_V3
        self.weights_V3, self.biases_V3, self.adaps_V3 = self.initialize_NN(layers_V3)
        self.layers_V4 = layers_V4
        self.weights_V4, self.biases_V4, self.adaps_V4 = self.initialize_NN(layers_V4)
        self.layers_V5 = layers_V5
        self.weights_V5, self.biases_V5, self.adaps_V5 = self.initialize_NN(layers_V5)
        self.layers_W1 = layers_W1
        self.weights_W1, self.biases_W1, self.adaps_W1 = self.initialize_NN(layers_W1)
        self.layers_W2 = layers_W2
        self.weights_W2, self.biases_W2, self.adaps_W2 = self.initialize_NN(layers_W2)
        self.layers_W3 = layers_W3
        self.weights_W3, self.biases_W3, self.adaps_W3 = self.initialize_NN(layers_W3)
        self.layers_W4 = layers_W4
        self.weights_W4, self.biases_W4, self.adaps_W4 = self.initialize_NN(layers_W4)
        self.layers_W5 = layers_W5
        self.weights_W5, self.biases_W5, self.adaps_W5 = self.initialize_NN(layers_W5)
        # Parameters
        self.t_range = t_range
        self.x_range = x_range
        self.lb = np.array([x_range[0], t_range[0]])
        self.ub = np.array([x_range[1], t_range[1]])

        # Output file
        self.f = f

        self.tx_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.xx_tf = tf.placeholder(tf.float32, shape=[None, 1])
        # Test Points                           ######################测试点
        self.t_test_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.x_test_tf = tf.placeholder(tf.float32, shape=[None, 1])

        # Generate Training and Testing Points  ########################生成训练和测试点
        self.generateTrain(num_train_tps)

        # Physics
        # Interior (IDE)  #############PDE

        self.gf, self.f1, self.e1, self.e2, self.e3, self.e4, self.e5, self.e6, self.e7, self.e8, self.e9, self.e10, self.Uf, self.V1f, self.V2f, self.V3f, self.V4f, self.V5f, \
        self.W1f, self.W2f, self.W3f, self.W4f, self.W5f, self.U_xf, self.U_tf, self.U_ttf, self.V1_xf, self.V2_xf, self.V3_xf, self.V4_xf, self.V5_xf, \
        self.W1_tf, self.W2_tf, self.W3_tf, self.W4_tf, self.W5_tf = self.pinn(self.tt, self.xf)
        # chushi
        _, _, _, _, _, _, _, _, _, _, _, _, self.Ui, self.V1i, self.V2i, self.V3i, self.V4i, self.V5i, self.W1i, self.W2i, self.W3i, self.W4i, self.W5i, self.U_xi, self.U_ti, self.U_tti, \
        self.V1_xi, self.V2_xi, self.V3_xi, self.V4_xi, self.V5_xi, self.W1_ti, self.W2_ti, self.W3_ti, self.W4_ti, self.W5_ti = self.pinn(
            self.ti, self.xi)
        _, _, _, _, _, _, _, _, _, _, _, _, self.Ub, self.V1b, self.V2b, self.V3b, self.V4b, self.V5b, self.W1b, self.W2b, self.W3b, self.W4b, self.W5b, self.U_xb, self.U_tb, \
        self.U_ttb, self.V1_xb, self.V2_xb, self.V3_xb, self.V4_xb, self.V5_xb, self.W1_tb, self.W2_tb, self.W3_tb, self.W4_tb, self.W5_tb = self.pinn(
            self.tb, self.xb)
        _, _, _, _, _, _, _, _, _, _, _, _, self.Ub2, self.V1b2, self.V2b2, self.V3b2, self.V4b2, self.V5b2, self.W1b2, self.W2b2, self.W3b2, self.W4b2, self.W5b2, \
        self.U_xb2, self.U_tb2, self.U_ttb2, self.V1_xb2, self.V2_xb2, self.V3_xb2, self.V4_xb2, self.V5_xb2, self.W1_tb2, self.W2_tb2, self.W3_tb2, self.W4_tb2, self.W5_tb2 \
            = self.pinn(self.tb2, self.xb2)
        # Test
        _, self.f1_test, self.e1_test, self.e2_test, self.e3_test, self.e4_test, self.e5_test, self.e6_test, self.e7_test, self.e8_test, self.e9_test, self.e10_test, \
        self.U_test, self.V1_test, self.V2_test, self.V3_test, self.V4_test, self.V5_test, self.W1_test, self.W2_test, self.W3_test, self.W4_test, self.W5_test, \
        self.U_x_test, self.U_t_test, self.U_tt_test, self.V1_x_test, self.V2_x_test, self.V3_x_test, self.V4_x_test, self.V5_x_test, self.W1_t_test, self.W2_t_test, \
        self.W3_t_test, self.W4_t_test, self.W5_t_test = self.pinn(self.t_test_tf, self.x_test_tf)

        # Interior (IDE)
        self.loss_f = tf.reduce_mean((self.gf + self.tt ** 5 * (
                    self.V1f - self.V2f + self.V3f - self.V4f + self.V5f) + self.U_xf - self.U_tf - self.Uf - self.U_ttf) ** 2)
        self.loss_e1 = tf.reduce_mean((self.V1_xf - self.W1f) ** 2)
        self.loss_e2 = tf.reduce_mean((self.V2_xf - self.tt ** 2 * self.xf ** 2 * self.W2f * 0.5) ** 2)
        self.loss_e3 = tf.reduce_mean((self.V3_xf - (self.tt ** 4 * self.xf ** 4 * self.W3f) / 24) ** 2)
        self.loss_e4 = tf.reduce_mean((self.V4_xf - (self.tt ** 6 * self.xf ** 6 * self.W4f) / 720) ** 2)
        self.loss_e5 = tf.reduce_mean((self.V5_xf - (self.tt ** 8 * self.xf ** 8 * self.W5f) / 40320) ** 2)
        self.loss_e6 = tf.reduce_mean((self.W1_tf - self.Uf**2) ** 2)
        self.loss_e7 = tf.reduce_mean((self.W2_tf - self.tt ** 2 * self.Uf**2) ** 2)
        self.loss_e8 = tf.reduce_mean((self.W3_tf - self.tt ** 4 * self.Uf**2) ** 2)
        self.loss_e9 = tf.reduce_mean((self.W4_tf - self.tt ** 6 * self.Uf**2) ** 2)
        self.loss_e10 = tf.reduce_mean((self.W5_tf - self.tt ** 8 * self.Uf**2) ** 2)
        # 初始条件和边界条件
        self.loss_i = tf.reduce_mean((self.Ui - 0) ** 2) + tf.reduce_mean((self.U_ti - self.xb2) ** 2) + tf.reduce_mean(
            (self.W1i) ** 2) + \
                      tf.reduce_mean((self.W2i) ** 2) + tf.reduce_mean((self.W3i) ** 2) + tf.reduce_mean(
            (self.W4i) ** 2) + tf.reduce_mean((self.W5i) ** 2)
        self.loss_b1 = tf.reduce_mean((self.Ub - self.tb) ** 2) + tf.reduce_mean((self.V1b) ** 2) + tf.reduce_mean(
            (self.V2b) ** 2) + tf.reduce_mean((self.V3b) ** 2) \
                       + tf.reduce_mean((self.V4b) ** 2) + tf.reduce_mean((self.V5b) ** 2)
        self.loss_b2 = tf.reduce_mean((self.Ub2 - self.tb2 - self.tb2 ** 2) ** 2)

        # # Total Loss
        self.loss = self.loss_f * 10+ self.loss_e1 * 1 + self.loss_e2 * 1 + self.loss_e3 * 100\
                    + self.loss_e4 * 1000+self.loss_e5 * 100+ self.loss_e6 *0.1 + self.loss_e7 * 1 + \
                    self.loss_e8 * 0.1 + self.loss_e9 * 1 + self.loss_e10 * 1 + self.loss_i * 1 + \
                    self.loss_b1 * 1+ self.loss_b2 * 10

        # Optimizer
        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

        # tf session
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

    #####################初始化神经网络参数

    def initialize_NN(self, layers):
        weights = []
        biases = []
        adaps = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            a = tf.Variable(1.0, dtype=tf.float32)
            weights.append(W)
            biases.append(b)
            adaps.append(a)
        return weights, biases, adaps

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def net_U(self, X):
        weights = self.weights_U
        biases = self.biases_U
        adaps = self.adaps_U
        num_layers = len(weights) + 1
        h = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            a = adaps[l]
            h = tf.tanh(tf.multiply(a, tf.add(tf.matmul(h, W), b)))
        W = weights[-1]
        b = biases[-1]
        U = tf.add(tf.matmul(h, W), b)
        return U

    def net_V1(self, X):
        weights = self.weights_V1
        biases = self.biases_V1
        adaps = self.adaps_V1
        num_layers = len(weights) + 1
        h = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            a = adaps[l]
            h = tf.tanh(tf.multiply(a, tf.add(tf.matmul(h, W), b)))
        W = weights[-1]
        b = biases[-1]
        V1 = tf.add(tf.matmul(h, W), b)
        return V1

    def net_V2(self, X):
        weights = self.weights_V2
        biases = self.biases_V2
        adaps = self.adaps_V2
        num_layers = len(weights) + 1
        h = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            a = adaps[l]
            h = tf.tanh(tf.multiply(a, tf.add(tf.matmul(h, W), b)))
        W = weights[-1]
        b = biases[-1]
        V2 = tf.add(tf.matmul(h, W), b)
        return V2

    def net_V3(self, X):
        weights = self.weights_V3
        biases = self.biases_V3
        adaps = self.adaps_V3
        num_layers = len(weights) + 1
        h = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            a = adaps[l]
            h = tf.tanh(tf.multiply(a, tf.add(tf.matmul(h, W), b)))
        W = weights[-1]
        b = biases[-1]
        V3 = tf.add(tf.matmul(h, W), b)
        return V3

    def net_V4(self, X):
        weights = self.weights_V4
        biases = self.biases_V4
        adaps = self.adaps_V4
        num_layers = len(weights) + 1
        h = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            a = adaps[l]
            h = tf.tanh(tf.multiply(a, tf.add(tf.matmul(h, W), b)))
        W = weights[-1]
        b = biases[-1]
        V4 = tf.add(tf.matmul(h, W), b)
        return V4

    def net_V5(self, X):
        weights = self.weights_V5
        biases = self.biases_V5
        adaps = self.adaps_V5
        num_layers = len(weights) + 1
        h = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            a = adaps[l]
            h = tf.tanh(tf.multiply(a, tf.add(tf.matmul(h, W), b)))
        W = weights[-1]
        b = biases[-1]
        V5 = tf.add(tf.matmul(h, W), b)
        return V5

    def net_W1(self, X):
        weights = self.weights_W1
        biases = self.biases_W1
        adaps = self.adaps_W1
        num_layers = len(weights) + 1
        h = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            a = adaps[l]
            h = tf.tanh(tf.multiply(a, tf.add(tf.matmul(h, W), b)))
        W = weights[-1]
        b = biases[-1]
        W1 = tf.add(tf.matmul(h, W), b)
        return W1

    def net_W2(self, X):
        weights = self.weights_W2
        biases = self.biases_W2
        adaps = self.adaps_W2
        num_layers = len(weights) + 1
        h = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            a = adaps[l]
            h = tf.tanh(tf.multiply(a, tf.add(tf.matmul(h, W), b)))
        W = weights[-1]
        b = biases[-1]
        W2 = tf.add(tf.matmul(h, W), b)
        return W2

    def net_W3(self, X):
        weights = self.weights_W3
        biases = self.biases_W3
        adaps = self.adaps_W3
        num_layers = len(weights) + 1
        h = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            a = adaps[l]
            h = tf.tanh(tf.multiply(a, tf.add(tf.matmul(h, W), b)))
        W = weights[-1]
        b = biases[-1]
        W3 = tf.add(tf.matmul(h, W), b)
        return W3

    def net_W4(self, X):
        weights = self.weights_W4
        biases = self.biases_W4
        adaps = self.adaps_W4
        num_layers = len(weights) + 1
        h = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            a = adaps[l]
            h = tf.tanh(tf.multiply(a, tf.add(tf.matmul(h, W), b)))
        W = weights[-1]
        b = biases[-1]
        W4 = tf.add(tf.matmul(h, W), b)
        return W4

    def net_W5(self, X):
        weights = self.weights_W5
        biases = self.biases_W5
        adaps = self.adaps_W5
        num_layers = len(weights) + 1
        h = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            a = adaps[l]
            h = tf.tanh(tf.multiply(a, tf.add(tf.matmul(h, W), b)))
        W = weights[-1]
        b = biases[-1]
        W5 = tf.add(tf.matmul(h, W), b)
        return W5

    def pinn(self, t, x):
        X = tf.concat([t, x], 1)
        U = self.net_U(X)
        V1 = self.net_V1(X)
        V2 = self.net_V2(X)
        V3 = self.net_V3(X)
        V4 = self.net_V4(X)
        V5 = self.net_V5(X)
        W1 = self.net_W1(X)
        W2 = self.net_W2(X)
        W3 = self.net_W3(X)
        W4 = self.net_W4(X)
        W5 = self.net_W5(X)

        gf = 2 * x - t ** 2 + 2 * x * t + 1 + t ** 2 * x + t + (
                    (t ** 6 * x ** 3 + 2 * t ** 5 * x ** 2 + t ** 4 * x - 12 * t ** 2 * x - 6 * t) * tf.cos(
                t ** 2 * x) + \
                    (-5 * t ** 4 * x ** 2 - 6 * t ** 3 * x - t ** 2 + 12) * tf.sin(
                t ** 2 * x) + t ** 5 * x ** 2 + 6 * t) / x ** 2

        U_x = tf.gradients(U, x)
        U_t = tf.gradients(U, t)
        U_tt = tf.gradients(U_t, t)
        V1_x = tf.gradients(V1, x)
        W1_t = tf.gradients(W1, t)
        V2_x = tf.gradients(V2, x)
        W2_t = tf.gradients(W2, t)
        V3_x = tf.gradients(V3, x)
        W3_t = tf.gradients(W3, t)
        V4_x = tf.gradients(V4, x)
        W4_t = tf.gradients(W4, t)
        V5_x = tf.gradients(V5, x)
        W5_t = tf.gradients(W5, t)

        f1 = gf + t ** 5 * (V1 - V2 + V3 - V4 + V5) - U_tt + U_x - U_t - U
        e1 = V1_x - W1
        e2 = V2_x - t ** 2 * x ** 2 * W2 * 0.5
        e3 = V3_x - (t ** 4 * x ** 4 * W3) / 24
        e4 = V4_x - (t ** 6 * x ** 6 * W4) / 720
        e5 = V5_x - (t ** 8 * x ** 8 * W5) / 40320
        e6 = W1_t - U**2
        e7 = W2_t - t ** 2 * U**2
        e8 = W3_t - t ** 4 * U**2
        e9 = W4_t - t ** 6 * U**2
        e10 = W5_t - t ** 8 * U**2
        return gf, f1, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, U, V1, V2, V3, V4, V5, W1, W2, W3, W4, W5, U_x, U_t, U_tt, V1_x, V2_x, V3_x, \
               V4_x, V5_x, W1_t, W2_t, W3_t, W4_t, W5_t

    def generateTrain(self, num_train_tps):
        tt = tf.linspace(np.float32(self.t_range[0]), np.float32(self.t_range[1]), 500)
        self.tt = tf.reshape(tt, [-1, 1])
        xf = tf.linspace(np.float32(self.x_range[0]), np.float32(self.x_range[1]), 500)
        self.xf = tf.reshape(xf, [-1, 1])

        xi = tf.linspace(np.float32(self.x_range[0]), np.float32(self.x_range[1]), 100)
        ti = self.t_range[0] * tf.ones(xi.shape)
        self.ti = tf.reshape(ti, [-1, 1])
        self.xi = tf.reshape(xi, [-1, 1])

        tb = tf.linspace(np.float32(self.t_range[0]), np.float32(self.t_range[1]), 100)
        xb = self.x_range[0] * tf.ones(tb.shape)
        self.tb = tf.reshape(tb, [-1, 1])
        self.xb = tf.reshape(xb, [-1, 1])

        tb2 = tf.linspace(np.float32(self.t_range[0]), np.float32(self.t_range[1]), 100)
        xb2 = self.x_range[1] * tf.ones(tb2.shape)
        self.tb2 = tf.reshape(tb2, [-1, 1])
        self.xb2 = tf.reshape(xb2, [-1, 1])

        return

    def train(self, Xx, it, num_train_tps):
        tx = np.linspace(self.t_range[0], self.t_range[1], num_train_tps)
        tx = np.reshape(tx, [-1, 1])
        xx = np.linspace(self.x_range[0], self.x_range[1], num_train_tps)
        xx = np.reshape(xx, [-1, 1])

        tf_dict = {self.xx_tf: tx, self.tx_tf: xx}

        self.sess.run(self.train_op_Adam, tf_dict)

        loss_value, loss_value_f, loss_value_e1, loss_value_e2, loss_value_e3, loss_value_e4, loss_value_e5, loss_value_e6, loss_value_e7, \
        loss_value_e8, loss_value_e9, loss_value_e10, loss_value_i, loss_value_b1, loss_value_b2 = self.sess.run(
            [self.loss, self.loss_f, self.loss_e1, self.loss_e2, self.loss_e3, self.loss_e4, self.loss_e5, self.loss_e6,
             self.loss_e7,
             self.loss_e8, self.loss_e9, self.loss_e10, self.loss_i, self.loss_b1, self.loss_b2
             ], tf_dict)
        loss_value_array = [loss_value_f *10, loss_value_e1 * 1, loss_value_e2 * 1, loss_value_e3 * 100,
                            loss_value_e4 * 1000, loss_value_e5 * 100,loss_value_e6 * 0.1, loss_value_e7 * 1,\
                            loss_value_e8 *0.1, loss_value_e9 *1,loss_value_e10 *1,\
                            loss_value_i * 1, loss_value_b1 * 1, loss_value_b2 * 10]

        np.set_printoptions(precision=6)
        content = 'It: %d, Loss: %.3e' % (it, loss_value) + ' losses ILRUDrxIC' + str(loss_value_array)
        print(content, flush=True)
        self.f.write(content + "\n")
        return loss_value, loss_value_array

    def test(self, num_test_tps):
        t = np.linspace(self.t_range[0], self.t_range[1], num_test_tps)
        t_2D = np.matmul(t[:, None], np.ones((1, num_test_tps)))
        t_test = np.reshape(t_2D, [-1, 1])
        x = np.linspace(self.x_range[0], self.x_range[1], num_test_tps)
        x_2D = np.matmul(np.ones((num_test_tps, 1)), x[None, :])
        x_test = np.reshape(x_2D, [-1, 1])
        tf_dict = {self.t_test_tf: t_test, self.x_test_tf: x_test}

        U_test = self.sess.run(self.U_test, tf_dict)
        V1_test = self.sess.run(self.V1_test, tf_dict)
        V2_test = self.sess.run(self.V2_test, tf_dict)
        V3_test = self.sess.run(self.V3_test, tf_dict)
        V4_test = self.sess.run(self.V4_test, tf_dict)
        V5_test = self.sess.run(self.V5_test, tf_dict)
        W1_test = self.sess.run(self.W1_test, tf_dict)
        W2_test = self.sess.run(self.W2_test, tf_dict)
        W3_test = self.sess.run(self.W3_test, tf_dict)
        W4_test = self.sess.run(self.W4_test, tf_dict)
        W5_test = self.sess.run(self.W5_test, tf_dict)
        f1_test = self.sess.run(self.f1_test, tf_dict)
        e1_test = self.sess.run(self.e1_test, tf_dict)
        e2_test = self.sess.run(self.e2_test, tf_dict)
        e3_test = self.sess.run(self.e3_test, tf_dict)
        e4_test = self.sess.run(self.e4_test, tf_dict)
        e5_test = self.sess.run(self.e5_test, tf_dict)
        e6_test = self.sess.run(self.e6_test, tf_dict)
        e7_test = self.sess.run(self.e7_test, tf_dict)
        e8_test = self.sess.run(self.e8_test, tf_dict)
        e9_test = self.sess.run(self.e9_test, tf_dict)
        e10_test = self.sess.run(self.e10_test, tf_dict)
        X_test = np.hstack((t_test, x_test))

        return X_test, U_test, V1_test, V2_test, V3_test, V4_test, V5_test, W1_test, W2_test, W3_test, W4_test, W5_test, f1_test, \
               e1_test, e2_test, e3_test, e4_test, e5_test, e6_test, e7_test, e8_test, e9_test, e10_test


def ElasImag(nIter=10000, print_period=10000, plot_period=10000):
    t_range = np.array([a, b])
    x_range = np.array([c, d])

    layers_U = [2, 40, 40, 40, 40, 40,  1]
    layers_V1 = [2, 40, 40, 40, 40, 40,  1]
    layers_V2 = [2, 40, 40, 40,  40, 40, 1]
    layers_V3 = [2, 40, 40, 40,  40, 40, 1]
    layers_V4 =[2, 40, 40, 40,   40, 40,1]
    layers_V5 = [2, 40, 40, 40,  40, 40, 1]
    layers_W1 = [2, 40, 40, 40, 40, 40,  1]
    layers_W2 = [2, 40, 40, 40, 40, 40,  1]
    layers_W3 = [2, 40, 40, 40, 40, 40,  1]
    layers_W4 = [2, 40, 40, 40,  40, 40, 1]
    layers_W5 = [2, 40, 40, 40,  40, 40, 1]
    f = open("loss_record_2D.txt", "w")

    tt = np.linspace(t_range[0], t_range[1], 200)
    tt = np.reshape(tt, [-1, 1])
    xx = np.linspace(x_range[0], x_range[1], 200)
    xx = np.reshape(xx, [-1, 1])
    Xx = tf.concat([tt, xx], 1)
    figure_path = './Figure/'
    if not os.path.isdir(figure_path):
        os.makedirs(figure_path)

    num_train_tps = 500
    num_test_tps = 100

    # Create the model
    model = PhysicsInformedNN(layers_U, layers_V1, layers_V2, layers_V3, layers_V4, layers_V5, layers_W1, layers_W2,
                              layers_W3, layers_W4, layers_W5, x_range, t_range, f, num_train_tps)

    it_array = []
    loss_array = []
    losses_array = []

    start_time = time.time()

    for it in range(1, nIter + 1):
        loss, losses = model.train(Xx, it, it % print_period == 0)
        if (it % print_period == 0):
            loss_array.append(loss)
            losses_array.append(losses)
            it_array.append(it)
            dt = time.time() - start_time
            print('Time: ', dt)
            start_time = time.time()
            if (it % plot_period == 0 or it == print_period):
                print("Result Plotted...")
                X_test, U_test, V1_test, V2_test, V3_test, V4_test, V5_test, W1_test, W2_test, W3_test, W4_test, W5_test, f_test, \
                e1_test, e2_test, e3_test, e4_test, e5_test, e6_test, e7_test, e8_test, e9_test, e10_test = model.test(
                    num_test_tps)
                U_exact = X_test[:, 0:1] + X_test[:, 0:1] ** 2 * X_test[:, 1:]
                A=np.hstack((X_test[:, 0:1],X_test[:, 0:1],U_exact))
                sio.savemat('U2D.mat', {'A': A})
                l2error = np.linalg.norm(U_test - U_exact, 2) / np.linalg.norm(U_exact, 2)
                print('L2 error: ', l2error)
                plt.figure(figsize=(8, 3))
                figtopic = 'U_test'
                t2D = np.reshape(X_test[:, 0:1], [num_test_tps, num_test_tps])
                x2D = np.reshape(X_test[:, 1:], [num_test_tps, num_test_tps])
                U_test2D = np.reshape(U_test, [num_test_tps, num_test_tps])
                cs = plt.contourf(t2D, x2D, U_test2D, 300, cmap=plt.cm.jet)
                plt.colorbar(cs)
                plt.xlabel('t')
                plt.ylabel('x')
                plt.title(figtopic)
                plt.show()
                plt.figure(figsize=(8, 3))
                figtopic = 'U_exact'
                t2D = np.reshape(X_test[:, 0:1], [num_test_tps, num_test_tps])
                x2D = np.reshape(X_test[:, 1:], [num_test_tps, num_test_tps])
                U_exact = X_test[:, 0:1] + X_test[:, 0:1] ** 2 * X_test[:, 1:]
                U_exact2D = np.reshape(U_exact, [num_test_tps, num_test_tps])
                cs = plt.contourf(t2D, x2D, U_exact2D, 300, cmap=plt.cm.jet)
                plt.colorbar(cs)
                plt.xlabel('t')
                plt.ylabel('x')
                plt.title(figtopic)
                plt.show()
                plt.figure(figsize=(8, 3))
                figtopic = 'U_err2D '
                t2D = np.reshape(X_test[:, 0:1], [num_test_tps, num_test_tps])
                x2D = np.reshape(X_test[:, 1:], [num_test_tps, num_test_tps])
                U_exact = X_test[:, 0:1] + X_test[:, 0:1] ** 2 * X_test[:, 1:]
                U_exact2D = np.reshape(U_exact, [num_test_tps, num_test_tps])
                U_test2D = np.reshape(U_test, [num_test_tps, num_test_tps])
                U_err2D = U_test2D - U_exact2D
                cs = plt.contourf(t2D, x2D, U_err2D, 300, cmap=plt.cm.jet)
                plt.colorbar(cs)
                plt.xlabel('t')
                plt.ylabel('x')
                plt.title(figtopic)
                plt.show()
                U_test2D = np.hstack((U_exact2D, U_test2D, U_err2D))
                sio.savemat('U_exact2D_data_2D.mat', {'U_exact2D': U_exact2D})
                sio.savemat('U_test2D_data_2D.mat', {'U_test2D': U_test2D})
                sio.savemat('U_err2D_data_2D.mat', {'U_err2D2D': U_err2D})
                sio.savemat('X_data_2D.mat', {'X': X_test})


ElasImag(nIter=50000, print_period=10000, plot_period=10000)