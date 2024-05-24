import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
import pandas as pd
import numpy as np
np.set_printoptions(threshold=np.inf)
import matplotlib
import matplotlib.pyplot as plt
import time
import os
import scipy.io as sio
import math
from math import exp, pi,sin,cos

np.random.seed(1)
tf.set_random_seed(1)
###########设置定义域范围
# a = 0
# b = 1
a =-pi/2
b = pi/2
class PhysicsInformedNN:
    # Initialize the class

    def __init__(self, layers_U, layers_V1,layers_V2,layers_V3, x_range, f, num_train_tps):

        # Initialize NNs
        self.layers_U = layers_U
        self.weights_U, self.biases_U, self.adaps_U = self.initialize_NN(layers_U)
        self.layers_V1 = layers_V1
        self.weights_V1, self.biases_V1, self.adaps_V1 = self.initialize_NN(layers_V1)
        self.layers_V2 = layers_V2
        self.weights_V2, self.biases_V2, self.adaps_V2 = self.initialize_NN(layers_V2)
        self.layers_V3 = layers_V3
        self.weights_V3, self.biases_V3, self.adaps_V3 = self.initialize_NN(layers_V3)

        # Parameters
        self.x_range =x_range
        self.lb = np.array([ x_range[0]])
        self.ub = np.array([ x_range[1]])
        # Output file
        self.f = f


        # Coordinates of datapoints             #####################数据点坐标
        self.xx_tf = tf.placeholder(tf.float32, shape=[None, 1])
        # Test Points                           ######################测试点
        self.x_test_tf = tf.placeholder(tf.float32, shape=[None, 1])

        # Generate Training and Testing Points  ########################生成训练和测试点
        self.generateTrain(num_train_tps)

        # Physics
        self.f1, self.e1,self.e2,self.e3, self.Uf, self.V1f,self.V2f,self.V3f,self.U_xf,self.V1_xf,self.V2_xf,self.V3_xf,self.V1e,self.V2e,self.V3e= self.pinn(self.xf)
        _,_,_,_,self.Ui, self.V1i,self.V2i,self.V3i,_,_,_,_,_,_,_= self.pinn(self.xi)
        _,_,_,_,_,_,_,_,_,_,_,_,self.V1e,self.V2e,self.V3e= self.pinn(self.xe)
        self.f1_test,  self.e1_test, self.e2_test,self.e3_test,self.U_test, self.V1_test,self.V2_test,self.V3_test,self.U_x_test,self.V1_x_test,self.V2_x_test,self.V3_x_test,_,_,_= self.pinn(self.x_test_tf)
        # 均方误差
        # self.loss_f = tf.reduce_mean((self.U_xf-(self.xf+2)*self.V1f-self.xf*(self.xf+2)*self.V2f-0.5*self.xf**2*(self.xf+2)*self.V3f-2*tf.exp(2*self.xf)-1.0+tf.exp(self.xf**2+2*self.xf))** 2)
        # self.loss_e1 = tf.reduce_mean((self.V1_xf -self.Uf)** 2)
        # self.loss_e2 = tf.reduce_mean((self.V2_xf - self.xf*self.Uf) ** 2)
        # self.loss_e3 = tf.reduce_mean((self.V3_xf -self.xf**2*self.Uf) ** 2)
        # self.loss_i= tf.reduce_mean((self.Ui-1) ** 2)+tf.reduce_mean((self.V1i-0) ** 2)\
        #              +tf.reduce_mean((self.V2i-0) ** 2)+tf.reduce_mean((self.V3i-0) ** 2)
        # self.loss = self.loss_f *1 + self.loss_e1 * 1 + self.loss_e2 *1 + self.loss_e3 * 0.1 + self.loss_i * 1
        # # 平均绝对误差
        # self.loss_f = tf.reduce_mean(abs(self.U_xf-(self.xf+2)*self.V1f-self.xf*(self.xf+2)*self.V2f-0.5*self.xf**2*(self.xf+2)*self.V3f-2*tf.exp(2*self.xf)-1.0+tf.exp(self.xf**2+2*self.xf)))
        # self.loss_e1 = tf.reduce_mean(abs(self.V1_xf -self.Uf))
        # self.loss_e2 = tf.reduce_mean(abs(self.V2_xf - self.xf*self.Uf))
        # self.loss_e3 = tf.reduce_mean(abs(self.V3_xf -self.xf**2*self.Uf))
        # self.loss_i= tf.reduce_mean(abs(self.Ui-1))+tf.reduce_mean(abs(self.V1i-0) )\
        #              +tf.reduce_mean(abs(self.V2i-0))+tf.reduce_mean(abs(self.V3i-0))
        # 均方根误差
        # self.loss_f = tf.sqrt(tf.reduce_mean((self.U_xf - (self.xf + 2) * self.V1f - self.xf * (
        #             self.xf + 2) * self.V2f - 0.5 * self.xf ** 2 * (self.xf + 2) * self.V3f - 2 * tf.exp(
        #     2 * self.xf) - 1.0 + tf.exp(self.xf ** 2 + 2 * self.xf)) ** 2))
        # self.loss_e1 = tf.sqrt(tf.reduce_mean((self.V1_xf - self.Uf) ** 2))
        # self.loss_e2 = tf.sqrt(tf.reduce_mean((self.V2_xf - self.xf * self.Uf) ** 2))
        # self.loss_e3 = tf.sqrt(tf.reduce_mean((self.V3_xf - self.xf ** 2 * self.Uf) ** 2))
        # self.loss_i = tf.sqrt(tf.reduce_mean((self.Ui - 1) ** 2)) + tf.sqrt(tf.reduce_mean((self.V1i - 0) ** 2) )\
        #               + tf.sqrt(tf.reduce_mean((self.V2i - 0) ** 2)) + tf.sqrt(tf.reduce_mean((self.V3i - 0) ** 2))

        # Fredholm积分微分方程
        # 均方误差
        self.loss_f = tf.reduce_mean((self.U_xf -  self.V1e + 0.5 * self.xf ** 2  * self.V2e - self.xf ** 4  * self.V3e * 1 / 24 - tf.cos(
            self.xf)) ** 2)
        self.loss_e1 = tf.reduce_mean((self.V1_xf - tf.sin(self.Uf)) ** 2)
        self.loss_e2 = tf.reduce_mean((self.V2_xf - self.xf ** 2 * tf.sin(self.Uf)) ** 2)
        self.loss_e3 = tf.reduce_mean((self.V3_xf - self.xf ** 4 * tf.sin(self.Uf)) ** 2)
        self.loss_i = tf.reduce_mean((self.Ui + 1) ** 2) + tf.reduce_mean((self.V1i) ** 2) \
                      + tf.reduce_mean((self.V2i) ** 2) + tf.reduce_mean((self.V3i) ** 2)

        # 平均绝对误差
        # self.loss_f = tf.reduce_mean(abs(self.U_xf-(self.xf**2-1.0)*self.V1e+0.5*self.xf**2*(self.xf**2-1.0)*self.V2e-self.xf**4*(self.xf**2-1.0)*self.V3e*1/24-tf.cos(self.xf)))
        # self.loss_e1 = tf.reduce_mean(abs(self.V1_xf -self.Uf))
        # self.loss_e2 = tf.reduce_mean(abs(self.V2_xf - self.xf**2*self.Uf))
        # self.loss_e3 = tf.reduce_mean(abs(self.V3_xf -self.xf**4*self.Uf))
        # self.loss_i= tf.reduce_mean(abs(self.Ui+1))+tf.reduce_mean(abs(self.V1i))\
        #             +tf.reduce_mean(abs(self.V2i))+tf.reduce_mean(abs(self.V3i))
        # 均方根误差
        # self.loss_f =tf.sqrt(tf.reduce_mean((self.U_xf - (self.xf ** 2 - 1.0) * self.V1e + 0.5 * self.xf ** 2 * (
        #             self.xf ** 2 - 1.0) * self.V2e - self.xf ** 4 * (self.xf ** 2 - 1.0) * self.V3e * 1 / 24 - tf.cos(
        #     self.xf)) ** 2))
        # self.loss_e1 =tf.sqrt(tf.reduce_mean((self.V1_xf - self.Uf) ** 2))
        # self.loss_e2 =tf.sqrt(tf.reduce_mean((self.V2_xf - self.xf ** 2 * self.Uf) ** 2))
        # self.loss_e3 =tf.sqrt(tf.reduce_mean((self.V3_xf - self.xf ** 4 * self.Uf) ** 2))
        # self.loss_i = tf.sqrt(tf.reduce_mean((self.Ui + 1) ** 2)) +  tf.sqrt(tf.reduce_mean((self.V1i) ** 2)) \
        #               + tf.sqrt(tf.reduce_mean((self.V2i) ** 2))+  tf.sqrt(tf.reduce_mean((self.V3i) ** 2))
        # Total Loss
        self.loss = self.loss_f *0.1+ self.loss_e1*1+ self.loss_e2*1 +self.loss_e3*0.1+self.loss_i*1


        # Optimizer
        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

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
            h = tf.nn.tanh(tf.multiply(a, tf.add(tf.matmul(h, W), b)))
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
            h = tf.nn.tanh(tf.multiply(a, tf.add(tf.matmul(h, W), b)))
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
            h = tf.nn.tanh(tf.multiply(a, tf.add(tf.matmul(h, W), b)))
        W = weights[-1]
        b = biases[-1]
        V2= tf.add(tf.matmul(h, W), b)
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
            h = tf.nn.tanh(tf.multiply(a, tf.add(tf.matmul(h, W), b)))
        W = weights[-1]
        b = biases[-1]
        V3 = tf.add(tf.matmul(h, W), b)
        return V3
    #
    def pinn(self,x):
        X=x
        # x=self.x_range[1] * tf.ones(x.shape)
        U = self.net_U(X)
        V1 =self.net_V1(X)
        V2 =self.net_V2(X)
        V3= self.net_V3(X)
        V1e = self.net_V1(X)
        V2e= self.net_V2(X)
        V3e = self.net_V3(X)

        U_x = tf.gradients(U, x)  # du/dx
        V1_x = tf.gradients(V1, x)  # dv/dx
        V2_x = tf.gradients(V2, x)  # dv/dx
        V3_x = tf.gradients(V3, x)  # dv/dx

        f1 = U_x -  V1e + 0.5 * x ** 2 * V2e - x ** 4 * V3e * 1 / 24 - tf.cos(x)
        e1 = V1_x -  tf.sin(U)
        e2 = V2_x - x ** 2 *  tf.sin(U)
        e3 = V3_x - x ** 4 *  tf.sin(U)

        return f1,e1,e2,e3,U,V1,V2,V3, U_x ,V1_x,V2_x,V3_x,V1e,V2e,V3e


    def generateTrain(self, num_train_tps):
        xf = tf.linspace(np.float32(self.x_range[0]), np.float32(self.x_range[1]),100)
        self.xf = tf.reshape(xf, [-1, 1])
        xi = self.x_range[0] * tf.ones(xf.shape)
        self.xi = tf.reshape(xi, [-1, 1])
        xe = self.x_range[1] * tf.ones(xf.shape)
        self.xe = tf.reshape(xe, [-1, 1])
        return



    def train(self, xx,  it, num_train_tps):
        xx = np.linspace(self.x_range[0], self.x_range[1],num_train_tps)
        xx = np.reshape(xx, [-1, 1])
        tf_dict = {self.xx_tf: xx}
        self.sess.run(self.train_op_Adam, tf_dict)

        loss_value, loss_value_f,loss_value_e1,loss_value_e2,loss_value_e3,loss_value_i= self.sess.run(
            [self.loss, self.loss_f,self.loss_e1,self.loss_e2,self.loss_e3,self.loss_i
             ], tf_dict)
        loss_value_array = [loss_value_f*0.1,loss_value_e1*1,loss_value_e2*1,loss_value_e3*0.1,loss_value_i*1]
        np.set_printoptions(precision=6)
        content = 'It: %d, Loss: %.3e' % (it, loss_value) + '  Losses ILRUDrxIC:' + str(loss_value_array)
        print(content, flush=True)
        self.f.write(content + "\n")
        return loss_value, loss_value_array

    def test(self, num_test_tps):
        x = np.linspace(self.x_range[0], self.x_range[1], num_test_tps)
        x_test = np.reshape(x, [-1, 1])
        tf_dict = {self.x_test_tf: x_test}
        U_test = self.sess.run(self.U_test, tf_dict)
        V1_test = self.sess.run(self.V1_test, tf_dict)
        V2_test = self.sess.run(self.V2_test, tf_dict)
        V3_test = self.sess.run(self.V3_test, tf_dict)
        U_test_x = tf.gradients(U_test, x_test)  # du/dx
        V1_test_x = tf.gradients(V1_test, x_test)  # du/dx
        V2_test_x = tf.gradients(V2_test, x_test)  # du/dx
        V3_test_x = tf.gradients(V3_test, x_test)  # du/dx
        f1_test = self.sess.run(self.f1_test, tf_dict)
        e1_test = self.sess.run(self.e1_test, tf_dict)
        e2_test = self.sess.run(self.e2_test, tf_dict)
        e3_test = self.sess.run(self.e3_test, tf_dict)
        return x_test, U_test, V1_test, V2_test,V3_test,f1_test,e1_test,e2_test,e3_test,U_test_x,V1_test_x,V2_test_x,V3_test_x


def ElasImag(nIter = 20000, print_period = 1000, plot_period = 1000):
    x_range = np.array([a, b])
    # Network Structure
    layers_U = [1, 40, 40, 1]
    layers_V1 = [1, 40, 40, 1]
    layers_V2 = [1, 40, 40, 1]
    layers_V3 = [1, 40, 40, 1]


    Xx = np.linspace(x_range[0], x_range[1], 100)
    Xx = np.reshape(Xx, [-1, 1])

    f = open("loss_record_F_3.txt", "w")


    num_train_tps =100
    num_test_tps =20

    model = PhysicsInformedNN(layers_U, layers_V1,layers_V2,layers_V3, x_range, f, num_train_tps)

    it_array = []
    loss_array = []
    losses_array = []
    start_time = time.time()
    for it in range(1, nIter+1):
        loss, losses = model.train(Xx, it, it%print_period==0)
        if (it%print_period==0):
            loss_array.append(loss)
            losses_array.append(losses)
            it_array.append(it)
            dt = time.time() - start_time
            print('Time: ', dt)
            start_time = time.time()
            if (it % plot_period == 0 or it == print_period):
                print("Result Plotted...")
                plt.figure(1)
                x_test, U_test, V1_test,V2_test,V3_test, _,_, _, _, _, _, _,_ = model.test(num_test_tps)
                u_exact = np.sin(x_test)
                U = np.hstack((x_test,u_exact, U_test))
                print(U)
                sio.savemat('U_data_F_3.mat', {'U': U})
                l2Uerror = np.linalg.norm(U_test - u_exact, 2) / np.linalg.norm(u_exact, 2)
                print('L2U error: ', l2Uerror)
                plt.figure()
                plt.plot(x_test, U_test, label='u(x) of PINN')
                plt.scatter(x_test, u_exact, s=30, c='r', marker='o', label='Exact u(x)')
                plt.legend(loc='upper left')
                plt.xlabel('x')
                plt.ylabel('u(x)')
                plt.xlim(x_range[0], x_range[1])
                plt.tight_layout()
                plt.show()
                figtopic = 'loss'
                plt.semilogy(it_array, loss_array, '-b')
                plt.xlabel("Num Iteration")
                plt.ylabel("Loss")
                plt.title(figtopic)
                plt.show()

ElasImag(nIter =40000, print_period =10000, plot_period =10000)