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

a = 0
b = 1

class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, layers_U, layers_V1,layers_V2,layers_V3,layers_LM, x_range, f, num_train_tps):

        # Initialize NNs
        self.layers_U = layers_U
        self.weights_U, self.biases_U, self.adaps_U = self.initialize_NN(layers_U)
        self.layers_V1 = layers_V1
        self.weights_V1, self.biases_V1, self.adaps_V1 = self.initialize_NN(layers_V1)
        self.layers_V2 = layers_V2
        self.weights_V2, self.biases_V2, self.adaps_V2 = self.initialize_NN(layers_V2)
        self.layers_V3 = layers_V3
        self.weights_V3, self.biases_V3, self.adaps_V3 = self.initialize_NN(layers_V3)
        self.layers_LM = layers_LM
        self.weights_LM, self.biases_LM, self.adaps_LM = self.initialize_NN(layers_LM)

        # Parameters
        self.x_range =x_range
        self.lb = np.array([ x_range[0]])
        self.ub = np.array([ x_range[1]])
        # Output file
        self.f = f
        self.xx_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.ux_star_tf = tf.placeholder(tf.float32, shape=[None, 1])
        # Test Points                           ######################测试点
        self.x_test_tf = tf.placeholder(tf.float32, shape=[None, 1])

        # Generate Training and Testing Points  ########################生成训练和测试点
        self.generateTrain(num_train_tps)

        # Physics
        self.f1, self.e1, self.e2, self.e3, self.Ux, self.V1x, self.V2x, self.V3x, self.U_xx, self.V1_xx, self.V2_xx, self.V3_xx, self.V1ex, self.V2ex, self.V3ex, self.LMx, self.LM_xx = self.pinn(
            self.xx_tf)
        self.f1, self.e1,self.e2,self.e3, self.Uf, self.V1f,self.V2f,self.V3f,self.U_xf,self.V1_xf,self.V2_xf,self.V3_xf,self.V1e,self.V2e,self.V3e,self.LMf, self.LM_xf= self.pinn(self.xf)
        _,_,_,_,self.Ui, self.V1i,self.V2i,self.V3i,_,_,_,_,_,_,_,_,_= self.pinn(self.xi)
        _,_,_,_,_,_,_,_,_,_,_,_,self.V1e,self.V2e,self.V3e,_,_= self.pinn(self.xe)
        self.f1_test,  self.e1_test, self.e2_test,self.e3_test,self.U_test, self.V1_test,self.V2_test,self.V3_test,self.U_x_test,self.V1_x_test,self.V2_x_test,self.V3_x_test,_,_,_, self.LM_test,self.LM_xtest= self.pinn(self.x_test_tf)

        self.loss_U = tf.reduce_mean((self.Ux - self.ux_star_tf) ** 2)
        self.loss_LM = tf.reduce_mean((self.LM_xf - 0*self.Uf) ** 2)
        self.loss_f = tf.reduce_mean((self.U_xf+self.LMf*(-(self.xf**2)*self.V1f-self.xf*(self.xf**2)*self.V2f-0.5*self.xf**2*(self.xf**2)*self.V3f)-2*tf.exp(2*self.xf)+self.xf*tf.exp(self.xf**2)*tf.log(tf.exp(2*self.xf))-2*tf.exp(self.xf**2)+2)** 2)
        self.loss_e1 = tf.reduce_mean((self.V1_xf -tf.log(self.Uf))** 2)
        self.loss_e2 = tf.reduce_mean((self.V2_xf - self.xf*tf.log(self.Uf)) ** 2)
        self.loss_e3 = tf.reduce_mean((self.V3_xf -self.xf**2*tf.log(self.Uf)) ** 2)
        self.loss_i= tf.reduce_mean((self.Ui-1) ** 2)+tf.reduce_mean((self.V1i-0) ** 2)\
                     +tf.reduce_mean((self.V2i-0) ** 2)+tf.reduce_mean((self.V3i-0) ** 2)

        # # Total Loss
        self.loss =self.loss_U*1+self.loss_LM*1000+ self.loss_f *0.1+ self.loss_e1*1+ self.loss_e2*1 +self.loss_e3*1+self.loss_i*1


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
            h = tf.tanh(tf.multiply(a, tf.add(tf.matmul(h, W), b)))
        W = weights[-1]
        b = biases[-1]
        U = tf.add(tf.matmul(h, W), b)
        return U**2

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
            h = tf.tanh(tf.multiply(a, tf.add(tf.matmul(h, W), b)))
        W = weights[-1]
        b = biases[-1]
        V3 = tf.add(tf.matmul(h, W), b)
        return V3


    def net_LM(self, X):
        weights = self.weights_LM
        biases = self.biases_LM
        adaps = self.adaps_LM
        num_layers = len(weights) + 1
        h = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            a = adaps[l]
            h = tf.tanh(tf.multiply(a, tf.add(tf.matmul(h, W), b)))
        W = weights[-1]
        b = biases[-1]
        LM = tf.add(tf.matmul(h, W), b)
        return LM
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
        LM = self.net_LM(X)


        U_x = tf.gradients(U, x)  # du/dx
        V1_x = tf.gradients(V1, x)  # dv1/dx
        V2_x = tf.gradients(V2, x)  # dv2dx
        V3_x = tf.gradients(V3, x)  # dv3/dx
        LM_x = tf.gradients(LM, x)  # dLM/dx

        f1 = U_x+LM*(-(x+2)*V1-x*(x+2)*V2-0.5*x**2*(x+2)*V3)-2*tf.exp(2*x)+tf.exp(x**2)*tf.log(tf.exp(2*x))*x-2*tf.exp(x**2)+2.0
        e1 = V1_x - tf.log(U)
        e2 = V2_x - x * tf.log(U)
        e3 = V3_x - x ** 2 * tf.log(U)



        return f1,e1,e2,e3,U,V1,V2,V3, U_x ,V1_x,V2_x,V3_x,V1e,V2e,V3e, LM, LM_x


    def generateTrain(self, num_train_tps):
        xf = tf.linspace(np.float32(self.x_range[0]), np.float32(self.x_range[1]), 200)
        self.xf = tf.reshape(xf, [-1, 1])
        xi = self.x_range[0] * tf.ones(xf.shape)
        self.xi = tf.reshape(xi, [-1, 1])
        xe = self.x_range[1] * tf.ones(xf.shape)
        self.xe = tf.reshape(xe, [-1, 1])
        return

    def train(self, Xx,Ux, it, printloss):
        xx=Xx
        xx = np.reshape(xx, [-1, 1])
        ux=Ux
        ux = ux.reshape(-1, 1)

        tf_dict = {self.xx_tf: xx,self.ux_star_tf: ux}


        self.sess.run(self.train_op_Adam, tf_dict)

        loss_value, loss_value_U,loss_value_LM,loss_value_f,loss_value_e1,loss_value_e2,loss_value_e3,loss_value_i= self.sess.run(
            [self.loss, self.loss_U,self.loss_LM,self.loss_f,self.loss_e1,self.loss_e2,self.loss_e3,self.loss_i
             ], tf_dict)
        loss_value_array = [loss_value_U*1,loss_value_LM*1000,loss_value_f*0.1,loss_value_e1*1,loss_value_e2*1,loss_value_e3*1,loss_value_i*1]
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
        LM_test = self.sess.run(self.LM_test, tf_dict)
        U_test_x = tf.gradients(U_test, x_test)
        V1_test_x = tf.gradients(V1_test, x_test)
        V2_test_x = tf.gradients(V2_test, x_test)
        V3_test_x = tf.gradients(V3_test, x_test)
        f1_test = self.sess.run(self.f1_test, tf_dict)
        e1_test = self.sess.run(self.e1_test, tf_dict)
        e2_test = self.sess.run(self.e2_test, tf_dict)
        e3_test = self.sess.run(self.e3_test, tf_dict)
        return x_test, U_test, V1_test, V2_test,V3_test,f1_test,e1_test,e2_test,e3_test,U_test_x,V1_test_x,V2_test_x,V3_test_x,LM_test

def import_data(filepath):

    data =np.loadtxt("U3.txt", delimiter=None, converters=None, skiprows=0, usecols=None, unpack=False, ndmin=0)
    Xx = data[:, 0:1]
    Ux = data[:,1:2]
    noise =0
    normal_data = np.random.normal(0, 1,Ux.shape)
    Ux =Ux * (1 + noise * normal_data)
    return Xx, Ux

def ElasImag(nIter = 20000, print_period = 1000, plot_period = 1000):
    x_range = np.array([a, b])
    # Network Structure
    layers_U = [1, 20, 20, 20, 1]
    layers_V1 =[1, 20, 20, 20, 1]
    layers_V2 = [1, 20, 20, 20, 1]
    layers_V3 = [1, 20, 20, 20, 1]
    layers_LM =  [1, 20,  1]

    Xx, Ux = import_data('U3.txt')

    f = open("loss_record_V_Lamada_2.txt", "w")

    num_train_tps = 200
    num_test_tps =20

    model = PhysicsInformedNN(layers_U, layers_V1,layers_V2,layers_V3,layers_LM, x_range, f, num_train_tps)

    it_array = []
    loss_array = []
    losses_array = []

    start_time = time.time()
    for it in range(1, nIter+1):
        loss, losses = model.train(Xx, Ux, it, it%print_period==0)
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
                x_test, U_test, V1_test, V2_test, V3_test, _, _, _, _, _,_,_,_,LM_test = model.test(num_test_tps)
                u_exact = np.exp(2*x_test)
                LM_exact = 1 * np.ones(LM_test.shape)
                l2Uerror = np.linalg.norm(U_test - u_exact, 2) / np.linalg.norm(u_exact, 2)
                l2LMerror = np.linalg.norm(LM_test - LM_exact, 2) / np.linalg.norm(LM_exact, 2)
                print('L2U error: ', l2Uerror)
                print('L2LM error: ', l2LMerror)
                xx = Xx
                ux = Ux
                U = np.hstack((x_test, LM_test, LM_exact, u_exact, U_test))
                print(U)
                sio.savemat('U_data_V_Lamada_1.mat', {'U': U})
                plt.plot(x_test, LM_test, label='LM_test of PINN')
                plt.figure()
                plt.plot(x_test, U_test, label='u(x) of PINN')
                plt.scatter(x_test, u_exact, s=30, c='r', marker='x', label='Exact u(x)')
                plt.plot(xx, ux, c='b', linestyle='' ' ', marker='o', label='ux of data')
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
ElasImag(nIter =20000, print_period =10000, plot_period =10000)