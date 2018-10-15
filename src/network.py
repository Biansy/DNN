import json
import random
import sys
import numpy as np
import mnist_loader


#定义二次方程函数
class QuadraticCost(object):

    @staticmethod
    def fn(a, y):
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        return (a-y) * sigmoid_prime(z)


#定义交叉熵代价函数
class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        return (a-y)


#### Main Network class
class Network(object):

    def __init__(self, sizes, cost=CrossEntropyCost):

        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost=cost

    def default_weight_initializer(self):
        """左右挤压后的初始化权重"""
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def large_weight_initializer(self):

        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        """前向传播"""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda = 0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False):
        """梯度下降算法"""

        if evaluation_data: n_data = len(evaluation_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in range(epochs):
            random.shuffle(training_data)   #打乱训练集列表内数据顺序
            mini_batches = [                #mini_batches从头开始取，每次取mini_batch_size个，凑成一个列表
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches: #对mini_batches里的数据进行挨个学习训练
                self.update_mini_batch( mini_batch, eta, lmbda, len(training_data))

            print("第{}轮训练完成".format(j+1))
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda, convert=False)
                training_cost.append(cost)
                print("训练数据集代价值: {}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print("训练数据集准确度: {} / {}".format(
                    accuracy, n))
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print("验证数据集代价值: {}".format(cost))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data, convert=False)
                evaluation_accuracy.append(accuracy)
                print("验证数据集准确度: {} / {}".format(
                    accuracy, n_data))
            print("")
        return evaluation_cost, evaluation_accuracy, \
            training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        ''''''
        nabla_b = [np.zeros(b.shape) for b in self.biases]      #初始DNN中每一层的 △b,△w都为0
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)              #进行反向传播,获得 △b,△w
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]   #把mini_batch中的每个 △b,△w分别加在一起
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [(1-eta*(lmbda/n))*w-(eta/len(mini_batch))*nw    #更新权重
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb                       #更新阈值
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """"""
        nabla_b = [np.zeros(b.shape) for b in self.biases]      #初始DNN中每一层的 △b,△w都为0
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # 前向传播
        a = x
        aList = [x] # 把每一层的“a”放在一个列表里
        zList = [ ] # 把每一层的“z”放在一个列表里
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, a)+b
            a = sigmoid(z)
            zList.append(z)
            aList.append(a)

        # 反向传播
        delta = (self.cost).delta(zList[-1], aList[-1], y)  #获得代价值“C”
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, aList[-2].transpose())
        for l in range(2, self.num_layers):
            z = zList[-l]
            sp = sigmoid_prime(z)   #逻辑函数，a对z的导
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, aList[-l-1].transpose())
        return (nabla_b, nabla_w)

    def accuracy(self, data, convert=False):
        """
        """
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                        for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in data]
        return sum(x == y for (x, y) in results)

    def total_cost(self, data, lmbda, convert=False):
        """
        """
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert: y = vectorized_result(y)
            cost += self.cost.fn(a, y)/len(data)
        cost += 0.5*(lmbda/len(data))*sum(
            np.linalg.norm(w)**2 for w in self.weights)
        return cost

    def save(self, filename):
        """把神经网络的权重存储至文件"""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()

def load(filename):
    """加载已有的神经网络"""
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net

def vectorized_result(j):
    """ 把数字向量化"""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def sigmoid(z):
    """逻辑函数"""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """逻辑函数的导"""
    return sigmoid(z)*(1-sigmoid(z))



training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
# network = Network([784,30,10])
# network.SGD(training_data, 20, 50, 0.01,
#             lmbda = 0.0,
#             evaluation_data=test_data,
#             monitor_evaluation_cost=True,
#             monitor_evaluation_accuracy=True,
#             monitor_training_cost=True,
#             monitor_training_accuracy=True)
# network.save("data\module")
module = load("data\module")
module.SGD(training_data, 30, 50, 0.01,
            lmbda = 0.0,
            evaluation_data=test_data,
            monitor_evaluation_cost=True,
            monitor_evaluation_accuracy=True,
            monitor_training_cost=True,
            monitor_training_accuracy=True)