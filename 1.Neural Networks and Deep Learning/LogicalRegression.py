# -*-coding:utf-8-*-
import h5py
import pickle
import numpy as np
from scipy import ndimage
from scipy.misc import imresize
from matplotlib import pyplot as plt


class LogicRegression(object):
    def __init__(self, dim, learning_rate=0.005, num_iterations=2000):
        """
        :param dim: size of the w vector we want (or number of parameters in this case).
        :param learning_rate: learning rate of the gradient descent update rule.
        :param num_iterations: number of iterations of the optimization loop.
        """
        self.dim = dim
        self.w = np.zeros(shape=(dim, 1))
        self.b = 0
        assert (self.w.shape == (dim, 1))
        assert (isinstance(self.b, float) or isinstance(self.b, int))
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.model_name = 'LogicalRegression.pickle'

    @staticmethod
    def sigmoid(z):
        """
        :param z: A scalar or numpy array of any size.
        :return: sigmoid(z)
        """
        # np.exp(-z)由于数值太小会溢出,调整学习率和数据归一化方法，对其有影响。
        return 1.0 / (1.0 + np.exp(-z))

    def feed_forward(self, x):
        return self.sigmoid(np.dot(self.w.T, x) + self.b)

    def propagate(self, x, y):
        m = x.shape[1]
        a = self.feed_forward(x)

        # 不指定axis对数组所有元素求和
        cost = (-1.0 / m) * np.sum(y * np.log(a) + (1 - y) * (np.log(1 - a)))
        # print "cost:", cost
        dw = (1.0 / m) * np.dot(x, (a - y).T)
        db = (1.0 / m) * np.sum(a - y)
        grads = {'dw': dw, 'db': db}

        return grads, cost

    def sgd(self, x, y, print_cost=False):
        """
        :param x: data of shape (num_px * num_px * 3, number of examples)
        :param y: true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)


        :param print_cost: True to print the loss every 100 steps
        :return: 
        """
        costs = []
        grads = {}

        for index in range(self.num_iterations):
            grads, cost = self.propagate(x, y)
            dw = grads['dw']
            db = grads['db']
            self.w = self.w - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db

            if index % 10 == 0:
                costs.append(cost)
            if print_cost and index % 100 == 0:
                print "Cost after iteration %i: %s" % (index, cost)

        return grads, costs

    def fit(self, x, y, print_cost):
        return self.sgd(x, y, print_cost)

    def predict(self, x):
        m = x.shape[1]
        y = np.zeros((1, m))
        a = self.sigmoid(np.dot(self.w.T, x) + self.b)

        for index in range(m):
            if a[0, index] > 0.5:
                y[0, index] = 1
            else:
                y[0, index] = 0

        return y

    def score(self, x, y):
        return 100 - np.mean(np.abs(self.predict(x) - y)) * 100


def load_dataset(show=True):
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    clses = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    if show:
        fig = plt.figure(figsize=(50, 50))
        for index in range(209):
            y = fig.add_subplot(14, 15, index + 1)
            y.imshow(train_set_x_orig[index])
        plt.show()

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, clses


def pre_handle_img(file_name, num_pixel):
    image = np.array(ndimage.imread(file_name, flatten=False))
    plt.imshow(image)
    plt.show()
    my_image = imresize(image, size=(num_pixel, num_pixel)).reshape((1, num_pixel * num_pixel * 3)).T

    return my_image / 255.


if __name__ == '__main__':
    train_set_x, train_set_y, test_set_x, test_set_y, classes = load_dataset(show=False)
    num_px = train_set_x.shape[1]
    train_set_x = train_set_x.reshape(train_set_x.shape[0], -1).T
    test_set_x = test_set_x.reshape(test_set_x.shape[0], -1).T
    train_set_x = train_set_x / 255.
    test_set_x = test_set_x / 255.

    print "train dataset x shape:", train_set_x.shape
    print "train dataset y shape:", train_set_y.shape

    dimensions = train_set_x.shape[0]

    model = LogicRegression(dim=dimensions)
    try:
        with open(model.model_name, 'rb') as f:
            model = pickle.load(f)
    except Exception, e:
        model.fit(train_set_x, train_set_y, print_cost=True)

        # 序列化算法
        with open(model.model_name, 'wb') as f:
            pickle.dump(model, f)

    print model.score(train_set_x, train_set_y)
    print model.score(test_set_x, test_set_y)

    # 预测第25张图片
    # print model.predict(train_set_x[:, 25].reshape(12288, 1))

    # 预测自己的图片
    images = ["images/my_image.jpg", "images/my_image2.jpg", "images/cat_in_iran.jpg"]
    for img in images:
        img = pre_handle_img(img, num_px)
        if np.squeeze(model.predict(img)):
            print "It's a cat!"
        else:
            print "It's non-cat!"
