from __future__ import division

import random
import numpy
class LinearRegression:

    # initialize

    def __init__(self, X, Y, alpha=0.0005, lam=0.1, printIter=True):
        x = numpy.array(X)
        m, n = x.shape
        # normalize data
            #self.xMean = numpy.mean(x, axis=0)
            #self.xStd = numpy.std(x, axis=0)
            #x = (x - self.xMean) / self.xStd
        #x = x / x.max(axis=0)
        # add const column to X
        const = numpy.array([1] * m).reshape(m, 1)
        self.X = numpy.append(const, x, axis=1)
        self.Y = numpy.array(Y)
        #self.Y = self.Y / self.Y.max()
        self.alpha = alpha
        self.lam = lam
        self.theta = numpy.array([0.0] * (n + 1))
        self.printIter = printIter

    # caluclate cost
    def __costFunc(self):
        "calculate sum square error"
        m, n = self.X.shape
        pred = numpy.dot(self.X, self.theta)
        #pred = numpy.mean(self.X, axis=1)
        err = pred - self.Y
        cost = sum(err ** 2) / (2 * m) + self.lam * \
            sum(self.theta[1:] ** 2) / (2 * m)
        return(cost)

    def errorFunc(self,y1,y2):
        err = numpy.array(y1) - numpy.array(y2)
        m = len(y1) 
        cost = sum(err ** 2) / (2 * m)
        return cost

    # gradient descend
    def __gradientDescend(self, iter):
        """
        gradient descend:
        X: feature matrix
        Y: response
        theta: predict parameter
        alpha: learning rate
        lam: lambda, penality on theta
       """

        m, n = self.X.shape

        for i in range(0, iter):
            # update theta[0]
            pred = numpy.dot(self.X, self.theta)
            err = numpy.array(pred - self.Y)
           
            gradient = numpy.dot(numpy.transpose(self.X),err) + self.lam * self.theta
            self.theta = self.theta - self.alpha * gradient * (1/m)

            cost = self.__costFunc()
            #pred = numpy.mean(self.X, axis=1)
            if self.printIter:
                print "Iteration", i, "\tcost=", cost
                # print "theta", self.theta
    
    
    # simple name
    def run(self, iter):
        self.__gradientDescend(iter)

    # prediction
    def predict(self, X):

        # add const column
        m, n = X.shape
        x = numpy.array(X)
        #x = (x - self.xMean) / self.xStd
        #x = x / x.max(axis=0)
        const = numpy.array([1] * m).reshape(m, 1)
        X = numpy.append(const, x, axis=1)
        pred = numpy.dot(X, self.theta)
        #print "#1:", pred, X, self.theta
        #print "#2:", self.theta
        return pred
