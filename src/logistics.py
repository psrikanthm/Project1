import numpy as np
import math
import matplotlib.pyplot as plt

def sigmoid(z):
    s = 1.0 / (1 + np.exp(-z))
    return s

def hypothesis(X, theta):
    h = sigmoid(np.dot(X, theta))
    return h

# compute the cost
def cost_function(X, Y, theta, regularize, lam):
    m, n = X.shape
    hypo = hypothesis(X, theta)
    ones_vector = np.array([1] * len(Y)).reshape(m, 1)
    cost = 1.0 * (-1 / m) * (np.dot(Y.T, np.log(hypo)) + 1.0 * np.dot((ones_vector - Y).T, np.log(1 - hypo)))
    if regularize == 1:
        norm2_theta = 1.0 * np.linalg.norm(theta[1:])
        cost = cost + lam / (2*m) * norm2_theta ** 2
    #print 'cost', cost
    return cost

def gradient_descent(X, y, theta, alpha, regularize, lambda_try):
    m, n = X.shape
    h = hypothesis(X, theta) - y
    err_sum = np.dot(X.T, h)
    if regularize == 1:
        # L2 regularization
        update_theta = theta * (1 - lambda_try/m) - alpha * err_sum / m

    elif regularize == 0:
        update_theta = theta - alpha * err_sum / m

    return update_theta

# logistic regression of raw features with & without regularization
def logistic_regression(X, alpha, theta, iters, regularize):
    fold = 2
    # with regularization
    if regularize == 1:

        lam_num = 6
        cost_train = np.zeros(shape=(fold, lam_num))
        cost_validate = np.zeros(shape=(fold, lam_num))

        penalty_values = np.arange(0.001, 0.008, lam_num)
        print '++++++   lambda   space    +++++'
        print penalty_values
        print
        th_lam = 0
        lam = np.zeros(shape=(lam_num,1))
        # out loop = try diff lambda
        for lambda_try in penalty_values:
            # process_theta = theta
            # middle loop = try on diff validation set and store cost (opt theta) to index column
            for j in range(fold):
                process_theta = theta
                current_predict_year = 2016 - j - 1
                lam[th_lam] = lambda_try
                print '********* down - predict *********', current_predict_year

                index1 = append_index(current_predict_year, 3)
                X1, Y1 = count_and_label(X, current_predict_year)
                X1 = X1[:, index1]
                index2 = append_index(current_predict_year + 1, 3)
                X2, Y2 = count_and_label(X, current_predict_year+1)
                X2 = X2[:, index2]
                # inner loop = get the theta at local optimal + check the validation error
                for i in range(iters):
                    new_theta = gradient_descent(X1, Y1, process_theta, alpha, regularize, lambda_try)
                    process_theta = new_theta
                    c1 = cost_function(X1, Y1, process_theta, regularize, lambda_try)
                    c2 = cost_function(X2, Y2, process_theta, regularize, lambda_try)
                print 'opt theta ', new_theta, 'medium loop= ', j, th_lam

                cost_train[j, th_lam] = c1
                cost_validate[j, th_lam] = c2
                print 'train cost= ', c1, 'valid cost= ', c2, 'when lambda= ', lambda_try, 'in', current_predict_year
            th_lam += 1

        print cost_validate
        print '******* up  mean_validation_error   ****** '
        m_cost_valid = np.mean(cost_validate, axis=0)
        print m_cost_valid
        print '******* up  mean_train_error   ****** '
        m_cost_train = np.mean(cost_train, axis=0)
        print m_cost_train

        lamba_index_min_cost = m_cost_valid.argmin()
        print lamba_index_min_cost

        plt.plot(lam, m_cost_valid, 'ro', lam, m_cost_train, 'g^')
        plt.axis([0, 0.01, 1000, max(m_cost_valid.max(), m_cost_train.max())])
        plt.title("Error in Logistics Regression with regularization")
        plt.xlabel("Lambda")
        plt.ylabel("Frequency")
        plt.show()

    # without regularization
    elif regularize == 0:
        # y = 2015's labels
        X, y = count_and_label(X, 2015)
        ind = [0, 1, 2, 3, 34, 35, 36, 37, 38, 39, 40, 41, 42]
        X = X[:,ind]


        for i in range(iters):
            new_theta = gradient_descent(X, y, theta, alpha, regularize, 0)
            theta = new_theta


    return theta

# compute the accuracy given hypothesis parameters(theta and X,) and labels
def accuracy(X, y, theta):
    h = hypothesis(X, theta)
    correct = 0
    for i in range(len(y)):
        if h[i] >= 0.5:
            h[i] = 1
        else:
            h[i] = 0
    for i in range(len(y)):
        if h[i] == y[i]:
            correct += 1
    ac = 1. * correct / len(y)
    print 'prediction sum ', h.sum()
    return ac

# counting the participating times before the prediction year + give the label of that year
def count_and_label(X, year):
    m, n = X.shape
    a = 2016 - year + 1
    Y = np.array([0]*m).reshape(m, 1)
    for i in range(m):
        if X[i, 43 - 3*(2016-year)] != 0:
            Y[i] = 1
        for y in range(a):
            j = 43 - 3*y
            if X[i, j] != 0:
                X[i, 3] = X[i, 3] - 1

    return X, Y

# year to predict, n is # of previous years of data used to predict
def append_index(year, n):

    base_index = [0, 1, 2, 3]
    index = []
    index.extend(base_index)

    end_index = 43 - 3 * (2016 - year)
    start_index = end_index - 3 * n
    index_append = list(range(start_index, end_index))
    index.extend(index_append)

    return index

# validation
def validation(X, theta, year):

    index2 = append_index(year)
    X2, Y2 = count_and_label(X, year)
    X2 = X2[:, index2]
    accuracy_rate = accuracy(X2, Y2, theta)
    print 'accuracy rate ', accuracy_rate


def prediction(X, theta):
    m,n = X.shape
    h = np.array(hypothesis(X, theta))

    for i in range(m):
        if h[i] >= 0.26:
            h[i] = 1
        else:
            h[i] = 0

    h = h.ravel()
    return h

def run(X, regularization):

    m, n = X.shape
    const = np.array([1] * m).reshape(m, 1)
    X = np.append(const, X, axis=1)
    X = 1.0 * X / X.max(axis=0)

    # initials of training
    alpha = 1
    iterations = 1500
    m, n = X.shape
    initial_theta = 2 * np.random.rand(13, 1)
    print 'initial random theta '
    print initial_theta
    theta = logistic_regression(X, alpha, initial_theta, iterations, regularization)

    ind = [0,1,2,3,37,38,39,40,41,42,43,44,45]

    p = prediction(X[:, ind], theta)
    print p
    print p.shape
    return p
