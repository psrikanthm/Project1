from __future__ import division
import sys

import csv

import numpy
import scipy.linalg
import scipy.stats

import pre_process
import LinearRegression
import logistics
import nb_2017

import matplotlib.pyplot as plt

"""
0 - Gender
1 - Age
2 - Count of Races
3 - 2003 Rank
4 - 2003 Timing
5 - 2003 Pace
6 - 2004 Rank
7 - 2004 Timing
8 - 2004 Pace
9 - 2005 Rank
10 - 2005 Timing
11 - 2005 Pace
12 - 2006 Rank
13 - 2006 Timing
14 - 2006 Pace
15 - 2007 Rank
16 - 2007 Timing
17 - 2007 Pace
18 - 2008 Rank
19 - 2008 Timing
20 - 2008 Pace
21 - 2009 Rank
22 - 2009 Timing
23 - 2009 Pace
24 - 2010 Rank
25 - 2010 Timing
26 - 2010 Pace
27 - 2011 Rank
28 - 2011 Timing
29 - 2011 Pace
30 - 2012 Rank
31 - 2012 Timing
32 - 2012 Pace
33 - 2013 Rank
34 - 2013 Timing
35 - 2013 Pace
36 - 2014 Rank
37 - 2014 Timing
38 - 2014 Pace
39 - 2015 Rank
40 - 2015 Timing
41 - 2015 Pace
42 - 2016 Rank
43 - 2016 Timing
44 - 2016 Pace
"""
def construct_features(s,X,i):
    non_zeros = s[numpy.nonzero(s)]
    mean = numpy.mean(non_zeros)
    w = numpy.linspace(10,20,num=len(non_zeros))
    w = w/numpy.sum(w)
    average = numpy.average(non_zeros,weights=w)
    hmean = scipy.stats.hmean(non_zeros)
    #return ([X[i,0],X[i,1],X[i,2],mean,mean*mean,average,average*average,hmean,hmean*hmean],)
    return ([X[i,0],X[i,1],X[i,1]*X[i,1],X[i,2]*X[i,2],mean,mean**2,mean**3,mean**4,mean**5,mean**6,mean**7,mean**9,average,mean*X[i,1],mean*X[i,0]],)

def construct_data(x,X,i,j):
    if numpy.count_nonzero(x[i][:j]) != 0:
        # Extract all possible statistics of past data - mean, weighted mean etc
        features = construct_features(x[i][:j],X,i)
        return features + (x[i,j], )
    else:
        return None

def get_y2_new(X):
    # take all previous completion times
    XOrg = numpy.copy(X)
    numpy.random.shuffle(X)

    X[:,34] = X[:,34] * numpy.power(42195/21098, 1.06)
    XOrg[:,34] = XOrg[:,34] * numpy.power(42195/21098, 1.06)
    
    x = X[:,4]
    for i in range(2,int(45/3)):
        x = numpy.c_[x, X[:,3*i + 1]]
    
    xorg = XOrg[:,4]
    for i in range(2,int(45/3)):
        xorg = numpy.c_[xorg, XOrg[:,3*i + 1]]

    t1 = []
    t2 = []

    m,n = X.shape
    for i in range(m):
        # this is based on 2016, follow the same for 2015, 2014, 2013....
        if x[i,-1] != 0:
            a = construct_data(x,X,i,-1)
            if a:
                t1.append(a[0])
                t2.append(a[1])
        elif x[i,-2] != 0:
            a = construct_data(x,X,i,-2)
            if a:
                t1.append(a[0])
                t2.append(a[1])
        elif x[i,-3] != 0:
            a = construct_data(x,X,i,-3)
            if a:
                t1.append(a[0])
                t2.append(a[1])
        elif x[i,-4] != 0:
            a = construct_data(x,X,i,-4)
            if a:
                t1.append(a[0])
                t2.append(a[1])
        elif x[i,-5] != 0:
            a = construct_data(x,X,i,-5)
            if a:
                t1.append(a[0])
                t2.append(a[1])
        elif x[i,-6] != 0:
            a = construct_data(x,X,i,-6)
            if a:
                t1.append(a[0])
                t2.append(a[1])
        elif x[i,-7] != 0:
            a = construct_data(x,X,i,-7)
            if a:
                t1.append(a[0])
                t2.append(a[1])
 #       elif x[i,-8] != 0:
 #           a = construct_data(x,X,i,-8)
 #           if a:
 #               t1.append(a[0])
 #               t2.append(a[1])
 #       elif x[i,-9] != 0:
 #           a = construct_data(x,X,i,-9)
 #           if a:
 #               t1.append(a[0])
 #               t2.append(a[1])

    t1 = numpy.array(t1)
    t2 = numpy.array(t2)
    
    # normalise the predictor (input) variables before anything else
    t1 = t1/t1.max(axis=0)

    #fig = plt.figure()
    #plt.plot(t1[:,-2],t2/t2.max(),'k^',t1[:,5],t1[:,5],'r--')

    #fig.suptitle('Finishing Time vs Weighted Mean of past Finishing Times', fontsize=20)
    #plt.xlabel('Weighted Mean of past Finishing Times', fontsize=16)
    #plt.ylabel('Most recent Finishig Time', fontsize=16)

    #plt.show()
    
    print "#1: ", len(t1)
    #training - 90 %, testing - 10 %
    # measure correlations with different features
    index = int(.9 * len(t1))
    
    x_train = t1[:index,:]
    y_train = t2[:index]
    
    #model = LinearRegression.LinearRegression(x, y, alpha = 0.5, lam = 0, printIter=False)
    #model.run(1000)

    #xPred = model.predict(x)
    #print numpy.linalg.norm(xPred - y) / numpy.linalg.norm(y)
    
    #plt.plot(y,xPred,'k^',y,y,'r--')
    #plt.show()

    x_test = t1[index:,:]
    y_test = t2[index:]
    
    ''' 
    m = len(x_train)
    nfold = m
    foldSize = int(m / nfold)

    
    # arrage to store training and testing error
    trainErr = numpy.array([0.0] * nfold)
    cvErr = numpy.array([0.0] * nfold)
    lamVector = [0.0] * nfold
    allIndex = range(0, m)
    lam = 0
    
    for i in range(0, nfold):
        print "iteration = ", i, "lamda = ", lam
        cvIndex = range((foldSize * i), foldSize * (i + 1))
        trainIndex = list(set(allIndex) - set(cvIndex))

        trainX = x_train[trainIndex, :]
        trainY = y_train[trainIndex]
        cvX = x_train[cvIndex, :]
        cvY = y_train[cvIndex]
        # set parameter
        alpha = 0.5
        lam += 0.6
        model = LinearRegression.LinearRegression(trainX, trainY, alpha, lam,  printIter=False)
        model.run(100)

        trainPred = model.predict(trainX)
        trainErr[i] = numpy.linalg.norm(trainPred - trainY) / numpy.linalg.norm(trainY)
        #trainErr[i] = ((trainPred-trainY)**2).sum()

        cvPred = model.predict(cvX)
        cvErr[i] = numpy.linalg.norm(cvPred - cvY) / numpy.linalg.norm(cvY)
        #cvErr[i] = ((cvPred-cvY)**2).sum()

        lamVector[i] = lam
        print "lam=",lam,"train Err=", trainErr[i], "cvErr=", cvErr[i]
    
    print "trainErr = ", sum(trainErr)/len(trainErr), "cvErr = ", sum(cvErr)/len(cvErr)

    fig, ax = plt.subplots()
    ax.plot(lamVector, trainErr, 'k^-', label='Training Error', linewidth=3)
    ax.plot(lamVector, cvErr, 'k^:', label='Validation Error', linewidth=3)

    fig.suptitle('Regularization', fontsize=20)
    plt.xlabel('hyper-parameter (lamda)', fontsize=16)
    plt.ylabel('Error', fontsize=16)
   
    legend = ax.legend(loc='upper center', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    
    plt.show()
    ''' 
    lam = 470
    alpha = 0.5
    model = LinearRegression.LinearRegression(x_train, y_train, alpha, lam, printIter=False)
    model.run(500)

    trainPred = model.predict(x_train)
    trainErr = numpy.linalg.norm(trainPred - y_train) / numpy.linalg.norm(y_train)    

    testPred = model.predict(x_test)
    testErr = numpy.linalg.norm(testPred - y_test) / numpy.linalg.norm(y_test)

    print "testErr: ", testErr, " trainErr: ", trainErr

    fig = plt.figure()
    plt.plot(y_test, testPred, 'k^',y_test,y_test,'r--')
    fig.suptitle('Validation Set Results', fontsize=20)
    plt.xlabel('Actual Finishing time (in seconds)', fontsize=12)
    plt.ylabel('Predicted Finishing time (in seconds)', fontsize=12)

    plt.show()
    
    m,n = XOrg.shape
    x_final = []
    for i in range(m):
        f, = construct_features(xorg[i,:], XOrg, i)
        x_final.append(f)
    x_final = numpy.array(x_final)
    x_final = x_final / x_final.max(axis = 0)

    print "#debug: feature construction complete"
    finalPred = model.predict(x_final)
    print "#debug: prediction complete"
   
    s = []
    for i in range(len(finalPred)):
        s.append(convert_seconds(finalPred[i]))

    return s

def convert_seconds(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)

def get_y1_logistic_regression(X):
    pass
    return 0

def naive_bayes(X):
    pass
    return 0

def dummy_fn(X):
    x = X[:,2]
    for i in range(1,int(45/3)):
        x = numpy.c_[x, X[:,3*i + 2]]
    
    m,n = x.shape
    t = []
    for i in range(m):
        s = numpy.array(numpy.nonzero(x[i,1:]))
        if x[i,0] > 1:
            t.append(numpy.min(numpy.diff(numpy.sort(s))))

    #plt.hist(t,bins=20)
    t = numpy.array(t)
    d = numpy.diff(numpy.unique(t)).min()
    left_of_first_bin = t.min() - float(d)/2
    right_of_last_bin = t.max() + float(d)/2
    plt.title('Distribution of Maximum difference between consecutive runs', fontsize=20)
    plt.hist(t, numpy.arange(left_of_first_bin, right_of_last_bin + d, d))
    plt.xlabel("Maximum difference between consecutive runs", fontsize=16)
    plt.ylabel("Number of Runners", fontsize=16)
    plt.show()
 

def data_analysis(X):
    y = X[:,40]  # train with 2015 timing
    w = X [:,41] # train with 2015 pace

    x = X[:,4]   # take all previous completion times
    for i in range(2,int(45/3)):
        x = numpy.c_[x, X[:,3*i + 1]]

    z = X[:,5]   # take all previous race paces
    for i in range(2,int(45/3)):
        z = numpy.c_[z, X[:,3*i + 2]]

    s = []
    m,n = x.shape
    for i in range(m):
        s.append(numpy.sum(x[i])/numpy.count_nonzero(x[i]))

    t = []
    for i in range(m):
        t.append(numpy.sum(z[i])/numpy.count_nonzero(z[i]))

    for i in range(len(y)):
        if y[i] == 0:
            y[i] = s[i]
 
    for i in range(len(w)):
        if w[i] == 0:
            w[i] = t[i]

    #plt.hist(w)
    plt.plot(w,y,'r^')
    plt.show()

    #Peter Riegel's formula to convert semi marathon timings to marathon timings
    #X[:,34] = X[:,34] * numpy.power(42195/21098, 1.06)

    #x = X[:,4]   # take all previous years 
    #for i in range(2,int(40/3)):
    #    x = numpy.c_[x, X[:,3*i + 1]]

    #s = numpy.mean(x, axis=1)
    #for i in range(len(y)):
    #    if y[i] == 0:
    #        y[i] = s[i]



def main():

    X = pre_process.run(sys.argv[1])
    dummy_fn(X)
    #data_analysis(X)
    #Y2_LR = get_y2_linear_regression(X)    
    '''
    Y2_LR = get_y2_new(X)
    
    Y1_LR = logistics.run(X,0)
    Y1_LR = Y1_LR.tolist()
    Y1_NB = nb_2017.get_y1_naive_bayes(X)
        
    ofile  = open('prediction.csv', "wb")
    writer = csv.writer(ofile, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
     
    for i in range(len(X)):
        row = []
        row.append(i+1)
        #row.append(Y1_LR[i])
        #row.append(Y1_NB[i])
        row.append(Y2_LR[i])
        writer.writerow(row)
 
    ofile.close()
    '''

if __name__ == "__main__":
    main()
