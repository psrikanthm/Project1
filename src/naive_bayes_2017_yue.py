import numpy as np
import pandas as pd
import scipy.stats
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix, accuracy_score


def priorY(data0_1):
    ## this function calculates the prior p(y)
    # we calculate two such priors: p(y=0) and p(y=1) in the training data
    n_0_1=np.count_nonzero(data0_1)
    n_total=data0_1.shape[0]
    ratio_as_p_y=n_0_1/n_total
    return ratio_as_p_y


def dataSeparation(train,t_train):
    ## this function separate data into data0 and data1 based on labels(0 or 1)
    #return data0_1
    class_zero_data = train[t_train == 0]
    class_one_data = train[t_train != 0]
    return class_zero_data,class_one_data


def priorY(t_train):
    ## this function calculates the prior p(y)
    # we calculate two such priors: p(y=0) and p(y=1) in the training data
    n_1=np.count_nonzero(t_train)
    n_total=len(t_train)
    ratio_class_one_over_total=n_1/n_total
    ratio_class_zero_over_total=1-ratio_class_one_over_total
    return ratio_class_zero_over_total,ratio_class_one_over_total


def likelihood(class_separate_data,new_data_point):
    ## this function calculates p(x|y)
    # we calculate p(x|y) separated based on the class labels
    # therefore, we calculate p(x|y=0) or p(x|y=1)
    df=pd.DataFrame(class_separate_data)
    # data types are inferred
    # compute a list with p(x_i|y=0) or p(x_i|y=1) for all i
    prob_list_for_all_features = []

    for i in df.columns:
        #print("feature",i)
        #look at each feature, calculate p(x_i|y=0) or p(x_i|y=1)
        if df[i].dtype == 'float64':
            #check the data type, if continuous, assume gaussian distribution
            x_i_mean = df[i].mean()
            x_i_std = df[i].std()
            #print("mean",x_i_mean,"std", x_i_std,"point", new_data_point[i].values[0])
            p_x_i_given_y = scipy.stats.norm(x_i_mean, x_i_std).pdf(new_data_point[i].values[0])
            #print(p_x_i_given_y)
            prob_list_for_all_features.append(p_x_i_given_y)
            #print("list:",prob_list_for_all_features)
        elif df[i].dtype == 'object' or df[i].dtype == 'int64':
            # if categorical, do the counting
            # using laplace smoothing to prevent zero prob.
            #print("df.value.counts", df[i].value_counts())
            #print("df.value.counts [new_data_point[i] ]",df[i].value_counts()[new_data_point[i]])
            #print("df.value.counts [new_data_point[i].values[0]]",df[i].value_counts()[new_data_point[i]])
            flag = df[i].isin([new_data_point[i].values[0]]).any()
            if flag == False:
                p_x_i_given_y = 1/(len(df[i])+2)
            else:
                p_x_i_given_y = (df[i].value_counts()[new_data_point[i].values[0]]+1)/(len(df[i])+2)
            #print(p_x_i_given_y)
            prob_list_for_all_features.append(p_x_i_given_y)
            #print("list:",prob_list_for_all_features)
        else:
            print("feature",i,"does not have the right data type")

    ### Naive assumption: features are conditionally independent
    # we take the product of all p(x_i|y=0)
    #print(prob_list_for_all_features)
    prob_x_given_y=np.prod(prob_list_for_all_features)
    return prob_x_given_y


def ratioPosterior(likelihood_0, likelihood_1, prior_y_0, prior_y_1):
    ## note the posterior p_0 = p(y=0|x)=p(x|y=0)p(y=0)/p(x)
    # and p_1 = p(y=1|x)=p(x|y=1)p(y=1)/p(x)
    # we predict y=0 if p_0 > p_1 or p_0/p_1>1
    #thus, p(x|y=0)p(y=0) / p(x|y=1)p(y=1)>1
    ratio_posterior = (likelihood_0 * prior_y_0) / (likelihood_1 * prior_y_1)

    if ratio_posterior > 1:
        return 0
    else:
        return 1

def predictOneInstance(new_data_point, prior_y_0, prior_y_1, train_zero_data, train_one_data):
    #predict for a random test data point
    #print("train_zero_data", train_zero_data)
    likelihood_0 = likelihood(train_zero_data, new_data_point)
    likelihood_1 = likelihood(train_one_data, new_data_point)

    predOne = ratioPosterior(likelihood_0, likelihood_1, prior_y_0, prior_y_1)
    return predOne



def naiveBayes(train, t_train, test):
    ## this function test a naive bayes model for multiple test data instances
    # separate training data by classes
    train_class_zero, train_class_one = dataSeparation(train, t_train)
    #compute prior p(y=0)& p(y=1) for all training data
    prior_y_zero_over_train, prior_y_one_over_train = priorY(t_train)

    result=[0]*len(test)
    df_test = pd.DataFrame(test)

    for i in range(len(test)):
        #print("this is",i, "th test data")
        result[i] = predictOneInstance(df_test[i:(i+1)], prior_y_zero_over_train,
                                       prior_y_one_over_train, train_class_zero, train_class_one)

    return result


##################################################
## test on iris dataset
from sklearn import datasets
# import some data to play with
iris = datasets.load_iris()
X = iris.data
Y = iris.target
train, test, t_train, t_test = cross_validation.train_test_split(X, Y, test_size=0.2, random_state=0)

resultPred = naiveBayes(train, t_train, test)
confusion_matrix(y_true = t_test, y_pred = resultPred)
#note our model is only for binary classification, so the third class is not predicted
# code can be modified easily for multi-classes classification

## test regular Naive Bayes from sklearn
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(train, t_train).predict(test)
confusion_matrix(y_true = t_test, y_pred = y_pred)

####################################################
####################################################
## marathon data reading and pre-pocessing

###Sri's code for pre-process the data
import csv

import numpy

def convert_pace(PaceStr):
    m, s = [int(i) for i in PaceStr.split(':')]
    return 60*m + s

def convert_time(TimeStr):
    h, m, s = [int(i) for i in TimeStr.split(':')]
    return 3600*h + 60*m + s

def categorize_age(Age):
    return Age - (Age%5)

def run(Filename):
    ifile  = open(Filename, "rt")
    rows = csv.reader(ifile)
    data = {}

    next(rows)
    for row in rows:

        age = categorize_age(int(row[2]))
        rank = int(row[4])
        timing = convert_time(row[5])
        pace = convert_pace(row[6])

        Id = int(row[0])
        if row[3] == 'M':
            gender = 1
        else:
            gender = 0

        d = (age, rank, timing, pace)

        year = int(row[7])

        if Id in data:
            (gender, _, races) = data[Id]
            races[year] = d
            data[Id] = (gender, age, races)
        else:
            races = {}
            races[year] = d
            data[Id] = (gender, age, races)

    for attribute, (g, a, val) in data.items():
        data[attribute] = (g, a, len(val), val)

    #x = numpy.zeros(shape=(len(data), 45))

    x = [[0 for col in range(45)] for row in range(len(data) + 10) ]
    for attribute, value in data.items():
        features = [0 for col in range(45)]
        features[0] = value[0]
        features[1] = value[1]
        features[2] = value[2]

        for year, t in value[3].items():
            if year >= 2003:
                index = (year - 2003) * 3 + 3
                features[index] = t[1]
                index += 1
                features[index] = t[2]
                index += 1
                features[index] = t[3]

        x[attribute] = features

    s = numpy.array(x)
    return s


###read the data and pre-process with Sri's code
X = run("data2016.csv")
X
#X.shape
#X

### convert the data into dataFrame in Pandas
df_X = pd.DataFrame(X)

# convert finising time and pace to continuous value
df_X.ix[:,4:45:3] = df_X.ix[:,4:45:3].astype('float64')
df_X.ix[:,5:45:3] = df_X.ix[:,5:45:3].astype('float64')

## make a copy of the data for future use
df_test = df_X

#look at the correlations of the whole dataset, realize that rank, time, and pace are highly
#correlated in each year
df_test.corr(method='pearson', min_periods=1)

#using all features for NB only achieve 66% accuracy
#convert rank in 2016, as long as the rank is not zero, the person attend the marathon
marathon_Y = df_test.ix[:,42] != 0
marathon_Y = marathon_Y.astype('int')
marathon_X = df_test.drop([42,43,44],axis=1)

train, test, t_train, t_test = cross_validation.train_test_split(marathon_X, marathon_Y, test_size=0.2, random_state=0)

resultPred = naiveBayes(train, t_train, test)
confusion_matrix(y_true = t_test, y_pred = resultPred)

accuracy_score(y_true = t_test, y_pred = resultPred)
#resultPred
#0.66

## drop rank and time for prediction gives the best results
###this is slicing the training and testing data horizontally for 2016

## without time of participate column

#######try to imporove accuracy by ranks
# idea: notice that ranks are relative and discrete, we categorize it by 100
ls1 = list(range(0,2))
ls2 = list(range(3,41,3))

marathon_M_1 = marathon_X.ix[:,ls1]
marathon_M_2 = np.ceil((marathon_X.ix[:,ls2])/100).astype('int')
#marathon_M_2.describe()
marathon_X_3 = pd.concat((marathon_M_1,marathon_M_2), axis=1)
#dftot

train, test, t_train, t_test = cross_validation.train_test_split(marathon_X_3, marathon_Y, test_size=0.2, random_state=0)

resultPred = naiveBayes(train, t_train, test)
confusion_matrix(y_true = t_test, y_pred = resultPred)

accuracy_score(y_true = t_test, y_pred = resultPred)
#resultPred

#0.90239894840617807


#experiment with only time or pace, get similar performance, pace gives the best results
### therefore, extracting a dataset with only paces from each year
# rename all features

#test on marathon dataset
#load the dataset
X = run("data2016.csv")
#X.shape
#X

#pre-processing the data

df_X = pd.DataFrame(X)
#df_X = df_X.astype('float64')
df_X.ix[:,3:45:3].dtypes
df_X.ix[:,4:45:3] = df_X.ix[:,4:45:3].astype('float64')
df_X.ix[:,5:45:3] = df_X.ix[:,5:45:3].astype('float64')
#df_X.ix[:,4:45:3].dtypes
#df_X.dtypes

marathon_X = df_X
### marathon pace last three years model
ls1 = list(range(0,2))
ls2 = list(range(5,45,3))

marathon_M_1 = marathon_X.ix[:,ls1]
marathon_M_2 = np.ceil((marathon_X.ix[:,ls2])/100).astype('int')
#marathon_M_2.describe()
marathon_pace = pd.concat((marathon_M_1,marathon_M_2), axis=1)
marathon_pace.columns=['gender','age','2003','2004','2005','2006','2007','2008',
                       '2009','2010','2011','2012','2013','2014','2015', '2016']
#marathon_pace


###basic pace model developing
##using years shifting for prediction
def paceNB(data, start_year_test, finish_year_test, predict_year_test):
    train = marathon_pace.ix[:,str(int(start_year_test)-1):str(int(finish_year_test)-1)].as_matrix()
    test = marathon_pace.ix[:,str(start_year_test):str(finish_year_test)].as_matrix()
    t_train = (marathon_pace.ix[:,str(int(predict_year_test)-1)] != 0).astype('int').as_matrix()
    t_test = (marathon_pace.ix[:,str(predict_year_test)] != 0).astype('int').as_matrix()
    #print("train", train.head(n=3))
    #print("test", test.head(n=3))
    #print("t_train",t_train.head(n=3))
    #print("t_test",t_test.head(n=3))

    resultPred = naiveBayes(train, t_train, test)
    #confusion_matrix(y_true = t_test, y_pred = resultPred)
    acc = accuracy_score(y_true = t_test, y_pred = resultPred)
    print("year predicted",predict_year_test, "accuracy:", acc)
    return resultPred


# 5 years for prediction
# using 2009-2013 for train, 2014 for train labels, 2010-2014 for train, and 2015 for test labels
#accu racy year predicted 2015 accuracy: 0.859927038486
result_5_years = paceNB(marathon_pace,2010,2014,2015)

# 3 years for prediction
# using 2011-2013 for train, 2014 for train labels, 2012-2014 for train, and 2015 for test labels
#year predicted 2015 accuracy: 0.858645282151
#three years for prediction
result = paceNB(marathon_pace,2012,2014,2015)

## all years for prediction
result = paceNB(marathon_pace,2004,2014,2015)

def paceNB_no_label(data, start_year_test, finish_year_test, predict_year_test):
    train = marathon_pace.ix[:,str(int(start_year_test)-1):str(int(finish_year_test)-1)].as_matrix()
    test = marathon_pace.ix[:,str(start_year_test):str(finish_year_test)].as_matrix()
    t_train = (marathon_pace.ix[:,str(int(predict_year_test)-1)] != 0).astype('int').as_matrix()
    #t_test = (marathon_pace.ix[:,str(predict_year_test)] != 0).astype('int').as_matrix()
    #print("train", train.head(n=3))
    #print("test", test.head(n=3))
    #print("t_train",t_train.head(n=3))
    #print("t_test",t_test.head(n=3))

    resultPred = naiveBayes(train, t_train, test)
    #confusion_matrix(y_true = t_test, y_pred = resultPred)
    #acc = accuracy_score(y_true = t_test, y_pred = resultPred)
    #print("year predicted",predict_year_test, "accuracy:", acc)
    return resultPred

# all years for predicting 2017
result = paceNB_no_label(marathon_pace,2004,2016,2017)
pd.DataFrame(result).describe()

# 3 years for predicting 2017
result = paceNB_no_label(marathon_pace,2014,2016,2017)
result.describe()
pd.DataFrame(result).describe()