import numpy as np
import pandas as pd
import scipy.stats
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix, accuracy_score
import sys

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
            p_x_i_given_y = scipy.stats.norm(x_i_mean, x_i_std).pdf(new_data_point[i].values[0])
            prob_list_for_all_features.append(p_x_i_given_y)
        elif df[i].dtype == 'object' or df[i].dtype == 'int64':
            # if categorical, do the counting
            # using laplace smoothing to prevent zero prob.
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
        result[i] = predictOneInstance(df_test[i:(i+1)], prior_y_zero_over_train,
                                       prior_y_one_over_train, train_class_zero, train_class_one)

    return result



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


#################################################################################
###basic pace model developing
##using years shifting for prediction
def paceNB(marathon_pace, start_year_test, finish_year_test, predict_year_test):
    train = marathon_pace.ix[:,str(int(start_year_test)-1):str(int(finish_year_test)-1)].as_matrix()
    test = marathon_pace.ix[:,str(start_year_test):str(finish_year_test)].as_matrix()
    t_train = (marathon_pace.ix[:,str(int(predict_year_test)-1)] != 0).astype('int').as_matrix()
    t_test = (marathon_pace.ix[:,str(predict_year_test)] != 0).astype('int').as_matrix()

    resultPred = naiveBayes(train, t_train, test)
    acc = accuracy_score(y_true = t_test, y_pred = resultPred)
    return resultPred



def paceNB_no_label(marathon_pace, start_year_test, finish_year_test, predict_year_test):
    train = marathon_pace.ix[:,str(int(start_year_test)-1):str(int(finish_year_test)-1)].as_matrix()
    test = marathon_pace.ix[:,str(start_year_test):str(finish_year_test)].as_matrix()
    t_train = (marathon_pace.ix[:,str(int(predict_year_test)-1)] != 0).astype('int').as_matrix()

    resultPred = naiveBayes(train, t_train, test)

    return resultPred

def get_y1_naive_bayes(X):
	#load the dataset
	#X = run(sys.argv[1])
#X.shape
#X

#pre-processing the data

	df_X = pd.DataFrame(X)
	#print(df_X.head(n=5))
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
	result = paceNB_no_label(marathon_pace,2014,2016,2017)
	#print(type(np.array(result)))
	return np.array(result)

def main():
    get_y1_naive_bayes()
	
if __name__ == "__main__":
    main()
