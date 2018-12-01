import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from xgboost import XGBRegressor
from sklearn import model_selection
from sklearn import datasets, cluster
from sklearn.impute import SimpleImputer
from sklearn import random_projection


initial_date = datetime.datetime(1980, 1, 1)

def date_parser(vector):
    vector[0] = (datetime.datetime.strptime(vector[0], "%Y-%m-%d") - initial_date).days
    if isinstance(vector[18], str):
        vector[18] = (datetime.datetime.strptime(vector[18], "%Y-%m-%d") - initial_date).days
    return vector

def performance_metric(actual, predicted):
    return 1 - sum(abs((actual - predicted) / actual)) / actual.shape[0]

def featAgg(dat, nclust=18):
	agglo = cluster.FeatureAgglomeration(n_clusters=nclust)
	agglo.fit(dat)
	return agglo.transform(dat)

def sparseRdProj(dat, val, ncomp=20):
	transformer = random_projection.SparseRandomProjection(n_components=ncomp)
	return transformer.fit_transform(dat, y=val)	


data = pd.read_csv("./data/data_train.csv", parse_dates=True, header=None).values
data = np.asarray(list(map(lambda x: date_parser(x), data)))
data = data.astype(float)
X = data[:, :-1]
Y = data[:, -1]

###inputer
inp = SimpleImputer(np.nan)
X_inp = inp.fit_transform(X)

###dimensionality reduction
#X_proj = sparseRdProj(X_inp, Y, 20)
X_proj = featAgg(X_inp, 18)

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X_proj, Y, test_size=0.2)

model = XGBRegressor(max_depth=13, learning_rate=0.097, min_child_weight=0, reg_lambda=0.005, random_state=39 )
model.fit(X_train, Y_train)

predictions = model.predict(X_test)

print(performance_metric(Y_test, predictions))