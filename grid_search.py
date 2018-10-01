# This file implements the grid search method explained in the paper

### Parameters to change in order to reproduce results ###
# city, choose between 'anaheim' or 'oakland'
# test_cluster_indexes choose between 'cluster_0_indexes', 'cluster_1_indexes', 'cluster_2_indexes'
# parameters in run() function
##########################################################

from datetime import datetime 

import datetime as dt
import time
import tensorflow as tf
import numpy as np
import pandas as pd
import sympy as sp
import os
import random
import math
import threading
import matplotlib.pyplot as plt
import sys
import pickle

from keras import optimizers
from keras import backend as K
from keras.callbacks import Callback
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, LSTM, Dropout, Reshape, BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.optimizers import RMSprop, SGD, Adam
from keras.utils.vis_utils import plot_model
from keras import backend as K
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import EarlyStopping
from numpy import genfromtxt
from sklearn.metrics import pairwise_distances_argmin_min, mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC, LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from math import floor


city = 'anaheim'
print (sys.argv)


def get_input_cluster(input_dim, output_dim, samples_per_day, cluster_c_indexes, root, weekdays=True, weekends=False, file_order=None, load_clean=True, normalized=False, test=True, load_from=None):
    x_train_total = np.array([]).reshape(0, input_dim)
    x_val_total = np.array([]).reshape(0, input_dim)
    x_test_total = np.array([]).reshape(0, input_dim)
    
    y_train_total = np.array([]).reshape(0, output_dim)
    y_val_total = np.array([]).reshape(0, output_dim)
    y_test_total = np.array([]).reshape(0, output_dim)
    
    X_total = np.array([]).reshape(0, input_dim)
    Y_total = np.array([]).reshape(0, output_dim)

    train_size = 0.8
    weekday_days = 20
    criteria = train_size*weekday_days*samples_per_day
    
    for i in cluster_c_indexes:
        
        if(load_clean):
            x_train, x_val, x_test, y_train, y_val, y_test = load_clean_data(i, weekdays, weekends, root=root, normalized=normalized, test=test, load_from=load_from)
            if(np.concatenate((y_train, y_val)).shape[0] != criteria): # some time series contain less samples
                continue

        x_train_total = np.concatenate((x_train_total, x_train), axis=0)
        x_val_total = np.concatenate((x_val_total, x_val), axis=0)
        if test:
            x_test_total = np.concatenate((x_test_total, x_test), axis=0)
        else:
            x_test_total = np.array([])    

        y_train_total = np.concatenate((y_train_total, y_train), axis=0)
        y_val_total = np.concatenate((y_val_total, y_val), axis=0)
        if test:
            y_test_total = np.concatenate((y_test_total, y_test), axis=0)
        else:
            y_test_total = np.array([])
    
    # print(y_train.shape, y_val.shape, np.concatenate((y_train, y_val)).shape)
    return x_train_total, x_val_total, x_test_total, y_train_total, y_val_total, y_test_total

def get_clusters(random, root):

    filepath = root + "file_labels.csv"
    df = pd.read_csv(filepath, index_col=0, header=None, names=['file_name', 'TrafficLevelAvg'])
    try:
        df = df.drop([None])
    except:
        print('INFO: no NONE found in file_labels when getting clusters')
    df.index = df.index.astype(np.int64)
    
    if(random):
        print('--USING RANDOM CLUSTERS--')
        indexes = np.arange(len(df.index))
        np.random.shuffle(indexes)
        segmentation_size = len(indexes) // 3
        low, medium, high = np.split(indexes, [segmentation_size, segmentation_size*2])
        print(len(low), len(medium), len(high))
    else:
        high = df[df['TrafficLevelAvg'] == 'high'].index
        medium = df[df['TrafficLevelAvg'] == 'medium'].index
        low = df[df['TrafficLevelAvg'] == 'low'].index
        
    return low, medium, high

cluster_0_indexes, cluster_1_indexes, cluster_2_indexes = get_clusters(random=False, root='dataset\\' + city + '\\') #TODO:change accordingly
test_cluster_indexes = cluster_2_indexes

cluster_c_indexes = list(cluster_0_indexes) + \
                    list(cluster_1_indexes) + \
                    list(cluster_2_indexes)

def load_clean_data(i, weekdays, weekends, root, test, normalized=False, load_from=None):

    if(load_from):
        train_destination = root + load_from + '\\train\\'
        val_destination = root + load_from + '\\val\\'
        test_destination = root + load_from + '\\test\\'
    else: 
        if(weekdays and normalized):
            # print('LOADING WEEKDAYS CLEAN NORMALIZED DATA')
            train_destination = root + "clean_data\\weekdays_normalized\\train\\"
            val_destination = root + "clean_data\\weekdays_normalized\\val\\"
            test_destination = root + "clean_data\\weekdays_normalized\\test\\"

        if(weekdays and not normalized):
            # print('LOADING WEEKDAYS CLEAN DATA')
            train_destination = root + "clean_data\\weekdays\\train\\"
            val_destination = root + "clean_data\\weekdays\\val\\"
            test_destination = root + "clean_data\\weekdays\\test\\"
        
        if(weekends and normalized):
            # print('LOADING WEEKENDS CLEAN NORMALIZED DATA')
            train_destination = root + "clean_data\\weekends_normalized\\train\\"
            val_destination = root + "clean_data\\weekends_normalized\\val\\"
            test_destination = root + "clean_data\\weekends_normalized\\test\\"
                
        if(weekends and not normalized):
            # print('LOADING WEEKENDS CLEAN DATA')
            train_destination = root + "clean_data\\weekends\\train\\"
            val_destination = root + "clean_data\\weekends\\val\\"
            test_destination = root + "clean_data\\weekends\\test\\"

    files = os.listdir(train_destination)
    
    train = np.load(train_destination + files[i])
    val = np.load(val_destination + files[i])
    
    if test:
        test = np.load(test_destination + files[i])

    x_train = train['arr_0']
    y_train = train['arr_1']

    x_val = val['arr_0']
    y_val = val['arr_1']

    x_test = np.array([])
    y_test = np.array([])

    if test:
        x_test = test['arr_0']
        y_test = test['arr_1']
    
    return x_train, x_val, x_test, y_train, y_val, y_test

def MAPE(y_true, y_pred): 
    mask = np.logical_and((y_true != 0), ~np.isnan(y_true))
    mask = np.logical_and(mask, ~np.isinf(y_true))
    mask = np.logical_and(mask, ~np.isnan(y_true))    
    mask = np.logical_and(mask, ~np.isinf(y_pred))
    mask = np.logical_and(mask, ~np.isnan(y_pred))
    return (np.fabs(y_true[mask] - y_pred[mask]) / y_true[mask]).mean() * 100


def MASE(y_true, y_pred):
    n = y_true.shape[0]
    d = np.abs(np.diff(y_true)).sum() / (n-1)
    errors = np.abs(y_true - y_pred)
    return errors.mean() / d

def tf_diff_axis_0(a):
    return a[1:]-a[:-1]

def grid_search(root, learning_rate, layers, neurons, epochs, batch_size):
    clusters = {'low':cluster_0_indexes, 'medium':cluster_1_indexes, 'high':cluster_2_indexes}
    load_from = ['clean_data\\weekdays_24i_t24']
    input_dim = [24]
    predict_range = [24]
    output_dim = 1

    result = np.array([]).reshape((0, 12))

    for lf, idim, pr in zip(load_from, input_dim, predict_range ):
        # constants
        five_minutes_in_a_day = (24 * 60) // 5
        samples_per_day = five_minutes_in_a_day - idim - pr + 1
        weekday_days = 20
        weekend_days = 8
        train_batch = 5
        validate_batch = 5
        trials = 1

        # grid search per cluster
        for k, v in clusters.items():
            # load X, y
            samples_per_cluster = samples_per_day*len(v)
            x_train, x_val, x_test, y_train, y_val, y_test = get_input_cluster(input_dim=idim, output_dim=output_dim, samples_per_day=samples_per_day, cluster_c_indexes=v, 
                                                                               root=root, weekdays=True, weekends=False, load_clean=True, 
                                                                               normalized=False, test=False, load_from=lf)
            pd.DataFrame(y_train).to_csv('y_train.csv')
            pd.DataFrame(y_val).to_csv('y_val.csv')

            x_train = x_train.reshape((len(v), (x_train.shape[0] // len(v)), idim))
            x_val = x_val.reshape((len(v), (x_val.shape[0] // len(v)), idim))
            y_train = y_train.reshape((len(v), (y_train.shape[0] // len(v)) * output_dim))
            y_val = y_val.reshape((len(v), (y_val.shape[0] // len(v)) * output_dim))
            x_train = np.concatenate((x_train, x_val), axis=1)
            y_train = np.concatenate((y_train, y_val), axis=1)
            
            print('x_train.shape:', x_train.shape)
            print('y_train.shape:', y_train.shape)
            del x_val
            del y_val
            del x_test
            del y_test

            # measure base model
            for days in range(train_batch, weekday_days, train_batch):
                x = x_train[:, :days*samples_per_day, :]
                x = np.transpose(x, axes=(2, 0, 1)).reshape((x.shape[2], x.shape[0]*x.shape[1])).T

                y = y_train[:, :days*samples_per_day]
                y = y.flatten()

                x_val = x_train[:, days*samples_per_day:(days + validate_batch)*samples_per_day, :]
                x_val = np.transpose(x_val, axes=(2, 0, 1)).reshape((x_val.shape[2], x_val.shape[0]*x_val.shape[1])).T

                y_val = y_train[:, days*samples_per_day:(days + validate_batch)*samples_per_day]
                y_val = y_val.flatten()

                model = LinearRegression()
                base_start_time = datetime.now()
                model.fit(x, y)
                base_time_elapsed = datetime.now() - base_start_time

                y_hat = model.predict(x_val)
                y_hat = y_hat.astype(np.float64)
                y_hat = np.ceil(y_hat)
                val_loss = mean_squared_error(y_val, y_hat)
                mae = mean_absolute_error(y_val, y_hat)
                mape = MAPE(y_val, y_hat)

            data = np.array(['base', k, 0, 0, 0, idim, lf[-3:], 0, base_time_elapsed, val_loss, mae, mape])
            print(data)
            data = data.reshape((1, data.shape[0]))
            result = np.concatenate((result, data), axis=0)

            # measure dnn
            for lr in learning_rate:
                for l in layers:
                    for n in neurons:
                        for t in range(trials):
                            # cross_val time series
                            val_losses = []
                            val_maes = []
                            val_mapes = []
                        
                            for days in range(train_batch, weekday_days, train_batch):
                                x = x_train[:, :days*samples_per_day, :]
                                x = np.transpose(x, axes=(2, 0, 1)).reshape((x.shape[2], x.shape[0]*x.shape[1])).T

                                y = y_train[:, :days*samples_per_day]
                                y = y.flatten()

                                x_val = x_train[:, days*samples_per_day:(days + validate_batch)*samples_per_day, :]
                                x_val = np.transpose(x_val, axes=(2, 0, 1)).reshape((x_val.shape[2], x_val.shape[0]*x_val.shape[1])).T

                                y_val = y_train[:, days*samples_per_day:(days + validate_batch)*samples_per_day]
                                y_val = y_val.flatten()
                                
                                # measure dnn
                                models = [create_grid_search_model(idim, output_dim, lr, l, n)]

                                start_time = datetime.now() 
                                history = models[0].fit(x, y, batch_size=batch_size, validation_data=(x_val, y_val), verbose=0, epochs=epochs)
                                time_elapsed = datetime.now() - start_time
                                val_loss = history.history['val_loss'][-1]

                                y_hat = models[0].predict(x_val, batch_size=batch_size)
                                y_hat = y_hat.flatten()
                                y_hat = y_hat.astype(np.float64)
                                y_hat = np.ceil(y_hat)
                                mae = mean_absolute_error(y_val, y_hat)
                                mape = MAPE(y_val, y_hat)

                                val_losses.append(val_loss)
                                val_maes.append(mae)
                                val_mapes.append(mape)
                                
                            avg_val_loss = np.mean(val_losses)
                            avg_mae = np.mean(val_maes)
                            avg_mape = np.mean(val_mapes)
            
                            data = np.array(['dnn', k, lr, l, n, idim, lf[-3:], epochs, time_elapsed, avg_val_loss, avg_mae, avg_mape])
                            print(data)
                            data = data.reshape((1, data.shape[0]))
                            result = np.concatenate((result, data), axis=0)

    return result

class my_model:

    def __init__(self, num_time_series, input_dim, predict_range, samples_per_day):
      self.num_time_series = num_time_series
      self.values = np.array([]).reshape((0, self.num_time_series))
      self.input_dim = input_dim
      self.predict_range = predict_range
      self.samples_per_day = samples_per_day
      self.five_minutes_in_a_day = 288
      self.weekday_days = 20
      self.ts_id_index_map = {}
   
    def fit(self, x_train, y_train, days, cluster_c_indexes):
        start_total_minutes = (self.input_dim + self.predict_range - 1) * 5
        start_hour = start_total_minutes // 60
        start_minute = start_total_minutes % 60
        date_range = pd.date_range('1/1/2016', periods=self.five_minutes_in_a_day * (days + (2 * (days // 5))), freq='5T') # (days // 5) to add weekend days, then remove them
        df = pd.DataFrame(np.zeros((len(date_range), self.num_time_series)) , index=date_range)
        df = df[df.index.dayofweek < 5]
        df = df[(df.index.time >= dt.time(start_hour, start_minute))]
        df[:] = y_train

        for d in range(5):
            for h in range(start_hour, 24):
                for m in range(0 if h is not start_hour else start_minute, 60, 5):
                    hourminaverage = df[(df.index.dayofweek == d) & (df.index.hour == h) & (df.index.minute == m)].mean()
                    self.values = np.vstack((self.values, hourminaverage)) # [avg, num_timeseries]

        self.values = self.values.reshape((5, self.values.shape[0] // 5, self.values.shape[1])) # [5_days, avg, num_time_series]

        count = 0
        for i in cluster_c_indexes:
            self.ts_id_index_map[i] = count
            count = count + 1

    def predict(self, x_test, val=False):
        y_hat = self.values
        print('y_hat:', y_hat.shape) # [5_days, avg, num_time_series]

        if(val):
            y_hat = y_hat[[4, 0, 1, 2, 3]] # Friday, Monday. Tuesday, Wednesday, Thursday
            if(x_test.shape[0] < 5*self.samples_per_day): # validation_batch*samples_per_day
                print('val last day')
                y_hat = np.expand_dims(y_hat[0], axis=0) # only Friday
        else:
            y_hat = y_hat[0:4] # predict last four days of the 20 days, which are Monday, Tuesday, Wednesday and Thursday
        y_hat = y_hat.reshape((y_hat.shape[0]*y_hat.shape[1], y_hat.shape[2])) # [dayofweek*avg, num_timeseries]
        print('y_hat:', y_hat.shape)
        return y_hat        
     
        

def report_on_grid_search(root, learning_rate, layers, neurons, epochs):
    df = pd.read_csv('grid_search_result.csv', index_col=0, header=0)
    clusters = {'low':cluster_0_indexes, 'medium':cluster_1_indexes, 'high':cluster_2_indexes, 'all':cluster_c_indexes}
    result = []
    for k, v in clusters.items():
        for lr in learning_rate:
            for l in layers:
                for n in neurons:
                    desc_time_elapsed = df[(df.k == k) & (df.lr == lr) & (df.l == l) & (df.n == n)].time_elapsed.describe()
                    desc_avg_val_loss = df[(df.k == k) & (df.lr == lr) & (df.l == l) & (df.n == n)].avg_val_loss.describe()
                    result.append([k, lr, l, n, input_dim, output_dim, epochs, desc_time_elapsed, desc_avg_val_loss])
    return result

def create_grid_search_model(input_dim, output_dim, lr, l, n):  
    model = Sequential()
    model.add(Dense(n, input_dim=input_dim, activation='relu'))
    for i in range(l):
        model.add(Dense(n, activation='relu'))
    model.add(Dense(output_dim, activation='linear'))
    rmsprop = RMSprop(lr=lr)
    model.compile(loss='mse', optimizer=rmsprop, metrics=['mape', 'mae'])
    return model

def run():
    batch_size = 16384
    learning_rate = 0.001
    k = 1
    normalized = False
    root = 'dataset\\anaheim\\'
    weekdays = True
    weekends = False   

    learning_rate = [10**-3]
    layers = [1, 3, 4, 6, 8]
    neurons = [16, 32, 64, 128]
    epochs = 500
    result = grid_search(root, learning_rate, layers, neurons, epochs, batch_size)
    result = np.array(result)
    result = pd.DataFrame(result)
    result.to_csv('grid_search_result_low.csv')

    # result = report_on_grid_search(root, learning_rate, layers, neurons, epochs)
    # result = np.array(result)
    # result = pd.DataFrame(result)
    # result.to_csv('grid_search_result.csv')

run()