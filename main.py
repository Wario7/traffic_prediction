# This file implements the experiments in the paper

### Parameters to change in order to reproduce results ###
# test_cluster_indexes choose between 'cluster_0_indexes', 'cluster_1_indexes', 'cluster_2_indexes'
# parameters in run() function
##########################################################

from datetime import datetime 
from sacred import Experiment
from sacred.observers import MongoObserver

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
import deprecated
import pickle

from keras import optimizers
from keras import backend as K
from keras.callbacks import Callback
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, LSTM, Dropout, Reshape, BatchNormalization, Flatten, Conv1D, Conv2D, Conv3D, MaxPooling2D, GRU, Input, TimeDistributed, Concatenate
from keras.layers.advanced_activations import PReLU
from keras.optimizers import RMSprop, SGD, Adam
from keras.regularizers import l1, l2
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

input_dim = 24
output_dim = 1
predict_range = 24
cities = ['anaheim', 'oakland']


def get_input_cluster(input_dim, output_dim, samples_per_day, cluster_c_indexes, root, weekdays=True, weekends=False, file_order=None, load_clean=True, normalized=False, test=True, load_from=None):
    x_train_total = np.array([]).reshape(0, input_dim)
    x_val_total = np.array([]).reshape(0, input_dim)
    x_test_total = np.array([]).reshape(0, input_dim)
    
    y_train_total = np.array([]).reshape(0, output_dim)
    y_val_total = np.array([]).reshape(0, output_dim)
    y_test_total = np.array([]).reshape(0, output_dim)
    
    X_total = np.array([]).reshape(0, input_dim)
    Y_total = np.array([]).reshape(0, output_dim)
    list_to_remove = []

    train_size = 0.8
    weekday_days = 20
    criteria = train_size*weekday_days*samples_per_day

    for i in cluster_c_indexes:
        
        if(load_clean):
            x_train, x_val, x_test, y_train, y_val, y_test = load_clean_data(i, weekdays, weekends, root=root, normalized=normalized, test=test, load_from=load_from)
            # print('x_train:', i, x_train[0])
            if(np.concatenate((y_train, y_val)).shape[0] != criteria): # some time series contain less samples
                print(np.concatenate((y_train, y_val)).shape[0], criteria)
                continue
            if(not np.any(y_train) or not np.any(y_val)):
                print('zeroes in', i)
                list_to_remove.append(i)
                continue
            # print(x_train.shape, idx)

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
    
    for i in list_to_remove:
        print('removing', i)
        cluster_c_indexes.remove(i)

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

cluster_0_indexes, cluster_1_indexes, cluster_2_indexes = get_clusters(random=False, root='dataset\\anaheim\\') #TODO:change accordingly
test_cluster_indexes = cluster_2_indexes

cluster_c_indexes = list(cluster_0_indexes) + \
                    list(cluster_1_indexes) + \
                    list(cluster_2_indexes)


def plot_history(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def plot_loss_val_loss(loss, val_loss):
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

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
    # print('root:', root)
    # print('file_num:', i)
    
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

# exp 2
@ex.capture
def method_1(batch_size, epochs, learning_rate, weekdays, weekends, normalized, root, _run):
    batch_size = 65536
    five_minutes_in_a_day = (24 * 60) // 5
    samples_per_day = five_minutes_in_a_day - input_dim - predict_range + 1
    weekday_days = 20
    weekend_days = 8
    train_size = 0.8
    load_from = 'clean_data\\weekdays_24i_t24'

    result = np.array([]).reshape(0, 6+3)

    for city in cities:
        print('testing:', city)
        root = 'dataset\\' + city + '\\'
        cluster_0_indexes, cluster_1_indexes, cluster_2_indexes = get_clusters(random=False, root=root)
        test_cluster_indexes = cluster_2_indexes

        cluster_c_indexes = list(cluster_0_indexes) + \
                            list(cluster_1_indexes) + \
                            list(cluster_2_indexes)

        clusters = {'low':cluster_0_indexes, 'medium':cluster_1_indexes, 'high':cluster_2_indexes, 'all':cluster_c_indexes}
        # clusters = {'low':cluster_0_indexes}

        for k, v in clusters.items():
            time_total_dnn = np.array([])
            time_total_base = np.array([])
            K.clear_session()
            losses = np.array([]).reshape(0, 2) # length of models
            mapes = np.array([]).reshape(0, 2) # length of models
            
            for i in v:
                print(i)
                x_train, x_val, x_test, y_train, y_val, y_test = get_input_cluster(input_dim=input_dim, output_dim=output_dim, samples_per_day=samples_per_day, 
                                                                                   cluster_c_indexes=[i], weekdays=weekdays, weekends=weekends, 
                                                                                   load_clean=True, root=root, normalized=normalized, load_from=load_from)
                x_train = np.concatenate((x_train, x_val), axis=0)
                y_train = np.concatenate((y_train, y_val), axis=0)
                del x_val
                del y_val
            
                if len(x_train) is 0 or len(x_test) is 0:
                    print('skipped:', i)
                    continue

                models = [eval('create_best_conf_model_' + k)(), LinearRegression()]

                # measure base temporal median
                start_time = datetime.now() 
                models[1].fit(x_train, y_train, days=weekday_days*train_size)
                time_elapsed_base = datetime.now() - start_time
                time_total_base = np.append(time_total_base, time_elapsed_base)

                # measure DNN
                start_time = datetime.now() 
                history = models[0].fit(x_train, y_train, batch_size=batch_size, verbose=0, epochs=epochs)
                time_elapsed_dnn = datetime.now() - start_time
                time_total_dnn = np.append(time_total_dnn, time_elapsed_dnn)

                metrics = test_model_in_method(models, x_test=x_test, y_test=y_test) # returns [loss, mape] per model

                data = np.append([city, k, i, 'dnn', load_from[-3:], time_elapsed_dnn], metrics[0].flatten())
                data = data.reshape((1, data.shape[0]))
                result = np.concatenate((result, data), axis=0)
                
                data = np.append([city, k, i, 'base', load_from[-3:], time_elapsed_base], metrics[1].flatten())
                data = data.reshape((1, data.shape[0]))
                result = np.concatenate((result, data), axis=0)

    result = pd.DataFrame(result)
    result.to_csv('dataset\\anaheim\\' + 'reports\\method_1_cities.csv')

    return result

# returns [loss, mae, mape] per model
def test_model_in_method(models, x_test, y_test, val=False, batch_size=256):
    five_minutes_in_a_day = (24 * 60) // 5
    samples_per_day = five_minutes_in_a_day - input_dim - predict_range + 1
    metrics = []
    for m in models:
        # y_hat = y_hat.flatten()
        if (type(m) == Sequential) or (type(m) == Model):
            y_hat = m.predict(x_test, batch_size=batch_size)
            y_hat = y_hat.astype(np.float64)
            y_hat = np.ceil(y_hat)
            loss = [m.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)[0]]
        elif(type(m) == LinearRegression):
            print('x_test.shape', x_test.shape)
            print('x_test_0:', x_test[0, 0, 0])
            print('x_test:', x_test[0, 0, 1])

            print('y_test_0:', y_test.T.flatten()[0:3])
            print('y_test:', y_test.T.flatten()[16*samples_per_day:16*samples_per_day+3])

            # to turn back
            # curren: [samples, time_dim, num_time_series]
            # [time_dim, num_time_series, samples]
            # [time_dim, num_time_series*samples]
            # [num_time_series*samples, time_dim]
            clf_x = np.transpose(x_test, axes=(1, 2, 0)).reshape((x_test.shape[1], x_test.shape[0]*x_test.shape[2])).T
            clf_y = y_test.T.flatten() # [num_time_series*samples]

            print('before clf x_test_0:', clf_x[0])
            print('before clf x_test:', clf_x[16*samples_per_day])

            print('before y_test_0:', clf_y[0:3])
            print('before y_test:', clf_y[16*samples_per_day:16*samples_per_day+3])

            y_hat = m.predict(clf_x)
            y_hat = y_hat.astype(np.float64)   
            loss = [mean_squared_error(clf_y, y_hat)]
            y_hat = y_hat.reshape((x_test.shape[2], x_test.shape[0]))
            y_hat = y_hat.T

        else:
            y_hat = m.predict(x_test, val)
            y_hat = y_hat.astype(np.float64)   
            loss = [mean_squared_error(y_test, y_hat)]
        # loss.extend(mean_squared_error(y_test, y_hat, multioutput='raw_values'))
        
        mae = [mean_absolute_error(y_test, y_hat)]
        # mae.extend(mean_absolute_error(y_test, y_hat, multioutput='raw_values'))

        mape = [MAPE(y_test, y_hat)]
        metrics.append([loss, mae, mape])
    
    metrics = np.array(metrics)
    return metrics

def method_2_big_input():
      # constants
    batch_size = 256
    five_minutes_in_a_day = (24 * 60) // 5
    samples_per_day = five_minutes_in_a_day - input_dim - predict_range + 1
    weekday_days = 20
    weekend_days = 8
    train_size = 0.8
    test_size = 0.2
    load_from = 'clean_data\\weekdays_24i_t24'
    epochs = 500
    result = np.array([]).reshape(0, 4+3)
    normalized = False

    for city in cities:
        root = 'dataset\\' + city + '\\'
        cluster_0_indexes, cluster_1_indexes, cluster_2_indexes = get_clusters(random=False, root=root)
        test_cluster_indexes = cluster_2_indexes

        cluster_c_indexes = list(cluster_0_indexes) + \
                            list(cluster_1_indexes) + \
                            list(cluster_2_indexes)

        time_taken_by_models = {}

        x_train, x_val, x_test, y_train, y_val, y_test = get_input_cluster(input_dim=input_dim, output_dim=output_dim, samples_per_day=samples_per_day, 
                                                                               cluster_c_indexes=cluster_c_indexes, root=root, weekdays=True, 
                                                                               weekends=False, load_clean=True, normalized=normalized, test=True, load_from=load_from)

        models = [create_best_conf_model_all_big_input(len(cluster_c_indexes), input_dim), LinearRegression()]

        x_train = x_train.reshape((len(cluster_c_indexes), (x_train.shape[0] // len(cluster_c_indexes)), input_dim))
        x_val = x_val.reshape((len(cluster_c_indexes), (x_val.shape[0] // len(cluster_c_indexes)), input_dim))
        y_train = y_train.reshape((len(cluster_c_indexes), (y_train.shape[0] // len(cluster_c_indexes)) * output_dim))
        y_val = y_val.reshape((len(cluster_c_indexes), (y_val.shape[0] // len(cluster_c_indexes)) * output_dim))
        x_test = x_test.reshape((len(cluster_c_indexes), (x_test.shape[0] // len(cluster_c_indexes)),  input_dim))
        y_test = y_test.reshape((len(cluster_c_indexes), (y_test.shape[0] // len(cluster_c_indexes)) * output_dim))
        x_train = np.concatenate((x_train, x_val), axis=1)
        y_train = np.concatenate((y_train, y_val), axis=1)

        clf_x = np.transpose(x_train, axes=(2, 0, 1)).reshape((x_train.shape[2], x_train.shape[0]*x_train.shape[1])).T
        clf_y = y_train.flatten()
        clf_x_test = np.transpose(x_test, axes=(2, 0, 1)).reshape((x_test.shape[2], x_test.shape[0]*x_test.shape[1])).T
        clf_y_test = y_test.flatten()

        x_train = np.transpose(x_train, axes=(1, 2, 0)) # [samples, input_dim, num_time_series]
        y_train = y_train.T # [samples, num_time_series]
        x_test = np.transpose(x_test, axes=(1, 2, 0))
        y_test = y_test.T

        # measure DNN
        start_time = datetime.now() 
        history = models[0].fit(x_train, y_train, batch_size=batch_size, verbose=0, epochs=epochs, validation_data=(x_test, y_test))
        with open('dataset\\anaheim\\reports\\history\\all_method_history_' + city + '.pkl', 'wb') as f:
                pickle.dump(history.history, f)
        models[0].save(root + 'models\\' + 'model_' + city + '_big_input.hdf5')
        time_elapsed = datetime.now() - start_time
        time_taken_by_models['dnn'] = time_elapsed

        start_time = datetime.now()
        models[1].fit(clf_x, clf_y)
        time_elapsed = datetime.now() - start_time
        time_taken_by_models['base'] = time_elapsed

        y_hat = models[0].predict(x_test, batch_size=batch_size)
        y_hat = y_hat.astype(np.float64)
        y_hat = np.ceil(y_hat)
        loss = mean_squared_error(y_test, y_hat)
        mae = mean_absolute_error(y_test, y_hat)
        mape = MAPE(y_test, y_hat)
        data = np.array([city, 'all', 'dnn', time_taken_by_models['dnn'], loss, mae, mape])
        print(data)
        data = data.reshape((1, data.shape[0]))
        result = np.concatenate((result, data), axis=0)

        loss = mean_squared_error(y_test, y_hat)
        mae = mean_absolute_error(y_test, y_hat)
        mape = MAPE(y_test, y_hat)
        data = np.array([city, 'low', 'dnn', time_taken_by_models['dnn'], loss, mae, mape])
        print(data)
        data = data.reshape((1, data.shape[0]))
        result = np.concatenate((result, data), axis=0)

        y_hat = models[1].predict(clf_x_test)
        y_hat = y_hat.astype(np.float64)
        y_hat = np.ceil(y_hat)
        loss = mean_squared_error(clf_y_test, y_hat)
        mae = mean_absolute_error(clf_y_test, y_hat)
        mape = MAPE(clf_y_test, y_hat)
        data = np.array([city, 'all', 'base', time_taken_by_models['base'], loss, mae, mape])
        print(data)
        data = data.reshape((1, data.shape[0]))
        result = np.concatenate((result, data), axis=0)

        y_hat = models[1].predict(clf_x_test)
        y_hat = y_hat.astype(np.float64)
        y_hat = np.ceil(y_hat)
        loss = mean_squared_error(clf_y_test, y_hat)
        mae = mean_absolute_error(clf_y_test, y_hat)
        mape = MAPE(clf_y_test, y_hat)
        data = np.array([city, 'low', 'base', time_taken_by_models['base'], loss, mae, mape])
        print(data)
        data = data.reshape((1, data.shape[0]))
        result = np.concatenate((result, data), axis=0)
        
        del x_val
        del y_val
        del x_test
        del y_test

    result = pd.DataFrame(result)
    result.to_csv('dataset\\anaheim\\' + 'reports\\method_2_cities_big_input.csv')

    return result


def latent_structure_method_new():
    # constants
    # batch_size = 256
    batch_size = 65536
    learning_rate = 10**-3
    five_minutes_in_a_day = (24 * 60) // 5
    samples_per_day = five_minutes_in_a_day - input_dim - predict_range + 1
    weekday_days = 20
    weekend_days = 8
    train_size = 0.8
    epochs = 500
    load_from = 'clean_data\\weekdays_24i_t24'
    result = np.array([]).reshape(0, 4+3)
    cluster_size = 3
    normalized = False

    for city in cities:
        root = 'dataset\\' + city + '\\'
        cluster_0_indexes, cluster_1_indexes, cluster_2_indexes = get_clusters(random=False, root=root)
        test_cluster_indexes = cluster_2_indexes

        cluster_c_indexes = list(cluster_0_indexes) + \
                            list(cluster_1_indexes) + \
                            list(cluster_2_indexes)

        N = len(cluster_c_indexes)
        subset = N // (cluster_size ** 2)

        time_taken_by_models = {}
        # clusters_original = {'low':cluster_0_indexes, 'medium':cluster_1_indexes, 'high':cluster_2_indexes, 'all':cluster_c_indexes}
        clusters_original = {'low':list(cluster_0_indexes), 'medium':list(cluster_1_indexes), 'high':list(cluster_2_indexes)}

        # load X, y
        x_train, x_val, x_test, y_train, y_val, y_test = get_input_cluster(input_dim=input_dim, output_dim=output_dim, samples_per_day=samples_per_day,
                                                                           cluster_c_indexes=cluster_c_indexes, 
                                                                           root=root, weekdays=True, weekends=False, 
                                                                           load_clean=True, normalized=normalized, test=True, load_from=load_from)

        models = [create_best_conf_model_all()]  
        
        x_train = x_train.reshape((len(cluster_c_indexes), (x_train.shape[0] // len(cluster_c_indexes)), input_dim))
        x_val = x_val.reshape((len(cluster_c_indexes), (x_val.shape[0] // len(cluster_c_indexes)), input_dim))
        y_train = y_train.reshape((len(cluster_c_indexes), (y_train.shape[0] // len(cluster_c_indexes)) * output_dim))
        y_val = y_val.reshape((len(cluster_c_indexes), (y_val.shape[0] // len(cluster_c_indexes)) * output_dim))
        x_test = x_test.reshape((len(cluster_c_indexes), (x_test.shape[0] // len(cluster_c_indexes)),  input_dim))
        y_test = y_test.reshape((len(cluster_c_indexes), (y_test.shape[0] // len(cluster_c_indexes)) * output_dim))
        x_train = np.concatenate((x_train, x_val), axis=1)
        y_train = np.concatenate((y_train, y_val), axis=1)

        clf_x = np.transpose(x_train, axes=(2, 0, 1)).reshape((x_train.shape[2], x_train.shape[0]*x_train.shape[1])).T
        clf_y = y_train.flatten()
        clf_x_test = np.transpose(x_test, axes=(2, 0, 1)).reshape((x_test.shape[2], x_test.shape[0]*x_test.shape[1])).T
        clf_y_test = y_test.flatten()

        # measure DNN
        start_time = datetime.now() 
        history = models[0].fit(clf_x, clf_y, batch_size=batch_size, verbose=0, epochs=epochs, validation_data=(clf_x_test, clf_y_test))
        with open('dataset\\anaheim\\reports\\latent_structure_method_history_' + city + '.pkl', 'wb') as f:
            pickle.dump(history.history, f)
        time_elapsed = datetime.now() - start_time
        time_taken_by_models['dnn'] = time_elapsed

        del x_val
        del y_val
        del x_test
        del y_test

        for k, v in clusters_original.items():
            # load X, y
            x_train, x_val, x_test, y_train, y_val, y_test = get_input_cluster(input_dim=input_dim, output_dim=output_dim, samples_per_day=samples_per_day, 
                                                                               cluster_c_indexes=v, root=root, weekdays=True, 
                                                                               weekends=False, load_clean=True, normalized=False, test=True, load_from=load_from)

            x_train = x_train.reshape((len(v), (x_train.shape[0] // len(v)), input_dim))
            x_val = x_val.reshape((len(v), (x_val.shape[0] // len(v)), input_dim))
            y_train = y_train.reshape((len(v), (y_train.shape[0] // len(v)) * output_dim))
            y_val = y_val.reshape((len(v), (y_val.shape[0] // len(v)) * output_dim))
            x_test = x_test.reshape((len(v), (x_test.shape[0] // len(v)),  input_dim))
            y_test = y_test.reshape((len(v), (y_test.shape[0] // len(v)) * output_dim))
            x_train = np.concatenate((x_train, x_val), axis=1)
            y_train = np.concatenate((y_train, y_val), axis=1)
            del x_val
            del y_val

            clf_x = np.transpose(x_train, axes=(2, 0, 1)).reshape((x_train.shape[2], x_train.shape[0]*x_train.shape[1])).T
            clf_y = y_train.flatten()
            clf_x_test = np.transpose(x_test, axes=(2, 0, 1)).reshape((x_test.shape[2], x_test.shape[0]*x_test.shape[1])).T
            clf_y_test = y_test.flatten()

            print(clf_x.shape)
            print(clf_y.shape)
            print(clf_x_test.shape)
            print(clf_y_test.shape)

            for i in range(len(models[0].layers) - 2):
                models[0].layers[i].trainable = False

            for i in range(len(models[0].layers)):
                print('layer', i, 'trainable:', models[0].layers[i].trainable)

            rmsprop = RMSprop(lr=learning_rate)
            models[0].compile(loss='mse', optimizer=rmsprop, metrics=['mape', 'mae'])

            # measure DNN
            start_time = datetime.now() 
            history = models[0].fit(clf_x, clf_y, batch_size=batch_size, verbose=0, epochs=epochs)
            time_elapsed = datetime.now() - start_time
            time_taken_by_models['dnn'] = time_taken_by_models['dnn'] + time_elapsed
            models[0].save(root + 'models\\' + 'model_latent_' + k + '_' + city + '.hdf5')

            y_hat = models[0].predict(clf_x_test, batch_size=batch_size)
            y_hat = y_hat.flatten()
            y_hat = y_hat.astype(np.float64)
            y_hat = np.ceil(y_hat)
            loss = mean_squared_error(clf_y_test, y_hat)
            mae = mean_absolute_error(clf_y_test, y_hat)
            mape = MAPE(clf_y_test, y_hat)
            data = np.array([city, k, 'dnn', time_taken_by_models['dnn'], loss, mae, mape])
            print(data)
            data = data.reshape((1, data.shape[0]))
            result = np.concatenate((result, data), axis=0)

    result = pd.DataFrame(result)
    result.to_csv('dataset\\anaheim\\' + 'reports\\latent_structure_method.csv')

    return result

def latent_structure_across_cities_new():
    # constants
    # batch_size = 256
    batch_size = 65536
    learning_rate = 10**-3
    five_minutes_in_a_day = (24 * 60) // 5
    samples_per_day = five_minutes_in_a_day - input_dim - predict_range + 1
    weekday_days = 20
    weekend_days = 8
    train_size = 0.8
    epochs = 500
    load_from = 'clean_data\\weekdays_24i_t24'
    cluster_size = 3
    normalized = False
    training = [False, True]

    for t in training:
        result = np.array([]).reshape(0, 5+t+3)
        cluster_size = 3

        for city_i in cities:

            root_source = 'dataset\\' + city_i + '\\'
            models = {'all':load_model(root_source + 'models\\' + 'model_' + city_i + '.hdf5')}

            cities_without_i = cities[:]
            cities_without_i.remove(city_i)

            for city_j in cities_without_i:
            # for city_j in cities:

                root_dest = 'dataset\\' + city_j + '\\'
                cluster_0_indexes, cluster_1_indexes, cluster_2_indexes = get_clusters(random=False, root=root_dest)
                test_cluster_indexes = cluster_2_indexes

                cluster_c_indexes = list(cluster_0_indexes) + \
                                    list(cluster_1_indexes) + \
                                    list(cluster_2_indexes)

                time_taken_by_models = {}
                # clusters = {'low':cluster_0_indexes, 'medium':cluster_1_indexes, 'high':cluster_2_indexes, 'all':cluster_c_indexes}
                clusters = {'low':list(cluster_0_indexes), 'medium':list(cluster_1_indexes), 'high':list(cluster_2_indexes)}

                for k, v in clusters.items():
                    # load X, y
                    x_train, x_val, x_test, y_train, y_val, y_test = get_input_cluster(input_dim=input_dim, output_dim=output_dim, samples_per_day=samples_per_day, 
                                                                                       cluster_c_indexes=v, root=root_dest, weekdays=True, 
                                                                                       weekends=False, load_clean=True, normalized=False, test=True, load_from=load_from)

                    x_train = x_train.reshape((len(v), (x_train.shape[0] // len(v)), input_dim))
                    x_val = x_val.reshape((len(v), (x_val.shape[0] // len(v)), input_dim))
                    y_train = y_train.reshape((len(v), (y_train.shape[0] // len(v)) * output_dim))
                    y_val = y_val.reshape((len(v), (y_val.shape[0] // len(v)) * output_dim))
                    x_test = x_test.reshape((len(v), (x_test.shape[0] // len(v)),  input_dim))
                    y_test = y_test.reshape((len(v), (y_test.shape[0] // len(v)) * output_dim))
                    x_train = np.concatenate((x_train, x_val), axis=1)
                    y_train = np.concatenate((y_train, y_val), axis=1)
                    del x_val
                    del y_val

                    clf_x = np.transpose(x_train, axes=(2, 0, 1)).reshape((x_train.shape[2], x_train.shape[0]*x_train.shape[1])).T
                    clf_y = y_train.flatten()
                    clf_x_test = np.transpose(x_test, axes=(2, 0, 1)).reshape((x_test.shape[2], x_test.shape[0]*x_test.shape[1])).T
                    clf_y_test = y_test.flatten()

                    print(clf_x.shape)
                    print(clf_y.shape)
                    print(clf_x_test.shape)
                    print(clf_y_test.shape)


                    # measure DNN
                    if t:
                        for m_key, m in models.items():
                            for i in range(len(m.layers) - 2):
                                m.layers[i].trainable = False
                            print(city_i, city_j, m_key, k)

                            rmsprop = RMSprop(lr=learning_rate)
                            m.compile(loss='mse', optimizer=rmsprop, metrics=['mape', 'mae'])

                            # measure DNN
                            start_time = datetime.now() 
                            history = m.fit(clf_x, clf_y, batch_size=batch_size, verbose=0, epochs=epochs, validation_data=(clf_x_test, clf_y_test))
                            time_elapsed = datetime.now() - start_time
                            time_taken_by_models['source_' + m_key + '_destination_' + k] = time_elapsed
                            with open('dataset\\anaheim\\reports\\plots\\losses\\latent_structure_across_city_history_' + city_i + '_' + city_j + '_' + k + '.pkl', 'wb') as f:
                                pickle.dump(history.history, f)

                    for m_key, m in models.items():
                        y_hat = m.predict(clf_x_test, batch_size=batch_size)
                        y_hat = y_hat.flatten()
                        y_hat = y_hat.astype(np.float64)
                        y_hat = np.ceil(y_hat)
                        loss = mean_squared_error(clf_y_test, y_hat)
                        mae = mean_absolute_error(clf_y_test, y_hat)
                        mape = MAPE(clf_y_test, y_hat)
                        if t:
                            data = np.array([city_i, city_j, m_key, k, load_from[-3:], time_taken_by_models['source_' + m_key + '_destination_' + k], loss, mae, mape])
                        else:
                            data = np.array([city_i, city_j, m_key, k, load_from[-3:], loss, mae, mape])

                        print(data)
                        data = data.reshape((1, data.shape[0]))
                        result = np.concatenate((result, data), axis=0)
    
        result = pd.DataFrame(result)
        if t:
            result.to_csv('dataset\\anaheim\\' + 'reports\\latent_structure_across_cities_training_new.csv')
        else:
            result.to_csv('dataset\\anaheim\\' + 'reports\\latent_structure_across_cities_new.csv')

    return result

# exp 3, 6
def method_5_big_input():
    # constants
    batch_size = 256
    five_minutes_in_a_day = (24 * 60) // 5
    samples_per_day = five_minutes_in_a_day - input_dim - predict_range + 1
    weekday_days = 20
    weekend_days = 8
    train_size = 0.8
    test_size = 0.2
    load_from = 'clean_data\\weekdays_24i_t24'
    epochs = 500
    # load_from = 'clean_data\\weekdays_12i_t1'
    # result = np.array([]).reshape(0, 5+(3*(output_dim+1))) # 3 metrics*(avg metric for all feature, avg metric for individual feature)
    result = np.array([]).reshape(0, 4+3)

    for city in cities:
        root = 'dataset\\' + city + '\\'
        cluster_0_indexes, cluster_1_indexes, cluster_2_indexes = get_clusters(random=False, root=root)
        test_cluster_indexes = cluster_2_indexes

        cluster_c_indexes = list(cluster_0_indexes) + \
                            list(cluster_1_indexes) + \
                            list(cluster_2_indexes)

        clusters = {'low':list(cluster_0_indexes), 'medium':list(cluster_1_indexes), 'high':list(cluster_2_indexes)}

        y_hat_dnn_all = np.array([]).reshape(samples_per_day*int(test_size*weekday_days), 0)
        y_hat_base_all = np.array([])
        y_test_dnn_all = np.array([]).reshape(samples_per_day*int(test_size*weekday_days), 0)
        y_test_base_all = np.array([])

        for k, v in clusters.items():
            # load X, y
            x_train, x_val, x_test, y_train, y_val, y_test = get_input_cluster(input_dim=input_dim, output_dim=output_dim, samples_per_day=samples_per_day, 
                                                                               cluster_c_indexes=v, root=root, weekdays=True, 
                                                                               weekends=False, load_clean=True, normalized=False, test=True, load_from=load_from)

            x_train = x_train.reshape((len(v), (x_train.shape[0] // len(v)), input_dim))
            x_val = x_val.reshape((len(v), (x_val.shape[0] // len(v)), input_dim))
            y_train = y_train.reshape((len(v), (y_train.shape[0] // len(v)) * output_dim))
            y_val = y_val.reshape((len(v), (y_val.shape[0] // len(v)) * output_dim))
            x_test = x_test.reshape((len(v), (x_test.shape[0] // len(v)),  input_dim))
            y_test = y_test.reshape((len(v), (y_test.shape[0] // len(v)) * output_dim))
            x_train = np.concatenate((x_train, x_val), axis=1)
            y_train = np.concatenate((y_train, y_val), axis=1)
            del x_val
            del y_val

            models = [eval('create_best_conf_model_big_input_' + k)(num_time_series=len(v), input_dim=input_dim), LinearRegression()]
            time_taken_by_models = {}

            clf_x = np.transpose(x_train, axes=(2, 0, 1)).reshape((x_train.shape[2], x_train.shape[0]*x_train.shape[1])).T
            clf_y = y_train.flatten()
            clf_x_test = np.transpose(x_test, axes=(2, 0, 1)).reshape((x_test.shape[2], x_test.shape[0]*x_test.shape[1])).T
            clf_y_test = y_test.flatten()

            print(clf_x.shape)
            print(clf_y.shape)
            print(clf_x_test.shape)
            print(clf_y_test.shape)

            x_train = np.transpose(x_train, axes=(1, 2, 0)) # [samples, input_dim, num_time_series]
            y_train = y_train.T # [samples, num_time_series]
            x_test = np.transpose(x_test, axes=(1, 2, 0))
            y_test = y_test.T

            # measure DNN
            start_time = datetime.now() 
            history = models[0].fit(x_train, y_train, batch_size=batch_size, verbose=0, epochs=epochs, validation_data=(x_test, y_test))
            with open('dataset\\anaheim\\reports\\history\\cluster_method_history_' + city + '_' + k + '.pkl', 'wb') as f:
                pickle.dump(history.history, f)
            time_elapsed = datetime.now() - start_time
            time_taken_by_models['dnn'] = time_elapsed

            start_time = datetime.now()
            models[1].fit(clf_x, clf_y)
            time_elapsed = datetime.now() - start_time
            time_taken_by_models['base'] = time_elapsed

            y_hat = models[0].predict(x_test, batch_size=batch_size)
            y_hat = y_hat.astype(np.float64)
            y_hat = np.ceil(y_hat)
            loss = mean_squared_error(y_test, y_hat)
            mae = mean_absolute_error(y_test, y_hat)
            mape = MAPE(y_test, y_hat)
            data = np.array([city, k, 'dnn', time_taken_by_models['dnn'], loss, mae, mape])
            print(data)
            data = data.reshape((1, data.shape[0]))
            result = np.concatenate((result, data), axis=0)
            
            print(y_test_dnn_all.shape)
            print(y_test.shape)
            y_test_dnn_all = np.concatenate((y_test_dnn_all, y_test), axis=1)
            y_hat_dnn_all = np.concatenate((y_hat_dnn_all, y_hat), axis=1)

            y_hat = models[1].predict(clf_x_test)
            y_hat = y_hat.astype(np.float64)
            y_hat = np.ceil(y_hat)
            loss = mean_squared_error(clf_y_test, y_hat)
            mae = mean_absolute_error(clf_y_test, y_hat)
            mape = MAPE(clf_y_test, y_hat)
            data = np.array([city, k, 'base', time_taken_by_models['base'], loss, mae, mape])
            print(data)
            data = data.reshape((1, data.shape[0]))
            result = np.concatenate((result, data), axis=0)
            
            y_test_base_all = np.concatenate((y_test_base_all, clf_y_test), axis=0)
            y_hat_base_all = np.concatenate((y_hat_base_all, y_hat), axis=0)

        loss = mean_squared_error(y_test_dnn_all, y_hat_dnn_all)
        mae = mean_absolute_error(y_test_dnn_all, y_hat_dnn_all)
        mape = MAPE(y_test_dnn_all, y_hat_dnn_all)
        data = np.array([city, 'all', 'dnn', time_taken_by_models['dnn'], loss, mae, mape])
        print(data)
        data = data.reshape((1, data.shape[0]))
        result = np.concatenate((result, data), axis=0)

        loss = mean_squared_error(y_test_base_all, y_hat_base_all)
        mae = mean_absolute_error(y_test_base_all, y_hat_base_all)
        mape = MAPE(y_test_base_all, y_hat_base_all)
        data = np.array([city, 'all', 'base', time_taken_by_models['base'], loss, mae, mape])
        print(data)
        data = data.reshape((1, data.shape[0]))
        result = np.concatenate((result, data), axis=0)

    result = pd.DataFrame(result)
    result.to_csv('dataset\\anaheim\\' + 'reports\\method_5_cities_big_input.csv')

    return result

def MAPE(y_true, y_pred): 
    print('y_true.shape:', y_true.shape)
    print('y_pred.shape:', y_pred.shape)
    mask = np.logical_and((y_true != 0), ~np.isnan(y_true))
    # print('zeroes:', len(y_true)-np.count_nonzero(y_true))
    mask = np.logical_and(mask, ~np.isinf(y_true))
    mask = np.logical_and(mask, ~np.isnan(y_true))    
    mask = np.logical_and(mask, ~np.isinf(y_pred))
    mask = np.logical_and(mask, ~np.isnan(y_pred))
    return (np.fabs(y_true[mask] - y_pred[mask]) / y_true[mask]).mean() * 100

def create_best_conf_model_all():
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, kernel_initializer='normal', activation='relu'))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dense(output_dim, kernel_initializer='normal', activation='linear'))
    rmsprop = RMSprop(lr=10**-3)
    model.compile(loss='mse', optimizer=rmsprop, metrics=['mape', 'mae'])
    return model  

def create_best_conf_model_all_big_input(num_time_series, input_dim):
    model = Sequential()
    model.add(Dense(64, input_shape=(input_dim, num_time_series), activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Flatten())
    model.add(Dense(num_time_series, activation='linear'))
    rmsprop = RMSprop(lr=10**-3)
    model.compile(loss='mse', optimizer=rmsprop, metrics=['mape', 'mae'])
    return model  

def create_best_conf_model_low():
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, kernel_initializer='normal', activation='relu'))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dense(output_dim, kernel_initializer='normal', activation='linear'))
    rmsprop = RMSprop(lr=10**-3)
    model.compile(loss='mse', optimizer=rmsprop, metrics=['mape', 'mae'])
    return model        

def create_best_conf_model_medium():  
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, kernel_initializer='normal', activation='relu'))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dense(output_dim, kernel_initializer='normal', activation='linear'))
    rmsprop = RMSprop(lr=10**-3)
    model.compile(loss='mse', optimizer=rmsprop, metrics=['mape', 'mae'])
    return model         

def create_best_conf_model_high():
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, kernel_initializer='normal', activation='relu'))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dense(output_dim, kernel_initializer='normal', activation='linear'))
    rmsprop = RMSprop(lr=10**-3)
    model.compile(loss='mse', optimizer=rmsprop, metrics=['mape', 'mae'])
    return model        

def create_best_conf_model_big_input_low(num_time_series, input_dim):
    model = Sequential()
    model.add(Dense(64, input_shape=(input_dim, num_time_series), activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Flatten())
    model.add(Dense(num_time_series, activation='linear'))
    rmsprop = RMSprop(lr=10**-3)
    model.compile(loss='mse', optimizer=rmsprop, metrics=['mape', 'mae'])
    return model

def create_best_conf_model_big_input_medium(num_time_series, input_dim):
    model = Sequential()
    model.add(Dense(64, input_shape=(input_dim, num_time_series), activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Flatten())
    model.add(Dense(num_time_series, activation='linear'))
    rmsprop = RMSprop(lr=10**-3)
    model.compile(loss='mse', optimizer=rmsprop, metrics=['mape', 'mae'])
    return model

def create_best_conf_model_big_input_high(num_time_series, input_dim):
    model = Sequential()
    model.add(Dense(64, input_shape=(input_dim, num_time_series), activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Flatten())
    model.add(Dense(num_time_series, activation='linear'))
    rmsprop = RMSprop(lr=10**-3)
    model.compile(loss='mse', optimizer=rmsprop, metrics=['mape', 'mae'])
    return model

@ex.capture
def test_lr_model(root, _run):
    # constants
    from xgboost import XGBRegressor
    five_minutes_in_a_day = (24 * 60) // 5
    samples_per_day = five_minutes_in_a_day - input_dim
    weekday_days = 20
    weekend_days = 8
    train_batch = 5
    validate_batch = 5

    # load X, y
    x_train, x_val, x_test, y_train, y_val, y_test = get_input_cluster(cluster_c_indexes, root, weekdays=True, weekends=False, load_clean=True, normalized=False, test=False)
    x_train = np.concatenate((x_train, x_val), axis=0)
    y_train = np.concatenate((y_train, y_val), axis=0)
    print(x_train.shape)
    print(y_train.shape)
    del x_val
    del y_val
    del x_test
    del y_test

    losses = np.array([])
    val_losses = np.array([])
    val_mapes = np.array([])
    val_mases = np.array([])
    for days in range(train_batch, weekday_days, train_batch):
        x = x_train[:days*samples_per_day]
        y = y_train[:days*samples_per_day]
        x_val = x_train[days*samples_per_day: (days + validate_batch)*samples_per_day]
        y_val = y_train[days*samples_per_day: (days + validate_batch)*samples_per_day]
        
        # create model
        # clf = XGBRegressor()
        clf = LinearRegression()
        clf.fit(x, y)
        y_pred = clf.predict(x_val)
        y_t = clf.predict(x)

        loss = mean_squared_error(y, y_t)
        losses = np.append(losses, loss)

        val_loss = mean_squared_error(y_val, y_pred)
        val_losses = np.append(val_losses, val_loss)

        val_mape = MAPE(y_val, y_pred)
        val_mapes = np.append(val_mapes, val_mape)

        val_mase = MASE(y_val, y_pred)
        val_mases = np.append(val_mases, val_mase)

    avg_loss = np.mean(losses)
    avg_val_loss = np.mean(val_losses)
    avg_val_mape = np.mean(val_mapes)
    avg_val_mase = np.mean(val_mases)

    print('score:', clf.score(x_val, y_val))

    print('avg_loss', avg_loss)
    print('avg_val_loss:', avg_val_loss)
    print('avg_val_mape:', avg_val_mape)
    print('avg_val_mase:', avg_val_mase)

def run():
    batch_size = 256
    learning_rate = 0.001
    epochs = 500
    k = 1
    normalized = False
    root = 'dataset\\anaheim\\'
    weekdays = True
    weekends = False   
    method_number = 'method_5_big_input'
    eval(method_number)()
    
    learning_rate = [10**-3]
    layers = [1, 2, 3, 4, 6, 8]
    neurons = [16, 32, 64, 128]
    epochs = 500
run()
