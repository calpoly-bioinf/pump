from sklearn.cluster import KMeans

from .model import *
from .data import *
from .eval import loss_supervised, evaluation, test_metrics,\
                 loss_supervised_unsupervised, do_validation, accuracy
from .utils import *
from collections import deque

import tensorflow as tf
from keras.layers import Input

from keras.models import Model

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import joblib

import scipy.io
import numpy as np
import pandas as pd

def main_supervised_1view(FLAGS):
    """
    Perform supervised training with sparsity penalty.
    :return acc: the accuracy on trainng set.
    :return ae_supervised: the trained autoencoder.
    :return sess: the current session.
    """
    print('Supervised training...')
    data = read_data_sets(FLAGS)

    # combine training and validation

    do_val = True

    ae_supervised = supervised_1view(data, FLAGS, do_val)


    # get the manifold
    data_whole = np.concatenate((data.train.data, data.test.data), axis = 0)
    target_whole = np.concatenate((data.train.labels, data.test.labels), axis = 0)
    data_sets_whole = DataSet(data_whole, target_whole)
    
#     acc_test, test_pred = test_metrics(ae_supervised,data.test.data,data.test.labels)

    return ae_supervised

def supervised_1view(data, FLAGS, do_val = True):
    print(np.shape(data.train.data), np.shape(data.train.labels))
    ae_shape = [data.train.data.shape[1], [1024, 512], data.train.labels.shape[1]] #data.train.data
    input_shape = Input (shape = np.shape(data.train.data)[1:])
    output_shape = Input (shape = np.shape(data.train.labels)[1:])
    print('/////////////////////////////:ae_shape',ae_shape)
    ae = Autoencoder(ae_shape)
    ae.compile(loss=loss_supervised(knowledge_alpha=FLAGS.knowledge_alpha,A=data.A),optimizer='adam',metrics=accuracy,run_eagerly=True)
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=500, restore_best_weights=True)
    history = ae.fit(data.train.data, data.train.labels, batch_size=FLAGS.batch_size,epochs=FLAGS.epochs,validation_split=0.1,verbose=True,callbacks=[callback])
    print(ae.summary())
    y_pred = ae.call(data.test.data)
    to_print = pd.DataFrame(y_pred)
    to_print['method'] = 'ae.call'
    print(to_print.to_csv(sep=','))
    encoded = ae.call_encoded(data.test.data)
    to_print = pd.DataFrame(encoded)
    to_print['method'] = 'ae.call_encoded'
    print(to_print.to_csv(sep=','))
    to_print = pd.DataFrame(data.test.labels)
    to_print['method'] = 'data.test.labels'
    print(to_print.to_csv(sep=','))
    test_metrics(y_pred, data.test.labels)
    return ae

