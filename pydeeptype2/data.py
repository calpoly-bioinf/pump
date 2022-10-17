from __future__ import division
from __future__ import print_function


import numpy as np
import scipy.io as sio
import pandas as pd


class DataSet(object):

  def __init__(self, data, labels):

    self._num_examples = labels.shape[0]

    self._data = data
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0
    self._A = None

  @property
  def data(self):
    return self._data

  @property
  def labels(self):
    return self._labels

  @property
  def A(self):
    return self._A

  @A.setter
  def A(self,A):
    self._A = A

  @property
  def num_examples(self):
    return self._num_examples
  @property
  def start_index(self):
      return self._index_in_epoch
  @labels.setter
  def labels(self,values):
    self._labels = values


  def next_batch(self, batch_size, UNSUPERVISED = False):
    """Return the next `batch_size` examples from this data set."""
    n_sample = self.num_examples
    start = self._index_in_epoch
    end = self._index_in_epoch + batch_size
    end = min(end, n_sample)
    id = range(start, end)
    data_input = self._data[id, :]
    if ~UNSUPERVISED:
        target_input = self._labels[id, :]
    else: target_input = []

    self._index_in_epoch = end

    if end == n_sample:
        self._index_in_epoch = 0

    return data_input, target_input



def read_data_sets(FLAGS, test = False):

    if test:
        index, data_whole, targets_whole = load_biology_data_for_test(FLAGS)
        data_set = DataSet(data_whole, targets_whole)
        return data_set, index

    else:
        class DataSets(object):
            pass
        data_sets = DataSets()
        data_train, data_test, targets_train, targets_test, A, sample_index, targets = load_biology_data(FLAGS)
        data_sets.train = DataSet(data_train, targets_train)
        data_sets.test = DataSet(data_test, targets_test)
        data_sets.A = A
        data_sets.sample_index = sample_index
        data_sets.targets = targets

        return data_sets
    
def read_data_sets_with_helper(helper_ret,index_file):
    class DataSets(object):
        pass
    data_sets = DataSets()
    data_train, data_test, targets_train, targets_test, A, sample_index, targets = load_biology_data_with_helper(helper_ret,index_file)
    data_sets.train = DataSet(data_train, targets_train)
    data_sets.test = DataSet(data_test, targets_test)
    data_sets.A = A
    data_sets.sample_index = sample_index
    data_sets.targets = targets

    return data_sets


def make_center_set(centers, assignments, FLAGS):

    Ass_matrix = np.zeros([FLAGS.num_train, FLAGS.num_classes])
    for i in range(FLAGS.num_train):
        Ass_matrix[i, assignments[i]] = 1
    center_matrix = np.dot(Ass_matrix, centers)
    center_set = DataSet(center_matrix, Ass_matrix)

    return center_set


def fill_feed_dict_ae_for_hidden(data_set, input_pl, FLAGS):
    input_feed = data_set.data
    feed_dict = {
        input_pl: input_feed}

    return feed_dict


def fill_feed_dict_ae_test(data_set, input_pl, target_pl, FLAGS):
    input_feed = data_set.data
    target_feed = data_set.labels
    feed_dict = {
        input_pl: input_feed,
        target_pl: target_feed}

    return feed_dict

def load_data_helper(data_file,targets_file,sample_sample_graph):
    data_df = pd.read_csv(data_file,index_col=0)
    #index_df = pd.read_csv(FLAGS.index_file,index_col=0)
    targets = pd.read_csv(targets_file,index_col=0).iloc[:,0]

    #data_df = data_df.loc[index_df.index]
    A = pd.read_csv(sample_sample_graph,index_col=0)
    
    return data_df,targets,A
    
def load_biology_data_with_helper(helper_ret,index_file,print_samples_dropped=False):
    index_df = pd.read_csv(index_file,index_col=0)
    data_df,targets,A = helper_ret
    to_be_dropped = set(data_df.index) - set(index_df.index)
    if print_samples_dropped:
        print("Samples being dropped",len(to_be_dropped),set(data_df.index) - set(index_df.index))
    data_df = data_df.loc[index_df.index]
    n_sample = len(data_df)
    # perform shuffle to batch reasons
    index = np.random.permutation(n_sample)
    sample_index = data_df.index
    
    A = A.reindex(index=index_df.index,columns=index_df.index) # same order as df
    
    data_df = data_df.iloc[index]
    index_df = index_df.iloc[index]
    targets = targets.loc[data_df.index]
        
    num_to_targets = pd.Series(targets.unique())
    num_to_targets.to_csv("/tmp/num_to_target.csv")
    n_label = len(num_to_targets)
    Y = np.zeros([n_sample, n_label])
    for i,target in enumerate(num_to_targets):
        id = (targets == target).values.nonzero()[0]
        Y[id, i] = 1
    X = data_df.values
    targets_as_nums = np.where(Y==1)[1]
    
    #index_df = index_df.iloc[index] # order change here
    train_index = np.where((index_df['train']==True) | (index_df['val']==True))[0]
    test_index = np.where(index_df['test']==True)[0]
    
    data_train = X[train_index, :]
    targets_train = np.float32(Y[train_index, :])
    targets_train[:,0] = index[train_index] #index_df.index[train_index]
    
#     data_validation = X[train_size:train_size+validation_size, :]
#     targets_validation = np.float32(Y[train_size:train_size+validation_size, :])

    data_test = X[test_index, :]
    targets_test = np.float32(Y[test_index, :])    
    targets_test[:,0] = index[test_index] #index_df.index[test_index]
    
    return data_train, data_test, targets_train, targets_test, A, sample_index, targets


def load_biology_data(FLAGS,print_samples_dropped=False):
    data_df = pd.read_csv(FLAGS.data_file,index_col=0)
    index_df = pd.read_csv(FLAGS.index_file,index_col=0)
    targets = pd.read_csv(FLAGS.targets_file,index_col=0).iloc[:,0]
    to_be_dropped = set(data_df.index) - set(index_df.index)
    if print_samples_dropped:
        print("Samples being dropped",len(to_be_dropped),set(data_df.index) - set(index_df.index))

    data_df = data_df.loc[index_df.index]
    n_sample = len(data_df)
    # perform shuffle to batch reasons
    index = np.random.permutation(n_sample)
    sample_index = data_df.index
    
    A = pd.read_csv(FLAGS.sample_sample_graph,index_col=0)
    A = A.loc[index_df.index,index_df.index] # same order as df
    
    data_df = data_df.iloc[index]
    index_df = index_df.iloc[index]
    targets = targets.loc[data_df.index]
        
    num_to_targets = pd.Series(targets.unique())
    num_to_targets.to_csv("/tmp/num_to_target.csv")
    n_label = len(num_to_targets)
    Y = np.zeros([n_sample, n_label])
    for i,target in enumerate(num_to_targets):
        id = (targets == target).values.nonzero()[0]
        Y[id, i] = 1
    X = data_df.values
    targets_as_nums = np.where(Y==1)[1]
    
    #index_df = index_df.iloc[index] # order change here
    train_index = np.where((index_df['train']==True) | (index_df['val']==True))[0]
    test_index = np.where(index_df['test']==True)[0]
    
    data_train = X[train_index, :]
    targets_train = np.float32(Y[train_index, :])
    targets_train[:,0] = index[train_index] #index_df.index[train_index]
    
#     data_validation = X[train_size:train_size+validation_size, :]
#     targets_validation = np.float32(Y[train_size:train_size+validation_size, :])

    data_test = X[test_index, :]
    targets_test = np.float32(Y[test_index, :])    
    targets_test[:,0] = index[test_index] #index_df.index[test_index]
    
    return data_train, data_test, targets_train, targets_test, A, sample_index, targets
