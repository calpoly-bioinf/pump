import tensorflow as tf
from training import *
from flags import set_flags
import pickle
import os
import data


if __name__ == '__main__':
    FLAGS = set_flags()
    seed = FLAGS.seed
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Create folders
    if not os.path.exists(FLAGS.results_dir):
        os.makedirs(FLAGS.results_dir)

    # create autoencoder and perform training
    
    AE = main_supervised_1view(FLAGS)
