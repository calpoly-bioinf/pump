from __future__ import division
import os
from os.path import join as pjoin
import sys
import numpy as np
import tensorflow as tf
import numpy as np
import argparse

def set_flags():
    parser = argparse.ArgumentParser(description='set flags for the autoencoder')
    NUM_NODES = [1024, 512]
    NUM_HIDDEN = len(NUM_NODES)

    NUM_BATCH_SIZE = 256
    LEARNING_RATE = 1.e-3
    EPOCHS = 100 # shorter training process to save time
    LAMBDA = 0.006
    ALPHA = 1.2

    DATA_FILE = '/large/metabric/expression_with_gene_ids_min_max.csv.gz'
    RESULT_DIR = 'results/'
    
    # Constants
    parser.add_argument('--batch_size', default=NUM_BATCH_SIZE,
                    help='Batch size. Must divide evenly into the dataset sizes')
    parser.add_argument('--learning_rate', default=LEARNING_RATE,
                    help='Initial learning rate')
    parser.add_argument('--epochs', default=EPOCHS,type=int,
                    help='Number of training steps')
    parser.add_argument('--display_steps', default=20,
                    help='Display the middle results.')
    parser.add_argument('--sample_sample_graph', default='/large/metabric/sample_sample_graph.csv')

    parser.add_argument('--beta', default=ALPHA, help='K-means loss coefficient.')
    parser.add_argument('--alpha', default=LAMBDA, help='sparsity penalty.')
    parser.add_argument('--parameter_tune', default=False, help='Tune parameters or not.')
    parser.add_argument('--knowledge_alpha', default=0,type=float)
    parser.add_argument('--seed', default=0,type=int)
    
    parser.add_argument('--index_file', default=None, help='index file for train/test/val.')
    
    # Directories
    parser.add_argument('--data_file', default=DATA_FILE, help='Data file location.')
    parser.add_argument('--targets_file', default=None, help='Targets file location.')
    parser.add_argument('--results_dir', default=RESULT_DIR, help='Directory to put the results.')

    args = parser.parse_args()
    #if type(args.NN_dims_1) == str:
    #    args.NN_dims_1 = str(args.NN_dims_1).split(',')
    #    args.NN_dims_1 = [int(x) for x in args.NN_dims_1]
    #    args.NN_dims_1 = [args.NN_dims_1[0],args.NN_dims_1[1:-1],args.NN_dims_1[-1]]
    return args