import numpy as np
import pandas as pd
import tensorflow as tf

import sklearn.preprocessing
from sklearn.neighbors import NearestNeighbors

import sys
from pathlib import Path

def process(df2,n_neighbors=-1,mode=['connectivity','distance'][1],cosine=True):
    if type(df2) == pd.Series:
        df2 = df2.to_frame()
    #if cosine:
    #    df2.values[:,:] = df2.divide(np.sqrt((df2**2).sum(axis=1)),axis=0) # this makes a euclidean distance calculation eqv to cosine sim
    #A = pd.DataFrame.sparse.from_spmatrix(NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(df2).kneighbors_graph(df2,mode=mode))
    if n_neighbors == -1:
        n_neighbors = len(df2)
    A = pd.DataFrame.sparse.from_spmatrix(NearestNeighbors(n_neighbors=n_neighbors,metric='cosine', algorithm='brute').fit(df2).kneighbors_graph(df2,mode=mode,n_neighbors=n_neighbors))
    A.index = df2.index
    A.columns = df2.index
    return A

"""
def create_graph(df,mode=['connectivity','distance'][1],cosine=True,subtype_graphs=['Her2']):
    # df must have Subtype and Sample ID columns
    df_subtype = df.set_index('Subtype')
    samples = df['Sample ID']
    graph = pd.DataFrame(np.zeros((len(samples),len(samples))),index=samples,columns=samples)
    for subtype in subtype_graphs:
        if type(subtype) == str and subtype != 'NC':
            subtype_graph = process(df_subtype.loc[subtype].set_index('Sample ID'),mode=mode,cosine=cosine)
            graph += subtype_graph.reindex(index=graph.index,columns=graph.columns).fillna(0)
    return graph
"""

def create_graph(df,mode=['connectivity','distance'][1],cosine=True):
    # df must have Subtype and Sample ID columns
    graph = process(df,mode=mode,cosine=cosine).fillna(0)
    return graph

#data_file=f"/large/metabric/expression_with_gene_ids_min_max_no_pam50.csv.gz",
#targets_file=f"{home}/metabric/pam50_claudinlow_subtype.csv",
#sample_sample_graph="../graphs/A_TMB.csv",
#encoding_file='/large/metabric/underspecification/results/experiment2_ae.call_encoded.csv',              #data_test_labels_file='/large/metabric/underspecification/results/experiment2_data.test.labels.csv'
class Data:
    def __init__(self,data_file,targets_file,sample_sample_graph,encoding_file,data_test_labels_file,num_to_target_file,index_dir): 
        self.index_dir = index_dir
        self.num_to_target_file = num_to_target_file
        self.data_file = data_file
        self.targets_file = targets_file
        self.sample_sample_graph = sample_sample_graph
        self.df_nn_sample_rep = pd.read_csv(encoding_file,sep=",",header=None,index_col=0)
        self.df_nn_sample_rep.columns = list(self.df_nn_sample_rep.columns[:-5]) + ['seed','rep','shifted','cluster','dataset'][::-1]
        self.df_data_test_labels_all = pd.read_csv(data_test_labels_file,sep=",",header=None,index_col=0)
        self.df_data_test_labels_all.columns = list(self.df_data_test_labels_all.columns[:-5]) + ['seed','rep','shifted','cluster','dataset'][::-1]
        self.df_targets = pd.read_csv(targets_file,index_col=0)
        
        self.helper_ret = pydeeptype2.data.load_data_helper(data_file,targets_file,sample_sample_graph)
        
    def load(self,cluster,shifted,rep,seed):
        num_to_target_file = self.num_to_target_file
        index_dir = self.index_dir
        cluster = int(cluster)
        shifted = str(shifted)
        rep = int(rep)
        key = f"{cluster},{shifted}"
        
        index_file=f"{index_dir}/{key}/index_df_{rep}.csv"

        seed = int(seed)
    
        np.random.seed(seed)
        tf.random.set_seed(seed)
        data_sets = pydeeptype2.data.read_data_sets_with_helper(self.helper_ret,index_file)

        map_dict = dict(pd.read_csv(num_to_target_file,index_col=0).iloc[:,0])

        ixs = np.where(
            (self.df_data_test_labels_all['cluster'] == cluster) & 
            (self.df_data_test_labels_all['rep'] == rep) &
            (self.df_data_test_labels_all['shifted'] == bool(shifted)) &
            (self.df_data_test_labels_all['seed'] == seed)
        )[0]
        
        test_labels_all = self.df_data_test_labels_all.iloc[ixs,:6].values
        indices, df_data_test_labels = pydeeptype2.eval.extract_index_np(test_labels_all)
        #df_data_test_labels = pd.DataFrame(df_data_test_labels,index=indices)
        #df_data_test_labels.columns = df_data_test_labels.columns.map(map_dict)
        
        df = self.df_nn_sample_rep.drop(['seed','rep','shifted','cluster','dataset'],axis=1).iloc[ixs,:]
        #df.index = y_data_test.index
        #df = self.df_nn_sample_rep.copy()
        df['Sample ID'] = data_sets.sample_index[indices]
        df = df.set_index('Sample ID')
        
        #df_labels = df_data_test_labels.astype(float)
        #df_labels.index = df.index
        #y_data_test = df_labels.idxmax(axis=1) #pd.Series(np.argmax(df_data_test_labels.values[ixs,:], axis=1),index=df.index).map(map_dict)
        df['Subtype'] = self.df_targets.loc[df.index]

        return df

def num_overlapping_connections(graph,graph2): # assumes binary+int graphs
    return np.sum(np.sum((graph == graph2) & (graph == 1)))

def mae(graph,graph2):
    #index_inter = list(set(graph.index).intersection(set(graph2.index)))
    #graph = graph.reindex(index=index_inter,columns=index_inter)
    #graph2 = graph2.reindex(index=index_inter,columns=index_inter)
    return np.sum(np.sum(np.abs(graph-graph2)))/(len(graph)*len(graph))

from sklearn.metrics.pairwise import cosine_similarity as cos_sim

def cosine_similarity(graph,graph2):
    #index_inter = list(set(graph.index).intersection(set(graph2.index)))
    #graph = graph.reindex(index=index_inter,columns=index_inter)
    #graph2 = graph2.reindex(index=index_inter,columns=index_inter)
    return cos_sim([list(graph2.values.flat)],[list(graph.values.flat)])[0,0]
    #return cos_sim(graph2.flat,graph.flat)

def compute_score(df_paired_perf,graph,dt,score_func=cosine_similarity):
    #source = df_paired_perf.set_index('subtype').loc[subtype].reset_index().set_index(['subtype','cluster','rep','seed'])
    source = df_paired_perf.set_index(['cluster','rep','seed'])
    source['score'] = 0

    #dt = Data(encoding_file=encoding_file,data_test_labels_file=data_test_labels_file)
    shifted = 'False' # we want to only run on this data
    for cluster in df_paired_perf.cluster.unique():
        for rep in df_paired_perf.rep.unique():
            for seed in df_paired_perf.seed.unique():
                encoding = dt.load(cluster,shifted,rep,seed)#.set_index('Sample ID')
                # got to drop bad columns and figure out the NaNs
                index_inter = list(set(graph.index).intersection(set(encoding.index)))
                subset_encoding = encoding.loc[index_inter]
                subset_graph = graph.reindex(index=index_inter,columns=index_inter)
                encoding_graph = create_graph(subset_encoding.drop('Subtype',axis=1))
                source.loc[(cluster,rep,seed),'score'] = score_func(encoding_graph,subset_graph)
                
    return source

def choose(source,top_n,ranking_column='score'):
    choices_unshifted_f1 = source.reset_index().groupby(['cluster','rep']).apply(
        lambda df: df.set_index('seed')['f1-score (unshifted)'].sort_values(ascending=False)[:top_n].index)
    choices_unshifted_f1.name = 'seed'
    if top_n == 1:
        for index in choices_unshifted_f1.index:
            choices = choices_unshifted_f1.loc[index]
            choices_unshifted_f1.loc[index] = int(choices[0])
        return choices_unshifted_f1.reset_index()
    
    for index in choices_unshifted_f1.index:
        choices = choices_unshifted_f1.loc[index]
        df_choices = source.reset_index().set_index(['cluster','rep','seed'])
        indices = []
        for choice in choices:
            indices.append(tuple(list(index)+[choice]))
        df_choices = df_choices.loc[indices]
        choices_unshifted_f1.loc[index] = df_choices.reset_index().set_index('seed')['score'].idxmin()
    return choices_unshifted_f1.reset_index()

def evaluate_mean(results,choices):
    return results.reset_index().set_index(['cluster','rep','seed']).loc[[tuple(e) for e in choices.values]]['f1-score (shifted)'].mean()