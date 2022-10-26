import numpy as np
import pandas as pd
import altair as alt
import scipy.stats as stats
import scipy.io
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn import svm, ensemble, neural_network 
from IPython.display import display, Markdown, Latex
import random
import sys
import os

import compare_graphs as cg


class UnderspecificationAnalysis:
    
    def __init__(self, output_dir="data"):
        if output_dir[-1] != "/":
            output_dir = output_dir + "/" 
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.output_dir = output_dir
        
    def initialize(self, X, y, num_clusters, cluster_focus, num_shifted_sets, random_state):
        if X.shape[0] != y.shape[0]:
            raise IndexError("X and y sets must have an equal number of samples.")
        if cluster_focus not in y.unique():
            raise ValueError("Cluster focus " + str(cluster_focus) + " must be a class in y.")
        self.Xy = X.join(y).dropna()
        self.X = self.Xy.drop(['Pam50 + Claudin-low subtype'], axis=1)
        self.y = self.Xy['Pam50 + Claudin-low subtype']
        self.num_clusters = num_clusters
        self.cluster_focus = cluster_focus
        self.num_shifted_sets = num_shifted_sets
        self.random_state = random_state
        
    def pca_analysis(self, X, y, subdirectory):
        
        # Standardize the data to have a mean of ~0 and a variance of 1
        X_std = StandardScaler().fit_transform(X)
        pca = PCA(n_components=20)
        principalComponents = pca.fit_transform(X_std)
        
        # PCA Variance Plot
        features = range(pca.n_components_)
        plt.bar(features, pca.explained_variance_ratio_, color='black')
        plt.xlabel('PCA features')
        plt.ylabel('variance %')
        plt.xticks(features)
        pca_components = pd.DataFrame(principalComponents,index=y.index)
        plt.title("Variance Drop Off after 0,1,2,...")
        plt.savefig(self.output_dir + subdirectory + '/pca_bar.png')
        plt.clf()
        
        # PCA Scatter Plot
        plt.scatter(pca_components[0], pca_components[1], alpha=.1, color='black')
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        plt.savefig(self.output_dir + subdirectory + '/pca_scatter.png')
        plt.clf()
        
        return pca_components
    
    def kmeans_analysis(self, X, y, num_clusters, subdirectory):
        
        Xy = X.join(y)
        
        # Plot k-means inertia
        ks = range(1, 10)
        inertias = []
        for k in ks:
            model = KMeans(n_clusters=k)
            model.fit(X)
            inertias.append(model.inertia_)
        plt.plot(ks, inertias, '-o', color='black')
        plt.xlabel('number of clusters, k')
        plt.ylabel('inertia')
        plt.xticks(ks)
        plt.savefig(self.output_dir + subdirectory + '/kmeans_inertia.png')
        plt.clf()
        
        # Plot user-specified k-means
        model = KMeans(n_clusters=num_clusters)
        model.fit(X)
        Xy['kmean-label'] = model.labels_
        kplt = Xy['kmean-label'].value_counts().loc[list(range(num_clusters))].plot.barh()
        kplt.figure.savefig(self.output_dir + subdirectory + '/kmeans.png')
        
        return Xy
        
    def plot_cluster_analysis(self, X, y, Xy, pca_components, subdirectory):
        data = pca_components.copy()
        data.columns = ["PC"+str(c+1) for c in data.columns]
        data['cluster'] = Xy['kmean-label']
        data_subtype = data.copy()
        chart = alt.Chart(data).mark_circle(size=60).encode(
            x="PC1",
            y="PC2",
            color='cluster:N',
        )
        chart.save(self.output_dir + subdirectory + '/clusters.html')
        
    def analyze_dataset(self, X, y, cluster_focus, subdirectory='analysis', num_clusters=3):
        if not os.path.exists(self.output_dir + subdirectory):
            os.makedirs(self.output_dir + subdirectory)
        if cluster_focus not in y.unique():
            raise ValueError("Cluster focus " + str(cluster_focus) + " must be a class in y.")
        if cluster_focus is not None:
            X = X.loc[y == cluster_focus]
            y = y.loc[y == cluster_focus]
        print("Running PCA Analysis ...")
        pca_components = self.pca_analysis(X, y, subdirectory)
        print("Running K-means Analysis ...")
        Xy = self.kmeans_analysis(X, y, num_clusters, subdirectory)
        print("Writing Cluster Analysis ...")
        self.plot_cluster_analysis(X, y, Xy, pca_components, subdirectory)
        print(f"-----[ COMPLETE ]-----\nCheck /{self.output_dir}{subdirectory} for the results.")
        return Xy
    
    def initialize_shifted_directory(self, directory):
        if not os.path.exists(self.output_dir + "indices/" + str(directory)):
            os.makedirs(self.output_dir + "indices/" + str(directory))
    
    def write_shifted_datasets(self, sets_by_group, subdirectory):
        for key in sets_by_group.keys():
            sets = sets_by_group[key]
            key_str = ",".join([str(c) for c in key])
            pd.Series(sets).to_json(self.output_dir + f'sets_{key_str}.json')
        for key in sets_by_group.keys():
            sets = sets_by_group[key]
            key_str = ",".join([str(c) for c in key])
            self.initialize_shifted_directory(key_str)
            for j,curr in enumerate(sets):
                index_df = curr
                index_df.to_csv(self.output_dir + f"{subdirectory}/{key_str}/index_df_{j}.csv",index=True)
    
    def create_clustering_shifted_datasets(self, X, y, Xy_filtered, cluster_focus, num_clusters=3, subdirectory='indices', num_shifted_sets=50, split_ratio=[0.7, 0.1, 0.2]):
        if not os.path.exists(self.output_dir + subdirectory):
            os.makedirs(self.output_dir + subdirectory)
        sets_by_group = {}
        df_filtered = Xy_filtered
        df = X.join(y).dropna()
        if cluster_focus not in y.unique():
            raise ValueError("Cluster focus " + str(cluster_focus) + " must be a class in y.")
        print("Creating shifted sets ...")
        if sum(split_ratio) != 1:
            raise ValueError("Split Ratio must total up to 1 exactly")

        for cluster in range(num_clusters): # clusters selected from colors above
            if type(cluster) != list:
                cluster = [cluster]

            shifted_set = []
            for c in cluster:
                shifted_set.extend(list(df_filtered[df_filtered['kmean-label'] == c].index.values))

            for shifted in [True,False]:
                sets = []
                sets_by_group[tuple(cluster+[shifted])] = sets
                for i in range(num_shifted_sets):
                    frac_in_shifted_set = len(shifted_set)/len(df_filtered)

                    # test_size = round(frac_in_shifted_set*df[y.name].value_counts().drop(cluster_focus).sum()) + len(shifted_set)
                    test_size = round(len(df)*split_ratio[2])
                    val_size = round(len(df)*split_ratio[1])
                    train_size = len(df) - test_size - val_size

                    df_not_subtype = df[df[y.name] != cluster_focus]

                    random.seed(i) # set the seed so we get the same dataset

                    # create a list of sample ids that are not of the specific subtype
                    test_list = list(df_not_subtype.index[random.sample(range(0, len(df_not_subtype)), test_size-len(shifted_set))])

                    # drop out shifted test set from subtype samples
                    df_dropped = df_filtered.drop(shifted_set)
                    # randomly create a non-shifted test set of the same size
                    if len(shifted_set) > len(df_dropped):
                        shifted_set = random.sample(shifted_set, len(df_dropped)) #shifted_set[:
                    reg_set = list(df_dropped.index[random.sample(range(0, len(df_dropped)), len(shifted_set))])               

                    # remove everything so far
                    remaining_set = df.drop(shifted_set).drop(reg_set).drop(test_list).index

                    # depending on the option, add in the portion of the test set for the subtype
                    if shifted:
                        test_list += shifted_set
                    else:
                        test_list += reg_set 
                    #df.loc[test_list]['Pam50 + Claudin-low subtype'].value_counts()

                    #val_list = shifted_set[random.sample(range(0, len(shifted_set)), val_size)]
                    val_list = remaining_set[random.sample(range(0, len(remaining_set)), val_size)]

                    #remnants = list(set(shifted_set) - set(test_list) - set(val_list))

                    # training is everything else
                    train_list = list(set(remaining_set) - set(val_list))

                    #print(len(train_list), len(val_list), len(test_list))

                    curr = pd.DataFrame(index=df.index,columns=["train", "test", "val"])
                    curr.loc[train_list,:] = False
                    curr.loc[train_list,'train'] = True
                    curr.loc[val_list,:] = False
                    curr.loc[val_list,'val'] = True
                    curr.loc[test_list,:] = False
                    curr.loc[test_list,'test'] = True
                    curr.loc[curr.isna().sum(axis=1)==3,:] = False
                    sets.append(curr)
        print("Writing out sets ...")
        self.write_shifted_datasets(sets_by_group, subdirectory)
        print(f"-----[ COMPLETE ]-----\nCheck /{self.output_dir}{subdirectory} for the results.")
        
    def load_shifted_split(self, X, y, file_path):
        splits_df = pd.read_csv(file_path)
        splits = {'train':[], 'test':[], 'val':[]}
        for index, row in splits_df.iterrows():
            if bool(row['train']):
                splits['train'].append(row['Sample ID'])
            elif bool(row['test']):
                splits['test'].append(row['Sample ID'])
            else:
                splits['val'].append(row['Sample ID'])
        Xtrain = X.loc[splits['train']]
        ytrain = y.loc[splits['train']]
        Xtest = X.loc[splits['test']]
        ytest = y.loc[splits['test']]
        Xval = X.loc[splits['val']]
        yval = y.loc[splits['val']]
        return {'Xtrain':Xtrain, 'Xval':Xval, 'Xtest':Xtest, 'ytrain':ytrain, 'yval':yval, 'ytest':ytest}
    
    def update_confusion_matrices(self, cms, test, pred):
        # [[TN, FP], 
        #  [FN, TP]]
        if test == pred:
            # Update positives
            for key in cms.keys():
                if key == test:
                    cms[key][1][1] += 1
                else:
                    cms[key][0][0] += 1
        else:
            # Update negatives
            cms[test][1][0] += 1
            cms[pred][0][1] += 1
        return cms
    
    def calculate_macro_average(self, metrics, N):
        # print(metrics)
        macro_support = metrics['support'].sum()
        macro_precision = metrics['precision'].sum() / N
        macro_recall = metrics['recall'].sum() / N
        macro_f1 = metrics['f1-score'].sum() / N
        return ['macro_avg', macro_precision, macro_recall, macro_f1, macro_support]
    
    def calculate_weighted_average(self, metrics, N):
        # print(metrics)
        weighted_support = metrics['support'].sum()
        weighted_precision = sum([row['precision'] * (row['support'] / weighted_support) for index, row in metrics.iterrows()])
        weighted_recall = sum([row['recall'] * (row['support'] / weighted_support) for index, row in metrics.iterrows()])
        weighted_f1 = sum([row['f1-score'] * (row['support'] / weighted_support) for index, row in metrics.iterrows()])
        return ['weighted_avg', weighted_precision, weighted_recall, weighted_f1, weighted_support]
    
    def calculate_performance_metrics(self, ytest, ypred, key, seed):
        rep = 0
        N = len(ytest.unique())
        ytest = ytest.to_list()
        cms = {}
        for subtype in ytest:
            cms[subtype] = [[0, 0], [0, 0]]
        for i in range(len(ytest)):
            cms = self.update_confusion_matrices(cms, ytest[i], ypred[i])
        # print(cms)
        metrics_columns = ['key', 'rep', 'seed', 'subtype', 'precision', 'recall', 'f1-score', 'support']
        metrics = pd.DataFrame(columns=metrics_columns)
        for subtype in cms.keys():
            precision = cms[subtype][0][0] / (cms[subtype][0][0] + cms[subtype][0][1])
            recall = cms[subtype][0][0] / (cms[subtype][0][0] + cms[subtype][1][0])
            f1 = cms[subtype][0][0] / (cms[subtype][0][0] + 0.5 * (cms[subtype][0][1] + cms[subtype][1][0]))
            support = cms[subtype][1][0] + cms[subtype][1][1]
            metrics = pd.concat([metrics, pd.DataFrame([[key, rep, seed, subtype, precision, recall, f1, support]], columns=metrics_columns)])
        macro_avg = pd.DataFrame([[key, rep, seed] + self.calculate_macro_average(metrics, N)], columns=metrics_columns)
        weighted_avg = pd.DataFrame([[key, rep, seed] + self.calculate_weighted_average(metrics, N)], columns=metrics_columns)
        metrics = pd.concat([metrics, macro_avg])
        metrics = pd.concat([metrics, weighted_avg])
        # print(metrics)
        return metrics
    
    def svm_performance(self, Xtrain, ytrain, Xtest, ytest, key, seed):
        clf = svm.SVC(kernel='linear', C=0.001, decision_function_shape='ovo', verbose=False).fit(Xtrain, ytrain)
        ypred = clf.predict(Xtest)
        return self.calculate_performance_metrics(ytest, ypred, key, seed)
        
            
    def rf_performance(self, Xtrain, ytrain, Xtest, ytest, key, seed):
        clf = ensemble.RandomForestClassifier(max_depth=3, random_state=seed).fit(Xtrain, ytrain)
        ypred = clf.predict(Xtest)
        return self.calculate_performance_metrics(ytest, ypred, key, seed)
            
    def nn_performance(self, Xtrain, ytrain, Xtest, ytest, key, seed):
        clf = neural_network.MLPClassifier(random_state=seed, max_iter=500).fit(Xtrain, ytrain)
        ypred = clf.predict(Xtest)
        return self.calculate_performance_metrics(ytest, ypred, key, seed)
        
    
    def evaluate_shifted_sets(self, X, y, subdirectory='results', random_state=0):
        if not os.path.exists(self.output_dir + "results"):
            os.makedirs(self.output_dir + "results")
        svm_results = pd.DataFrame(columns=['key', 'rep', 'seed', 'subtype', 'precision', 'recall', 'f1-score', 'support'])
        # rf_results = pd.DataFrame(columns=['key', 'rep', 'seed', 'subtype', 'precision', 'recall', 'f1-score', 'support'])
        # nn_results = pd.DataFrame(columns=['key', 'rep', 'seed', 'subtype', 'precision', 'recall', 'f1-score', 'support'])
        print("Evaluating models ...")
        for subdir in [x[0] for x in os.walk(self.output_dir + 'indices')]:
            if subdir[-1] == 'e':
                for file in [f for f in os.listdir(subdir) if os.path.isfile(os.path.join(subdir, f))]:
                    print("    Evaluating: " + subdir + "/" + file)
                    sets = self.load_shifted_split(X, y, subdir + "/" + file)
                    Xtrain = pd.concat([sets['Xtrain'], sets['Xval']])
                    ytrain = pd.concat([sets['ytrain'], sets['yval']])
                    key = subdir.split("/")[-1]
                    seed = file.split("_")[-1]
                    seed = int(seed[:len(seed)-4])
                    svm_results = pd.concat([svm_results, self.svm_performance(Xtrain, ytrain, sets['Xtest'], sets['ytest'], key, seed)])
                    # rf_results = pd.concat([rf_results, self.rf_performance(Xtrain, ytrain, sets['Xtest'], sets['ytest'], key, seed)])
                    # nn_results = pd.concat([nn_results, self.nn_performance(Xtrain, ytrain, sets['Xtest'], sets['ytest'], key, seed)])
        svm_results.reset_index(drop=True)
        print("Writing out results ...")
        svm_results.to_csv(self.output_dir + f"{subdirectory}/svm_results.csv")
        # rf_results.reset_index(drop=True)
        # rf_results.to_csv(self.output_dir + f"{subdirectory}/rf_results.csv")
        # nn_results.reset_index(drop=True)
        # nn_results.to_csv(self.output_dir + f"{subdirectory}/nn_results.csv")
        print(f"-----[ COMPLETE ]-----\nCheck /{self.output_dir}{subdirectory} for the results.")
        
    def analyze(self, X, y, num_clusters=3, cluster_focus=None, num_shifted_sets=50, random_state=0):
        print("[ 10% ] Initializing data ...")
        self.initialize(X, y, num_clusters, cluster_focus, num_shifted_sets, random_state)
        print("[ 20% ] Analyzing data ...")
        # Xy = self.analyze_data()
        print("[ 30% ] Creating shifted datasets ...")
        # self.create_shifted_datasets(Xy)
        print("[ 50% ] Evaluating shifted performances ...")
        self.evaluate_shifted_performances(random_state)
        print("[ 90% ] Analyzing for underspecification ...")
        # self.write_analysis()
        print(" ----- [ ANALYSIS COMPLETE ] -----")
        
    def process(self, df):
        df = df.set_index('shifted')
        unshifted = df.loc['False','f1-score']
        shifted = df.loc['True','f1-score']
        return unshifted-shifted
    
    def evaluate_shifted_unshifted(self, results_df, paired_perf_df):
        paired_perf_df.name = "f1-score (unshifted-shifted)"
        paired_perf_df = paired_perf_df.reset_index()
        shifted_f1 = results_df.set_index('shifted').loc['True',['rep','seed','subtype','cluster','f1-score']]
        shifted_f1.columns = list(shifted_f1.columns[:-1]) + ['f1-score (shifted)']
        paired_perf_df = paired_perf_df.set_index(['rep','seed','subtype','cluster']).join(shifted_f1.set_index(['rep','seed','subtype','cluster'])).reset_index()
        unshifted_f1 = results_df.set_index('shifted').loc['False',['rep','seed','subtype','cluster','f1-score']]
        unshifted_f1.columns = list(unshifted_f1.columns[:-1]) + ['f1-score (unshifted)']
        paired_perf_df = paired_perf_df.set_index(['rep','seed','subtype','cluster']).join(unshifted_f1.set_index(['rep','seed','subtype','cluster'])).reset_index()
        return paired_perf_df
    
    def plot_cluster_performances(self, paired_perf_df, subdirectory):
        source = paired_perf_df.set_index('subtype').loc['Her2']
        source = source.set_index(['rep','cluster'])
        for index in source.index.unique():
            f = source.loc[index,'f1-score (shifted)']
            r = source.loc[index,'f1-score (unshifted)']
            source.loc[index,'rank'] = stats.rankdata(-(f+r)/2)
            source.loc[index,'avg f1-score'] = ((f+r)/2).values #2*(f*r)/(f+r)
        source = source.reset_index()
        top_k = 5
        mask = source['rank'] > top_k
        source.loc[:,'rank'] = source.loc[:,'rank'].astype(int).astype(str)
        source.loc[mask,'rank'] = f'>{top_k}'
        chart = alt.Chart(source).transform_calculate(
            Seed="datum.seed"# + '-' + datum.seed"
        ).mark_point(filled=True).encode(
            x=alt.X('f1-score (unshifted):Q',scale=alt.Scale(domain=[0, 1])),
            y=alt.Y('f1-score (shifted):Q'),
            size='avg f1-score:Q',
            column='cluster',
            row='rep',
            color='rank:N',
            tooltip=['Seed:Q','rank','avg f1-score','f1-score (unshifted)','f1-score (shifted)']
        )
        chart.resolve_scale(x='independent')
        chart.save(self.output_dir + subdirectory + '/ranking_results.html',scale_factor=2.0)
        
    def analyze_underspecification(self, filename, subdirectory='results'):
        results_df = pd.read_csv(self.output_dir + subdirectory + "/" + filename)
        results_df['cluster'] = ''
        results_df['shifted'] = ''
        results_df[['cluster','shifted']] = list(results_df['key'].str.split(","))
        paired_perf_df = results_df.groupby(['rep','seed','subtype','cluster'])[['shifted','f1-score']].apply(self.process)
        paired_perf_df = self.evaluate_shifted_unshifted(results_df, paired_perf_df)
        self.plot_cluster_performances(paired_perf_df, subdirectory)
    
#     def load_results(self, results_file):
#         self.results_df = pd.read_csv(self.data_path + "/" + results_file, sep="|")
#         self.results_df['cluster'] = ''
#         self.results_df['shifted'] = ''
#         self.results_df[['cluster','shifted']] = list(self.results_df['key'].str.split(","))
#         self.paired_perf_df = self.results_df.groupby(['rep','seed','subtype','cluster'])[['shifted','f1-score']].apply(self.process)
#         self.evaluate_shifted_unshifted()
        
#     def load_data(self, data_file, expressions_file, targets_file, sample_sample_graph, encoding_file, data_test_labels_file, num_to_target_file):
#         sel_data = self.paired_perf_df.set_index(['subtype','cluster','rep']).loc[[('Her2','2',1),('Her2','2',2)]].reset_index()
#         self.cg = cg.Data(
#             self.data_path + "/" + data_file,
#             self.data_path + "/" + targets_file,
#             self.data_path + "/" + sample_sample_graph,
#             self.data_path + "/" + encoding_file,
#             self.data_path + "/" + data_test_labels_file,
#             self.data_path + "/" + num_to_target_file,
#             self.index_path)
#         self.genes_df = pd.read_csv(self.data_path + "/" + expressions_file,index_col=0)
    
#     def evaluate_shifted_unshifted(self):
#         self.paired_perf_df.name = "f1-score (unshifted-shifted)"
#         self.paired_perf_df = self.paired_perf_df.reset_index()
#         shifted_f1 = self.results_df.set_index('shifted').loc['True',['rep','seed','subtype','cluster','f1-score']]
#         shifted_f1.columns = list(shifted_f1.columns[:-1]) + ['f1-score (shifted)']
#         self.paired_perf_df = self.paired_perf_df.set_index(['rep','seed','subtype','cluster']).join(shifted_f1.set_index(['rep','seed','subtype','cluster'])).reset_index()
#         unshifted_f1 = self.results_df.set_index('shifted').loc['False',['rep','seed','subtype','cluster','f1-score']]
#         unshifted_f1.columns = list(unshifted_f1.columns[:-1]) + ['f1-score (unshifted)']
#         self.paired_perf_df = self.paired_perf_df.set_index(['rep','seed','subtype','cluster']).join(unshifted_f1.set_index(['rep','seed','subtype','cluster'])).reset_index()
        
#     def generate_knowledge_graph(self, knowledge, neighbors=5, algorithm='ball_tree', mode='distance'):
#         for gene in knowledge:
#             if gene not in list(self.genes_df.columns):
#                 raise IndexError("Gene " + str(gene) + " does not exist in the dataset.")
#         self.knowledge = knowledge
#         hX,hy,hA = self.cg.helper_ret
#         X = self.genes_df[self.knowledge]
#         yher2 = hy[hy=='Her2']
#         Xher2 = X.loc[yher2.index]
#         scaler = MinMaxScaler()
#         Xher3 = scaler.fit_transform(Xher2)
#         n_neighbors = 10
#         nbrs = NearestNeighbors(n_neighbors=neighbors, algorithm=algorithm).fit(Xher3)
#         self.A = nbrs.kneighbors_graph(Xher3, mode=mode)
#         self.A = pd.DataFrame(self.A.toarray(),index=Xher2.index,columns=Xher2.index)
    
#     def sel_top(self, group_df,by,ascending=False):
#         return group_df.sort_values(by=by,ascending=ascending).iloc[0]
        
#     def calculate_and_rank_unshifted(self, num_samples=5):
#         sel_data = self.paired_perf_df.set_index('subtype').loc['Her2'].reset_index()
#         f = sel_data['f1-score (shifted)']
#         r = sel_data['f1-score (unshifted)']
#         sel_data['avg f1-score'] = ((f+r)/2).values # 2/(1/f+1/r)
#         sel_data_by_unshifted = sel_data.groupby(['cluster','rep']).apply(lambda group_df: self.sel_top(group_df,'f1-score (unshifted)'))
#         sel_data_by_unshifted['avg f1-score - f1-score (unshifted)'] = sel_data_by_unshifted['avg f1-score']-sel_data_by_unshifted['f1-score (unshifted)']
#         self.selected = sel_data_by_unshifted.sort_values(by='avg f1-score - f1-score (unshifted)').iloc[:5]
#         sel_data2 = sel_data.set_index(['cluster','rep']).loc[self.selected.index].reset_index()
#         unshifted_ranks = self.get_ranks(self.selected,sel_data2,'f1-score (unshifted)')
#         unshifted_corr_results = pd.DataFrame(self.calc_scores(unshifted_ranks,self.selected,'f1-score (unshifted)') ).T
#         results = cg.compute_score(sel_data2,self.A,self.cg)
#         results = results.reset_index().set_index(['cluster','rep'])
#         # STOPPED AT: 
#         #import copy
#         # by = 'combined score and f1-score (unshifted)'
#         # copy_unshifted_ranks = copy.deepcopy(unshifted_ranks)
#         # for cluster,rep in unshifted_ranks.keys():

        
        


#     # TODO: This needs to be printed out to a downloadeable file. 
#     def plot_cluster_performances(self):
#         source = self.paired_perf_df.set_index('subtype').loc['Her2']
#         source = source.set_index(['rep','cluster'])
#         for index in source.index.unique():
#             f = source.loc[index,'f1-score (shifted)']
#             r = source.loc[index,'f1-score (unshifted)']
#             source.loc[index,'rank'] = ss.rankdata(-(f+r)/2)
#             source.loc[index,'avg f1-score'] = ((f+r)/2).values #2*(f*r)/(f+r)
#         source = source.reset_index()
#         top_k = 5
#         mask = source['rank'] > top_k
#         source.loc[:,'rank'] = source.loc[:,'rank'].astype(int).astype(str)
#         source.loc[mask,'rank'] = f'>{top_k}'
#         chart = alt.Chart(source).transform_calculate(
#             Seed="datum.seed"# + '-' + datum.seed"
#         ).mark_point(filled=True).encode(
#             x=alt.X('f1-score (unshifted):Q',scale=alt.Scale(domain=[0, 1])),
#             y=alt.Y('f1-score (shifted):Q'),
#             size='avg f1-score:Q',
#             column='cluster',
#             row='rep',
#             color='rank:N',
#             tooltip=['Seed:Q','rank','avg f1-score','f1-score (unshifted)','f1-score (shifted)']
#         )
#         chart.resolve_scale(x='independent')
#         chart.save('ranking_results.pdf',scale_factor=2.0)
#         # chart.save('ranking_results.html')
    
            
#     # can we imporve the ranking?
#     def get_ranks(self, selected,sel_data2,by,n = 5,ascending=False):
#         ranks = {}
#         for ix in selected.index:
#             ranks[ix] = sel_data2.set_index(['cluster','rep']).loc[ix].sort_values(by=by,ascending=ascending).iloc[:n]
#         return ranks

#     def calc_scores(self, ranks,selected,by,true_col='avg f1-score',verbose=False):
#         scores = {}
#         for ix in selected.index:
#             scores[ix]={}
#             x1 = ranks[ix][true_col]
#             x2 = ranks[ix][by]
#             tau, p_value = stats.kendalltau(x1, x2)
#             r, p_value = stats.pearsonr(x1, x2)
#             if verbose:
#                 x1 = ranks[ix][['seed',true_col]]
#                 x2 = ranks[ix][['seed',by]]
#                 # print(ix)
#                 # print(x1.sort_values(by=true_col))
#                 # print(x2.sort_values(by=by))
#                 # print(tau)
#             scores[ix]['tau'] = tau
#             scores[ix]['r'] = r
#         return scores


    
#     def shifted_stress_test_analysis(self):
#         self.paired_perf_df = self.results.groupby(['rep','seed','subtype','cluster'])[['shifted','f1-score']].apply(process)
#         self.evaluate_shifted_unshifted()
        