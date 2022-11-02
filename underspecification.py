import numpy as np
import pandas as pd
import altair as alt
import scipy.stats as stats
import scipy.io
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.cluster import KMeans
from sklearn import svm, ensemble, neural_network 
from IPython.display import display, Markdown, Latex
import random
import sys
import os

import compare_graphs as cg


class PUMP:
    
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
        
    def analyze_dataset(self, X, y, methods=['clustering'], cluster_focus=None, subdirectory='analysis', num_clusters=3):
        for method in methods:
            if method not in ['clustering']:
                raise ValueError("Method " + method + " is not included in package methods")
        if not os.path.exists(self.output_dir + subdirectory):
            os.makedirs(self.output_dir + subdirectory)
        if cluster_focus is not None:
            if cluster_focus not in y.unique():
                raise ValueError("Cluster focus " + str(cluster_focus) + " must be a class in y.")
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
    
    def create_shifted_datasets(self, X, y, Xy_filtered, cluster_focus, num_clusters=3, shift_type='clustering', subdirectory='indices', num_shifted_sets=50, split_ratio=[0.7, 0.1, 0.2]):
        if not os.path.exists(self.output_dir + subdirectory):
            os.makedirs(self.output_dir + subdirectory)
        sets_by_group = {}
        df_filtered = Xy_filtered
        df = X.join(y).dropna()
        if shift_type != 'clustering':
            raise ValueError('Non-clustering shifting methods are not yet available. Please select \'clustering\' for shift type')
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
                    test_size = round(len(df)*split_ratio[2])
                    val_size = round(len(df)*split_ratio[1])
                    train_size = len(df) - test_size - val_size

                    df_not_class = df[df[y.name] != cluster_focus]

                    random.seed(i)
                    test_list = list(df_not_class.index[random.sample(range(0, len(df_not_class)), test_size-len(shifted_set))])

                    df_dropped = df_filtered.drop(shifted_set)
                    if len(shifted_set) > len(df_dropped):
                        shifted_set = random.sample(shifted_set, len(df_dropped))
                    reg_set = list(df_dropped.index[random.sample(range(0, len(df_dropped)), len(shifted_set))]) 
                    remaining_set = df.drop(shifted_set).drop(reg_set).drop(test_list).index
                    if shifted:
                        test_list += shifted_set
                    else:
                        test_list += reg_set 
                    val_list = remaining_set[random.sample(range(0, len(remaining_set)), val_size)]
                    train_list = list(set(remaining_set) - set(val_list))

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
    
    
    def calculate_performance_metrics(self, ytest, ypred, key, rep, seed):
        metrics_columns = ['key', 'rep', 'seed', 'class', 'precision', 'recall', 'f1-score', 'support']
        metrics_scores = pd.DataFrame(classification_report(ytest, ypred, target_names=ytest.unique(), output_dict=True)).transpose()
        metrics_scores.reset_index(inplace=True)
        metrics_scores = metrics_scores.rename(columns={'index':'class'})
        metrics_df = pd.DataFrame(columns=metrics_columns)
        for index, row in metrics_scores.iterrows():
            metrics_row = [key, rep, seed] + metrics_scores.loc[index].values.flatten().tolist()
            metrics_df = pd.concat([metrics_df, pd.DataFrame([metrics_row], columns=metrics_columns)])
        return metrics_df
    
    def svm_performance(self, Xtrain, ytrain, Xtest, ytest, key, rep, seed):
        clf = svm.SVC(kernel='linear', C=0.001, decision_function_shape='ovo', verbose=False).fit(Xtrain, ytrain)
        ypred = clf.predict(Xtest)
        return self.calculate_performance_metrics(ytest, ypred, key, rep, seed)
        
            
    def rf_performance(self, Xtrain, ytrain, Xtest, ytest, key, rep, seed):
        clf = ensemble.RandomForestClassifier(max_depth=3, random_state=rep).fit(Xtrain, ytrain)
        ypred = clf.predict(Xtest)
        return self.calculate_performance_metrics(ytest, ypred, key, rep, seed)
            
    def nn_performance(self, Xtrain, ytrain, Xtest, ytest, key, rep, seed):
        clf = neural_network.MLPClassifier(
            random_state=rep,
            early_stopping=True
        ).fit(Xtrain.values, ytrain)
        ypred = clf.predict(Xtest.values)
        return self.calculate_performance_metrics(ytest, ypred, key, rep, seed)
    
    def evaluate_shifted_sets(self, X, y, subdirectory='results', models=['mlp'], random_states=[0, 1, 2, 3, 4]):
        if not os.path.exists(self.output_dir + "results"):
            os.makedirs(self.output_dir + "results")
        svm_results = pd.DataFrame(columns=['key', 'rep', 'seed', 'class', 'precision', 'recall', 'f1-score', 'support'])
        rf_results = pd.DataFrame(columns=['key', 'rep', 'seed', 'class', 'precision', 'recall', 'f1-score', 'support'])
        nn_results = pd.DataFrame(columns=['key', 'rep', 'seed', 'class', 'precision', 'recall', 'f1-score', 'support'])
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
                    if 'svm' in models:
                        svm_results = pd.concat([svm_results, self.svm_performance(Xtrain, ytrain, sets['Xtest'], sets['ytest'], key, 0, seed)])
                    for random_state in random_states:
                        if 'rf' in models:
                            rf_results = pd.concat([rf_results, self.rf_performance(Xtrain, ytrain, sets['Xtest'], sets['ytest'], key, random_state, seed)])
                        if 'mlp' in models:
                            nn_results = pd.concat([nn_results, self.nn_performance(Xtrain, ytrain, sets['Xtest'], sets['ytest'], key, random_state, seed)])
        print("Writing out results ...")
        if 'svm' in models:
            svm_results.reset_index(drop=True)
            svm_results.to_csv(self.output_dir + f"{subdirectory}/svm_results.csv")
        if 'rf' in models:
            rf_results.reset_index(drop=True)
            rf_results.to_csv(self.output_dir + f"{subdirectory}/rf_results.csv")
        if 'mlp' in models:
            nn_results.reset_index(drop=True)
            nn_results.to_csv(self.output_dir + f"{subdirectory}/nn_results.csv")
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
        shifted_f1 = results_df.set_index('shifted').loc['True',['rep','seed','class','cluster','f1-score']]
        shifted_f1.columns = list(shifted_f1.columns[:-1]) + ['f1-score (shifted)']
        paired_perf_df = paired_perf_df.set_index(['rep','seed','class','cluster']).join(shifted_f1.set_index(['rep','seed','class','cluster'])).reset_index()
        unshifted_f1 = results_df.set_index('shifted').loc['False',['rep','seed','class','cluster','f1-score']]
        unshifted_f1.columns = list(unshifted_f1.columns[:-1]) + ['f1-score (unshifted)']
        paired_perf_df = paired_perf_df.set_index(['rep','seed','class','cluster']).join(unshifted_f1.set_index(['rep','seed','class','cluster'])).reset_index()
        return paired_perf_df
    
    def plot_cluster_performances(self, paired_perf_df, subdirectory):
        source = paired_perf_df.set_index('class')
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
        chart.save(self.output_dir + subdirectory + '/ranking_results.html')
        
    def plot_shifted_results(self, filename, subdirectory='results'):
        results_df = pd.read_csv(self.output_dir + subdirectory + "/" + filename)
        results_df['cluster'] = ''
        results_df['shifted'] = ''
        results_df[['cluster','shifted']] = list(results_df['key'].str.split(","))
        paired_perf_df = results_df.groupby(['rep','seed','class','cluster'])[['shifted','f1-score']].apply(self.process)
        paired_perf_df = self.evaluate_shifted_unshifted(results_df, paired_perf_df)
        self.plot_cluster_performances(paired_perf_df, subdirectory)
        
    def sel_top(self, group_df,by,ascending=False):
        return group_df.sort_values(by=by,ascending=ascending).iloc[0]
        
    def select_top_models(self, filename, subdirectory='results', num_models=3, class_filter=None):
        results_df = pd.read_csv(self.output_dir + subdirectory + "/" + filename)
        results_df['cluster'] = ''
        results_df['shifted'] = ''
        results_df[['cluster','shifted']] = list(results_df['key'].str.split(","))
        paired_perf_df = results_df.groupby(['rep','seed','class','cluster'])[['shifted','f1-score']].apply(self.process)
        paired_perf_df = self.evaluate_shifted_unshifted(results_df, paired_perf_df)
        if class_filter is not None:
            sel_data = paired_perf_df.set_index('class').loc[class_filter].reset_index()
        else:
            sel_data = paired_perf_df.set_index('class').reset_index()
        f = sel_data['f1-score (shifted)']
        r = sel_data['f1-score (unshifted)']
        sel_data['avg f1-score'] = ((f+r)/2).values # 2/(1/f+1/r)
        sel_data_by_unshifted = sel_data.groupby(['cluster','rep']).apply(lambda group_df: self.sel_top(group_df,'f1-score (unshifted)'))
        sel_data_by_unshifted['avg f1-score - f1-score (unshifted)'] = sel_data_by_unshifted['avg f1-score']-sel_data_by_unshifted['f1-score (unshifted)']
        sel_data_by_unshifted = sel_data_by_unshifted.sort_values(by='avg f1-score - f1-score (unshifted)')
        return sel_data_by_unshifted.drop(['f1-score (unshifted-shifted)', 'f1-score (shifted)', 'f1-score (unshifted)', 'avg f1-score'], axis=1).iloc[:num_models]
