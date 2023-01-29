from typing import List, Union
import pandas as pd
import altair as alt
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
from sklearn import svm, ensemble, neural_network 
import random
import os
from sklearn.metrics import silhouette_score
from .transformers.analysis_transformer import AnalysisTransformer
from .transformers.null_transformer import NullTransformer

class GenerateByKMeans:
    """Class to generate datasets in ways to test for underspecification"""
    def __init__(self,range_n_clusters=[2,3],random_state=42):
        self._range_n_clusters = range_n_clusters
        self._random_state=random_state
        
    def fit(self,X,y):
        self._clusterer_dict = {}
        self._cluster_labels_dict = {}
        self._silhouette_avgs_dict = {} 
        for n_clusters in self._range_n_clusters:
            # Initialize the clusterer with n_clusters value and a random generator
            # seed for reproducibility.
            self._clusterer_dict[n_clusters] = KMeans(n_clusters=n_clusters, random_state=self._random_state)
            self._cluster_labels_dict[n_clusters] = self._clusterer_dict[n_clusters].fit_predict(X)

            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters
            self._silhouette_avgs_dict[n_clusters] = silhouette_score(X, self._cluster_labels_dict[n_clusters])
            
        self._best_n_clusters = pd.Series(self._silhouette_avgs_dict).idxmax()
        return self
    
    def cluster_labels(self):
        return self._cluster_labels_dict[self._best_n_clusters]

class PUMP:
    """
    Main PUMP analysis class

    Attributes
    ----------
    X : pandas.DataFrame
        The input data to be analyzed. Must be a pandas DataFrame with the same
        number of rows as y.
    y : pandas.DataFrame
        The output data to be analyzed. Must be a pandas DataFrame with the same
        number of rows as X and a single column.
    output_dir : str
        The directory to save the output files to. Defaults to "data".
        If the directory does not exist, it will be created.
        NOTE: You should define a different output directory for each analysis.

    TODO: If we want one streamlined analysis, this should probably take a List[AnalysisTransformer]
    see the analyze_dataset method for an example. Alternatively, each instance of PUMP could also take
    one AnalysisTransformer and multiple PUMP instances could be used to analyze different datasets/
    transformations. Not sure which idea is best, but in any case transformers should probably be defined
    here.

    TODO: Reconsider what methods should actually be public.

    TODO: Some logic should be extracted out of here...
    """
    def __init__(self, X, y, output_dir="data"):
        if output_dir[-1] != "/":
            output_dir = output_dir + "/" 
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # ---- Initialize Data ----
        if X.shape[0] != y.shape[0]:
            raise IndexError("X and y sets must have an equal number of samples.")
        self._X = X
        self._y = y
        self._Xy = X.join(y).dropna()
        self.output_dir = output_dir

    def analyze(self, cluster_focus=None, num_clusters=3, num_shifted_sets=50, random_state=0):
        # TODO: This should not take cluster_focus as an arg if we want this to be general...
        # I suggest taking a List[TransformerMixin] as an arg instead. All transformers are analyzed then.
        raise NotImplementedError("Full analysis not yet implemented.")
        
    def pca_analysis(self, transformed_x, transformed_y, analysis_name):
        """
        Creates PCA plots of the data in output_dir/{analysis_name}/analysis

        Args:
        ----------
            transformed_x (pandas.DataFrame): The transformed X data
            transformed_y (pandas.DataFrame): The transformed y data
            analysis_name (str): The name of the analysis

        Returns:
        ----------
            pca_components (pandas.DataFrame): The PCA components
        """
        # Standardize the data to have a mean of ~0 and a variance of 1
        # TODO: This should probably be done beforehand (i.e. pass in an already standardized X)
        X_std = StandardScaler().fit_transform(transformed_x)
        pca = PCA(n_components=20)
        principalComponents = pca.fit_transform(X_std)
        
        # PCA Variance Plot
        features = range(pca.n_components_)
        plt.bar(features, pca.explained_variance_ratio_, color='black')
        plt.xlabel('PCA features')
        plt.ylabel('variance %')
        plt.xticks(features)
        pca_components = pd.DataFrame(principalComponents,index=transformed_y.index)
        plt.title("Variance Drop Off after 0,1,2,...")
        plt.savefig(self.output_dir + f'{analysis_name}/analysis/pca_bar.png')
        plt.clf()
        
        # PCA Scatter Plot
        plt.scatter(pca_components[0], pca_components[1], alpha=.1, color='black')
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        plt.savefig(self.output_dir + f'{analysis_name}/analysis/pca_scatter.png')
        plt.clf()
        
        return pca_components
    
    def kmeans_analysis(self, transformed_x, transformed_y, num_clusters, analysis_name):
        """
        Creates k-means plots of the data in output_dir/{analysis_name}/analysis

        Args:
        ----------
            transformed_x (pandas.DataFrame): The transformed X data
            transformed_y (pandas.DataFrame): The transformed y data
            num_clusters (int): The number of clusters to use
            analysis_name (str): The name of the analysis

        Returns:
        ----------
            Xy: Joined cols of X and y along w/ kmeans cluster labels
        """

        Xy = transformed_x.join(transformed_y)
        
        # Plot k-means inertia
        ks = range(1, 10)
        inertias = []
        for k in ks:
            model = KMeans(n_clusters=k)
            model.fit(transformed_x)
            inertias.append(model.inertia_)
        plt.plot(ks, inertias, '-o', color='black')
        plt.xlabel('number of clusters, k')
        plt.ylabel('inertia')
        plt.xticks(ks)
        plt.savefig(self.output_dir + f'{analysis_name}/analysis/kmeans_inertia.png')
        plt.clf()
        
        # Plot user-specified k-means
        model = KMeans(n_clusters=num_clusters)
        model.fit(transformed_x)
        Xy['kmean-label'] = model.labels_
        kplt = Xy['kmean-label'].value_counts().loc[list(range(num_clusters))].plot.barh()
        kplt.figure.savefig(self.output_dir + f'{analysis_name}/analysis/kmeans.png')
        
        return Xy
        
    def plot_cluster_analysis(self, Xy, pca_components, analysis_name):
        """
        Plots cluster analysis in output_dir/{analysis_name}/analysis/clusters.html
        
        Args:
        ----------
            Xy (pandas.DataFrame): Joined cols of X and y along w/ kmeans cluster labels
            pca_components (pandas.DataFrame): The PCA components
            analysis_name (str): The name of the analysis

        TODO: Change Xy to something more descriptive
        """
        data = pca_components.copy()
        data.columns = [f"PC{str(c + 1)}" for c in data.columns]
        data['cluster'] = Xy['kmean-label']
        chart = alt.Chart(data).mark_circle(size=60).encode(
            x="PC1",
            y="PC2",
            color='cluster:N',
        )
        chart.save(self.output_dir + f'{analysis_name}/analysis/clusters.html')
        
    def analyze_dataset(self, transformers: List[AnalysisTransformer]=None, num_clusters:int=3):
        """
        Analyzes the dataset using the specified methods and saves the results to output_dir/{analysis_name}/analysis

        Parameters
        ----------
        transformers : List[AnalysisTransformer] (default=None; will use [NullTransformer] w/ name "analysis_0")
            The transformers to use for analysis (each transformer passed will create a different analysis)
        num_clusters : int (default=3)
            The number of clusters to use for k-means clustering.

        Returns
        -------
        kmeans_transformed_results : Dict[str, pandas.DataFrame]
            A dictionary of the kmeans transformed results for each analysis.
            TODO: This probably shouldn't be returned at all... seems odd
        """
        # Store kmeans transformed results for each analysis
        kmeans_transformed_results = {}
        # Create a null transformer if none are specified
        transformers = transformers or [NullTransformer("analysis_0")]
        for transformer in transformers:
            if not os.path.exists(self.output_dir + f'{transformer.get_name()}/analysis'):
                os.makedirs(self.output_dir + f'{transformer.get_name()}/analysis')
            X, y = self._X.copy(), self._y.copy()
            X, y = transformer.fit_transform(X, y)
            print(f"-----[ {transformer.get_name()} ]-----")
            print("Running PCA Analysis ...")
            pca_components = self.pca_analysis(X, y, transformer.get_name())
            print("Running K-means Analysis ...")
            Xy = self.kmeans_analysis(X, y, num_clusters, transformer.get_name())
            print("Writing Cluster Analysis ...")
            self.plot_cluster_analysis(Xy, pca_components, transformer.get_name())
            kmeans_transformed_results[transformer.get_name()] = Xy
            print(f"-----[ {transformer.get_name()} COMPLETE ]-----")
        print(f"-----[ COMPLETE ]-----\nCheck /{self.output_dir} for the results.")
        return kmeans_transformed_results
    
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
    
    def create_shifted_datasets(self, Xy_filtered, cluster_focus, num_clusters=3, shift_type='clustering', subdirectory='indices', num_shifted_sets=50, split_ratio=[0.7, 0.1, 0.2]):
        """
        Creates shifted datasets.

        TODO: Better documentation. Not 100% sure about all the args...
        TODO: List[AnalysisTransformer] as stated previously.
        """
        if not os.path.exists(self.output_dir + subdirectory):
            os.makedirs(self.output_dir + subdirectory)
        sets_by_group = {}
        df_filtered = Xy_filtered
        df = self._X.join(self._y).dropna()
        if shift_type != 'clustering':
            raise ValueError('Non-clustering shifting methods are not yet available. Please select \'clustering\' for shift type')
        print("Creating shifted sets ...")
        # TODO: This could be problematic... float rounding errors?
        if sum(split_ratio) != 1:
            raise ValueError("Split Ratio must total up to 1 exactly")

        for cluster in range(num_clusters): # clusters selected from colors above
            shifted_set = list(df_filtered[df_filtered['kmean-label'] == cluster].index.values)

            for shifted in [True,False]:
                sets = []
                sets_by_group[cluster, shifted] = sets
                for i in range(num_shifted_sets):
                    test_size = round(len(df)*split_ratio[2])
                    val_size = round(len(df)*split_ratio[1])

                    df_not_class = df[df[self._y.name] != cluster_focus]

                    random.seed(i)
                    test_list = list(
                        df_not_class.index[
                            random.sample(
                                range(len(df_not_class)),
                                test_size - len(shifted_set),
                            )
                        ]
                    )

                    df_dropped = df_filtered.drop(shifted_set)
                    if len(shifted_set) > len(df_dropped):
                        shifted_set = random.sample(shifted_set, len(df_dropped))
                    reg_set = list(
                        df_dropped.index[
                            random.sample(range(len(df_dropped)), len(shifted_set))
                        ]
                    )
                    remaining_set = df.drop(shifted_set).drop(reg_set).drop(test_list).index
                    test_list += shifted_set if shifted else reg_set
                    val_list = remaining_set[random.sample(range(len(remaining_set)), val_size)]
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
