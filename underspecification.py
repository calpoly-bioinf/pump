import numpy as np
import pandas as pd
import altair as alt
import scipy.stats as ss
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from IPython.display import display, Markdown, Latex
import sys

import compare_graphs as cg

# sys.path.insert(0,f'{REPO_HOME}/underspecification')
# sys.path.insert(0,f'{REPO_HOME}/deeptype2')


class UnderspecificationAnalysis:
    
    def __init__(self, data_path, index_path):
        self.data_path = data_path
        self.index_path = index_path
        self.results_df = None
        self.genes_df = None
        self.knowledge = None
        self.paired_perf_df = None
        self.cg = None
        self.A = None
        self.selected = None
    
    def load_results(self, results_file):
        self.results_df = pd.read_csv(self.data_path + "/" + results_file, sep="|")
        self.results_df['cluster'] = ''
        self.results_df['shifted'] = ''
        self.results_df[['cluster','shifted']] = list(self.results_df['key'].str.split(","))
        self.paired_perf_df = self.results_df.groupby(['rep','seed','subtype','cluster'])[['shifted','f1-score']].apply(self.process)
        self.evaluate_shifted_unshifted
        
    def load_data(self, data_file, expressions_file, targets_file, sample_sample_graph, encoding_file, data_test_labels_file, num_to_target_file):
        print(type(self.paired_perf_df))
        self.paired_perf_df = pd.DataFrame(self.paired_perf_df)
        sel_data = (self.paired_perf_df).set_index(['subtype','cluster','rep']).loc[[('Her2','2',1),('Her2','2',2)]].reset_index()
        self.cg = cg.Data(
            self.data_path + "/" + data_file,
            self.data_path + "/" + targets_file,
            self.data_path + "/" + sample_sample_graph,
            self.data_path + "/" + encoding_file,
            self.data_path + "/" + data_test_labels_file,
            self.data_path + "/" + num_to_target_file,
            self.index_path)
        self.genes_df = pd.read_csv(self.data_path + "/" + expressions_file,index_col=0)
    
    def process(self, df):
        df = df.set_index('shifted')
        unshifted = df.loc['False','f1-score']
        shifted = df.loc['True','f1-score']
        return unshifted-shifted
    
    def evaluate_shifted_unshifted(self):
        self.paired_perf_df.name = "f1-score (unshifted-shifted)"
        self.paired_perf_df = self.paired_perf_df.reset_index()
        shifted_f1 = self.results_df.set_index('shifted').loc['True',['rep','seed','subtype','cluster','f1-score']]
        shifted_f1.columns = list(shifted_f1.columns[:-1]) + ['f1-score (shifted)']
        self.paired_perf_df = self.paired_perf_df.set_index(['rep','seed','subtype','cluster']).join(shifted_f1.set_index(['rep','seed','subtype','cluster'])).reset_index()
        unshifted_f1 = self.results_df.set_index('shifted').loc['False',['rep','seed','subtype','cluster','f1-score']]
        unshifted_f1.columns = list(unshifted_f1.columns[:-1]) + ['f1-score (unshifted)']
        self.paired_perf_df = self.paired_perf_df.set_index(['rep','seed','subtype','cluster']).join(unshifted_f1.set_index(['rep','seed','subtype','cluster'])).reset_index()
        
    def generate_knowledge_graph(self, knowledge, neighbors=5, algorithm='ball_tree', mode='distance'):
        for gene in knowledge:
            if gene not in self.genes_df.columns:
                raise IndexError("Gene " + str(gene) + " does not exist in the dataset.")
        self.knowledge = knowledge
        hX,hy,hA = self.cg.helper_ret
        X = self.genes_df[knowledge_genes]
        yher2 = hy[hy=='Her2']
        Xher2 = X.loc[yher2.index]
        scaler = MinMaxScaler()
        Xher3 = scaler.fit_transform(Xher2)
        n_neighbors = 10
        nbrs = NearestNeighbors(n_neighbors=neighbors, algorithm=algorithm).fit(Xher3)
        self.A = nbrs.kneighbors_graph(Xher3, mode=mode)
        self.A = pd.DataFrame(self.A.toarray(),index=Xher2.index,columns=Xher2.index)
    
    def sel_top(self, group_df,by,ascending=False):
        return group_df.sort_values(by=by,ascending=ascending).iloc[0]
        
    def calculate_and_rank_unshifted(self, num_samples=5):
        sel_data = df_paired_perf.set_index('subtype').loc['Her2'].reset_index()
        f = sel_data['f1-score (shifted)']
        r = sel_data['f1-score (unshifted)']
        sel_data['avg f1-score'] = ((f+r)/2).values # 2/(1/f+1/r)
        sel_data_by_unshifted = sel_data.groupby(['cluster','rep']).apply(lambda group_df: sel_top(group_df,'f1-score (unshifted)'))
        sel_data_by_unshifted['avg f1-score - f1-score (unshifted)'] = sel_data_by_unshifted['avg f1-score']-sel_data_by_unshifted['f1-score (unshifted)']
        self.selected = sel_data_by_unshifted.sort_values(by='avg f1-score - f1-score (unshifted)').iloc[:5]

    # TODO: This needs to be printed out to a downloadeable file. 
    def plot_cluster_performances(self):
        source = self.paired_perf_df.set_index('subtype').loc['Her2']
        source = source.set_index(['rep','cluster'])
        for index in source.index.unique():
            f = source.loc[index,'f1-score (shifted)']
            r = source.loc[index,'f1-score (unshifted)']
            source.loc[index,'rank'] = ss.rankdata(-(f+r)/2)
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
    
    def shifted_stress_test_analysis(self):
        self.paired_perf_df = self.results.groupby(['rep','seed','subtype','cluster'])[['shifted','f1-score']].apply(process)
        self.evaluate_shifted_unshifted()
        