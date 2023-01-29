from .analysis import Analysis
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

class KmeansAnalysis(Analysis):
    """
    Kmeans analysis

    Attributes
    ----------
    (See Analysis class)
    num_clusters : int
    """

    def __init__(self, X, y, analysis_name, output_dir, transformers=None, num_clusters=3):
        super().__init__(X, y, analysis_name, output_dir, transformers)
        self._num_clusters = num_clusters

    def run(self):
        """
        Creates k-means plots of the data

        Returns:
        ----------
            Xy: Joined cols of X and y along w/ kmeans cluster labels
        """
        self._run_transformers()

        Xy = self._X.join(self._y)
        
        # Plot k-means inertia
        ks = range(1, 10)
        inertias = []
        for k in ks:
            model = KMeans(n_clusters=k)
            model.fit(self._X)
            inertias.append(model.inertia_)
        plt.plot(ks, inertias, '-o', color='black')
        plt.xlabel('number of clusters, k')
        plt.ylabel('inertia')
        plt.xticks(ks)
        plt.savefig(self._output_dir + 'kmeans_inertia.png')
        plt.clf()
        
        # Plot user-specified k-means
        model = KMeans(n_clusters=self._num_clusters)
        model.fit(self._X)
        Xy['kmean-label'] = model.labels_
        kplt = Xy['kmean-label'].value_counts().loc[list(range(self._num_clusters))].plot.barh()
        kplt.figure.savefig(self._output_dir + 'kmeans.png')
        
        return Xy



