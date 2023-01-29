from .analysis import Analysis
from .kmeans_analysis import KmeansAnalysis
from .pca_analysis import PCAAnalysis
import altair as alt

class PUMPDefaultAnalysis(Analysis):
    """
    The default analysis for PUMP (PCA + Kmeans + Cluster plotting)
    """

    def __init__(self, X, y, analysis_name, output_dir, transformers=None, num_clusters=3):
        super().__init__(X, y, analysis_name, output_dir, transformers)
        self._num_clusters = num_clusters

    def _plot_cluster_analysis(self, Xy, pca_components):
        data = pca_components.copy()
        data.columns = [f"PC{str(c + 1)}" for c in data.columns]
        data['cluster'] = Xy['kmean-label']
        chart = alt.Chart(data).mark_circle(size=60).encode(
            x="PC1",
            y="PC2",
            color='cluster:N',
        )
        chart.save(f'{self._output_dir}{self._analysis_name}/clusters.html')

    def run(self):
        """
        Runs the default analysis
        """

        self._run_transformers()

        print("Running PCA Analysis ...")
        pca = PCAAnalysis(self._X, self._y, self._analysis_name, self._output_dir, self._transformers)
        pca_components = pca.run()

        print("Running Kmeans Analysis ...")
        kmeans = KmeansAnalysis(self._X, self._y, self._analysis_name, self._output_dir, self._transformers, self._num_clusters)
        kmeans_components = kmeans.run()

        print("Plotting Cluster Analysis ...")
        self._plot_cluster_analysis(kmeans_components, pca_components)

        return {
            'pca': pca_components,
            'kmeans': kmeans_components
        }

        

