from .analysis import Analysis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd

class PCAAnalysis(Analysis):
    """
    PCA analysis

    Attributes
    ----------
    (See Analysis class)
    """

    def run(self):
        """
        Creates PCA plots of the data

        Returns:
        ----------
            pca_components (pandas.DataFrame): The PCA components
        """
        self._run_transformers()

        # Standardize the data to have a mean of ~0 and a variance of 1
        X_std = StandardScaler().fit_transform(self._X)
        pca = PCA(n_components=20)
        principalComponents = pca.fit_transform(X_std)

        # PCA Variance Plot
        features = range(pca.n_components_)
        plt.bar(features, pca.explained_variance_ratio_, color='black')
        plt.xlabel('PCA features')
        plt.ylabel('variance %')
        plt.xticks(features)
        pca_components = pd.DataFrame(principalComponents,index=self._y.index)
        plt.title("Variance Drop Off after 0,1,2,...")
        plt.savefig(self._output_dir + 'pca_bar.png')
        plt.clf()

        # PCA Scatter Plot
        plt.scatter(pca_components[0], pca_components[1], alpha=.1, color='black')
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        plt.savefig(self._output_dir + 'pca_scatter.png')
        plt.clf()

        return pca_components



