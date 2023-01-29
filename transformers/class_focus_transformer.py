from sklearn.base import BaseEstimator, TransformerMixin

class ClassFocusTransformer(BaseEstimator, TransformerMixin):
    """Transformer designed to focus on a single class"""
    def __init__(self, cluster_focus=None):
        self._cluster_focus = cluster_focus
    
    def fit(self, X, y):
        if self._cluster_focus not in y.unique():
            raise ValueError(
                f"Cluster focus {str(self._cluster_focus)} must be a class in y."
            )
        return self
    
    def transform(self, X, y):
        X = X.loc[y == self._cluster_focus]
        y = y.loc[y == self._cluster_focus]
        return X,y

    def fit_transform(self, X, y):
        self.fit(X,y)
        return self.transform(X,y)