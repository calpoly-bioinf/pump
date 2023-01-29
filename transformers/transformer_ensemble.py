from sklearn.base import BaseEstimator, TransformerMixin

class TransformerEnsemble(BaseEstimator, TransformerMixin):
    """
    Class designed to sequence transformations

    Attributes
    ----------
    transformers : list
        List of transformers to apply to data. Transformers must have a
        fit, transform, and fit_transform method. Transformations are
        applied in the order they are given in the list.
    """
    def __init__(self, transformers=None):
        self._transformers = transformers

    def fit(self, X, y):
        for transformer in self._transformers:
            transformer.fit(X,y)
        return self

    def transform(self, X, y):
        for transformer in self._transformers:
            X,y = transformer.transform(X,y)
        return X,y

    def fit_transform(self, X, y):
        self.fit(X,y)
        return self.transform(X,y)