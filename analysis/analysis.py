from typing import List
from sklearn.base import TransformerMixin
import pandas as pd
import os


class Analysis:
    """
    Base class for an analysis (i.e. PCA analysis, Kmeans, etc.)

    Attributes
    ----------
        X : pd.DataFrame
            X data
        y : pd.DataFrame # TODO: Probably should be a pd.Series
            y data
        analysis_name : str
            Name of analysis
        output_dir : str
            Output will be saved to output_dir/analysis_name
        transformers : List[TransformerMixin] = None
            List of transformers to apply to data. Transformers must have a
            fit, transform, and fit_transform method. Transformations are
            applied in the order they are given in the list.
        kwargs : dict
            Keyword arguments for analysis
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        analysis_name: str,
        output_dir: str="data/analysis",
        transformers: List[TransformerMixin] = None,
        **kwargs
    ):
        self._X = X
        self._y = y
        self._analysis_name = analysis_name
        self._output_dir = output_dir
        self._transformers = transformers or []
        self._kwargs = kwargs

        if not self._output_dir.endswith("/"):
            self._output_dir += "/"
        self._output_dir += f"{self._analysis_name}/"

        if not os.path.exists(self._output_dir):
            os.makedirs(self._output_dir)

    def _run_transformers(self):
        for transformer in self._transformers:
            self._X, self._y = transformer.fit_transform(self._X, self._y)

    @property
    def name(self):
        return self._analysis_name

    def run(self):
        raise NotImplementedError
