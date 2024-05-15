import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer

class TFMeanIDF(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.tfidf = TfidfVectorizer()
    
    def fit(self, X, y=None):
        self.tfidf.fit(X)
        return self
    
    def transform(self, X, y=None):
        tfidf_matrix = self.tfidf.transform(X)
        tf_mean = tfidf_matrix.mean(axis=0)
        return tfidf_matrix.multiply(tf_mean)
