# segmentation.py
import numpy as np
from sklearn.cluster import KMeans


class BaseSegmentation:
    """Base class for segmentation methods."""
    def fit(self, X, *args, **kwargs):
        raise NotImplementedError
    def assign(self, X):
        raise NotImplementedError


class KMeansSeg(BaseSegmentation):
    """K-Means based segmentation."""
    def __init__(self, n_segments, random_state=0):
        self.k = n_segments
        self.random_state = random_state
        self.model = None

    def fit(self, X, *args, **kwargs):
        self.model = KMeans(
            n_clusters=self.k,
            n_init=10,
            random_state=self.random_state
        ).fit(X)
        return self

    def assign(self, X):
        if self.model is None:
            raise RuntimeError("KMeansSeg: call fit() first")
        return self.model.predict(X)


# segmentation.py
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


class BaseSegmentation:
    """Base class for segmentation methods."""
    def fit(self, X, *args, **kwargs):
        raise NotImplementedError

    def assign(self, X):
        raise NotImplementedError


class KMeansSeg(BaseSegmentation):
    """K-Means based segmentation."""
    def __init__(self, n_segments, random_state=0):
        self.k = n_segments
        self.random_state = random_state
        self.model = None

    def fit(self, X, *args, **kwargs):
        self.model = KMeans(
            n_clusters=self.k,
            n_init=10,
            random_state=self.random_state
        ).fit(X)
        return self

    def assign(self, X):
        if self.model is None:
            raise RuntimeError("KMeansSeg: call fit() first")
        return self.model.predict(X)


class GMMSeg(BaseSegmentation):
    """Gaussian Mixture Model based segmentation."""
    def __init__(self, n_segments, covariance_type="full", random_state=0):
        self.k = n_segments
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.model = None

    def fit(self, X, *args, **kwargs):
        self.model = GaussianMixture(
            n_components=self.k,
            covariance_type=self.covariance_type,
            random_state=self.random_state
        )
        self.model.fit(X)
        return self

    def assign(self, X):
        if self.model is None:
            raise RuntimeError("GMMSeg: call fit() first")
        return self.model.predict(X)

    def bic(self, X):
        """Convenience wrapper for BIC on given data."""
        if self.model is None:
            raise RuntimeError("GMMSeg: call fit() before bic()")
        return self.model.bic(X)
