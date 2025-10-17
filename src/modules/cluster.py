from typing import Union

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin


def cluster(
    preds: np.ndarray, confs: np.ndarray, algorithm: Union[ClusterMixin, BaseEstimator]
):
    labels = algorithm.fit_predict(preds)

    p, c = [], []
    for label in np.unique(labels):
        p.append(preds[labels == label].mean(axis=0))
        c.append(confs[labels == label].mean())

    p = np.stack(p)
    c = np.stack(c)
    return p, c
