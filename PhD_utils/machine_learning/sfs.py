from typing import List, Optional, Union

from sklearn.base import clone, BaseEstimator
from sklearn.model_selection import cross_val_score, BaseCrossValidator, StratifiedKFold
from sklearn.pipeline import Pipeline

import numpy as np

from tqdm import tqdm

class SubFeatureSelector:

    def __init__(self, features_idxs: List[int]):
        self.features_idxs = features_idxs

    def fit(self, X: np.ndarray, y: np.ndarray):
        return self
    
    def transform(self, X: np.ndarray):
        return X[:, self.features_idxs]

class SequentialFeatureSelector:

    def __init__(self, estimator: BaseEstimator, n_features_to_select:int=10,
                 scoring:Optional[str]=None, cv:Union[int, BaseCrossValidator]=5,
                 n_jobs:Optional[int]=-1):

        self.estimator = clone(estimator)
        self.scoring = scoring
        self.n_features_to_select = n_features_to_select
        self.cv = cv
        self.n_jobs = n_jobs

        if isinstance(cv, int):
            self.cv = StratifiedKFold(n_splits=cv, shuffle=True)
            
    
    def fit(self, X: np.ndarray, y: np.ndarray,
            groups: Optional[np.ndarray]=None, verbose:bool=False):

        self.scores_ = []
        self.selected_features_ = []
        self.best_estimator_ = None
        self.best_score_ = -np.inf
        self.best_features_ = []

        pbar = tqdm(
            range(self.n_features_to_select),
            leave=False,
            disable=not verbose
        )

        for _ in pbar:
            current_scores = []
            for feature in set(range(X.shape[1])) - set(self.selected_features_):
                X_selected = X[:, self.selected_features_ + [feature]]
                current_scores.append(cross_val_score(
                    self.estimator, X_selected, y,
                    groups=groups,
                    scoring=self.scoring,
                    cv=self.cv, n_jobs=self.n_jobs
                    ).mean()
                )

            self.selected_features_.append(np.argmax(current_scores))

            new_score = np.max(current_scores)
            if new_score >= self.best_score_:
                self.best_score_ = new_score
                self.best_features_ = self.selected_features_.copy()
                self.best_estimator_ = Pipeline([
                    ('fs', SubFeatureSelector(self.best_features_)),
                    ('clf', clone(self.estimator))
                ]).fit(X, y)

            self.scores_.append(new_score)

        return self

    def transform(self, X: np.ndarray):
        return X[:, self.selected_features_]