from typing import List, Optional, Union

from sklearn.base import clone, BaseEstimator
from sklearn.model_selection import cross_val_score, BaseCrossValidator, StratifiedKFold, StratifiedGroupKFold
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

    def __init__(self, estimator: BaseEstimator, n_features_to_select:int=None,
                 scoring:Optional[str]=None, cv:Union[int, BaseCrossValidator]=5,
                 n_jobs:Optional[int]=-1):

        self.estimator = clone(estimator)
        self.scoring = scoring
        self.n_features_to_select = n_features_to_select
        self.cv = cv
        self.n_jobs = n_jobs
            
    
    def fit(self, X: np.ndarray, y: np.ndarray,
            groups: Optional[np.ndarray]=None, verbose:bool=True):

        if isinstance(self.cv, int):
            if groups is not None:
                cv = StratifiedGroupKFold(n_splits=self.cv, shuffle=True)
            else:
                cv = StratifiedKFold(n_splits=self.cv, shuffle=True)
        else:
            cv = self.cv
        
        if self.n_features_to_select is None:
            n_features_to_select = X.shape[1]

        self.scores_ = []
        self.selected_features_ = []
        self.best_estimator_ = None
        self.best_score_ = -np.inf
        self.best_features_ = []

        all_features = set(range(X.shape[1]))

        pbar = tqdm(
            range(n_features_to_select),
            leave=False,
            disable=not verbose
        )

        for _ in pbar:
            new_scores = []
            new_scores_features = []
            for feature in (all_features - set(self.selected_features_)):
                new_scores_features.append(feature)
                X_selected = X[:, self.selected_features_ + [feature]]
                new_scores.append(cross_val_score(
                    self.estimator, X_selected, y,
                    groups=groups,
                    scoring=self.scoring,
                    cv=cv, n_jobs=self.n_jobs
                    ).mean()
                )

            best_score = np.max(new_scores)
            best_score_idx = new_scores.index(best_score)
            self.selected_features_.append(new_scores_features[best_score_idx])
            self.scores_.append(best_score)

            if best_score >= self.best_score_:
                self.best_score_ = best_score
                self.best_features_ = self.selected_features_.copy()
                self.best_estimator_ = Pipeline([
                    ('fs', SubFeatureSelector(self.best_features_)),
                    ('clf', clone(self.estimator))
                ]).fit(X, y)

        return self

    def transform(self, X: np.ndarray):
        return X[:, self.selected_features_]