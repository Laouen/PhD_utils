from typing import Optional, Union
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import BaseCrossValidator


def plot_ROC(y_trues: np.ndarray, y_probas: np.ndarray,
            label: str,
            ax: Optional[plt.Axes],
            color: str = 'blue',
            alpha: float = 1.0):

    ax = ax or plt.gca()

    assert len(y_probas.shape) == 1 or (len(y_probas.shape) == 2 and y_probas.shape[1] == 2), 'y_probas must be a vector of the possibe class or a matrix with 2 columns for the probas of each class' 
    probas = y_probas[:, 1] if len(y_probas.shape) > 1 else y_probas

    fpr, tpr, t = roc_curve(y_trues, probas)
    # AUC computed on the ROC fpr and tpr returns a
    # ROC_AUC score as using roc_auc_score function
    roc_auc = auc(fpr, tpr)
    ax.plot(
        fpr, tpr, color=color,
        label=f'{label} (AUC = {roc_auc:0.2f})',
        lw=2, alpha=alpha
    )

    return roc_auc

def plot_CV_ROC(clf: BaseEstimator, X: np.ndarray, y: np.ndarray,
                target: Optional[str]=None,
                cv: Union[int,BaseCrossValidator]=5,
                ax: Optional[plt.Axes]=None,
                title: str='ROC Curve'):

    if target is not None:
        y_target = (y == target).astype(int)
    else:
        y_target = y

    if isinstance(cv, int):
        cv = StratifiedKFold(n_splits=cv, shuffle=True)
    
    ax = ax or plt.gca()

    ax.plot([0,1],[0,1], linestyle = '--', lw = 2, color = 'black')

    y_trues = []
    y_probas = []
    for i, (train_idx, test_idx) in enumerate(cv.split(X,y)):
        x_train, x_test, y_train, y_test = X[train_idx], X[test_idx], y_target[train_idx], y_target[test_idx]

        probas = clf.fit(x_train,y_train).predict_proba(x_test)

        y_trues.append(y_test)
        y_probas.append(probas)

        plot_ROC(y_test, probas, label=f'Fold {i}', ax=ax, alpha=0.3)

    y_trues = np.concatenate(y_trues)
    y_probas = np.concatenate(y_probas)
    auc_score = plot_ROC(y_trues, y_probas, label=f'Total', ax=ax)

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc='lower right')

    return auc_score