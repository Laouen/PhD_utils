import unittest
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from tqdm import tqdm

from PhD_utils.machine_learning import sfs

class TestSQS(unittest.TestCase):

    def test_correctly_ordered_features(self):

        y = np.array([0] * 80 + [1] * 100 + [0] * 20)

        # informative features
        # X1 separates the two classes 80% of the time
        X1 = np.array([1] * 100 + [0] * 100).reshape(-1, 1)
        # X1 + X2 separates the two classes 90% of the time
        X2 = np.array([0] * 80 + [1] * 20 + [0] * 100).reshape(-1, 1)
        # X1 + X2 + X3 separates the two classes 100% of the time
        X3 = np.array([0] * 180 + [1] * 20).reshape(-1, 1)

        # add noise features
        X4 = np.random.rand(200, 1)
        X5 = np.random.rand(200, 1)
        X6 = np.random.rand(200, 1)

        X = np.concatenate([X1, X2, X3, X4, X5, X6], axis=1)

        for p1 in tqdm(range(X.shape[1]), leave=False):
            for p2 in tqdm(range(X.shape[1]), leave=False):
                for p3 in tqdm(range(X.shape[1]), leave=False):
                    if p1 == p2 or p1 == p3 or p2 == p3:
                        continue

                    with self.subTest(f'p1={p1}, p2={p2}, p3={p3}'):
                        f_pos = [p1, p2, p3]
                        n_pos = [x for x in range(X.shape[1]) if x not in f_pos]

                        idxs = np.array(range(X.shape[1])) 
                
                        idxs[f_pos] = [0,1,2]
                        idxs[n_pos] = [3,4,5]

                        sfs_clf = sfs.SequentialFeatureSelector(
                            estimator=DecisionTreeClassifier()
                        ).fit(X[:,idxs], y) # train with the features in the order set order

                        # First three features should be the informative ones
                        self.assertEqual(sfs_clf.selected_features_[0], p1)
                        self.assertIn(sfs_clf.selected_features_[1], [p2, p3])
                        self.assertIn(sfs_clf.selected_features_[2], [p2, p3])

                        # assert score increments until 3
                        self.assertLess(sfs_clf.scores_[0], sfs_clf.scores_[1])
                        self.assertLessEqual(sfs_clf.scores_[1], sfs_clf.scores_[2])
                        # assert scores worsen after 3
                        self.assertGreaterEqual(sfs_clf.scores_[2], sfs_clf.scores_[3])
                        self.assertGreaterEqual(sfs_clf.scores_[3], sfs_clf.scores_[4])
                        self.assertGreaterEqual(sfs_clf.scores_[4], sfs_clf.scores_[5])


if __name__ == '__main__':
    unittest.main()