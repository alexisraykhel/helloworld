if __name__ == "__main__":
    import os
    import hashlib
    import atexit
    import warnings
    import pandas as pd
    import numpy as np
    import random
    import copy
    import hashlib
    import os.path

    from sklearn.base import clone
    from sklearn.pipeline import Pipeline, FeatureUnion
    from sklearn.svm import SVC, SVR, LinearSVC as lSVC, NuSVC
    from sklearn.ensemble import AdaBoostClassifier as ADA
    from sklearn.ensemble import BaggingClassifier as BAG
    from sklearn.ensemble import ExtraTreesClassifier as ETREES
    from sklearn.ensemble import GradientBoostingClassifier as GB
    from sklearn.ensemble import RandomForestClassifier as RF, RandomForestRegressor as RF_R
    from sklearn.ensemble import RandomTreesEmbedding as RTE
    from sklearn.neighbors import KNeighborsClassifier as KNN, KNeighborsRegressor as KNNR, NearestNeighbors
    from sklearn.neighbors import RadiusNeighborsClassifier as RNC
    from sklearn.linear_model import LogisticRegression as LR, LinearRegression as LinR
    from sklearn.linear_model import LogisticRegressionCV as LRCV
    from sklearn.linear_model import SGDClassifier as SGD, SGDRegressor as SGD_R
    import sklearn.linear_model as lin
    from sklearn.naive_bayes import BernoulliNB as BNB, MultinomialNB as MNB, GaussianNB as GNB
    from sklearn import tree
    from sklearn.tree import DecisionTreeClassifier as DT
    from sklearn.mixture import GMM
    from sklearn.random_projection import GaussianRandomProjection as GRP



    from sklearn import metrics
    from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
    from sklearn.neural_network import BernoulliRBM as BRBM
    import sklearn.manifold as manifold
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA, LinearDiscriminantAnalysis as LDA
    from sklearn.preprocessing import Normalizer, PolynomialFeatures, OneHotEncoder, FunctionTransformer, normalize, minmax_scale

    from sklearn.feature_selection import RFE, RFECV, SelectKBest, SelectFromModel
    from sklearn.decomposition import PCA, KernelPCA as KPCA, IncrementalPCA as IPCA, SparsePCA, RandomizedPCA as RPCA, TruncatedSVD as TSVD, FastICA as ICA, NMF
    from sklearn.cross_validation import KFold

    from sklearn.calibration import CalibratedClassifierCV as calibrator
    from sklearn.ensemble import VotingClassifier as vote
    import argparse
    import time

    from sklearn.cross_validation import train_test_split
    from sklearn.cross_validation import StratifiedKFold

    from sklearn.kernel_approximation import RBFSampler, Nystroem
    from sklearn.kernel_ridge import KernelRidge
    import sklearn.multiclass as multi
    import sklearn.cross_decomposition as cd





    import scipy.stats as stats
    import scipy.cluster.hierarchy as hcl
    import scipy.spatial.distance as dists
    import sklearn.cluster as cluster


    from itertools import combinations

    from utils import cpredict, Helper, MultiPredictor, DumbAverage, vert, upsample, stretch, pshuffle, correlations, pconcat, Consumer, mp_readfile, cclone, groupby_avg, cv, score, CorrClusterer, Smoother, scale, unique_rows, print_predictions, read_json
    import scipy.linalg as lin
    import scipy.linalg.interpolative as interpolative






def total_time(start):
    print("Time it took:"+str(round(time.time()-start)))


def check_nlp_improvement(fast=False):
    if fast:
        clf = RF(n_estimators=100, n_jobs=-1,criterion="entropy",max_features='auto',min_samples_split=5)
        folds = 5
    else:
        clf = RF(n_estimators=1000, n_jobs=-1, criterion="entropy", max_features=100, min_samples_split=5)
        folds = 10

    paramlist = [str(i) for i in clf.get_params().values()]
    parlist = str(np.sort(paramlist))+str(folds)
    h = hashlib.sha1()
    h.update(parlist.encode('utf-8'))
    sig = h.hexdigest()
    try:
        baseline = np.load("nlp_baseline_"+str(sig)+".npy")
    except Exception:
        print("Establishing baseline, this will run once")
        X_train, y_train, X_test, test_ids = read_json(do_descriptions=False)
        baseline = cv(X_train, y_train, None, MinMaxScaler(), clf, folds=folds, metric=metrics.log_loss, verbose=True)
        np.save("nlp_baseline_"+str(sig),baseline)

    print("Baseline:",baseline)

    X_train, y_train, X_test, test_ids = read_json(do_descriptions=True)
    print ("Checking performance, this may take several minutes")
    res = cv(X_train, y_train, None, MinMaxScaler(), clf, folds=folds, metric=metrics.log_loss, verbose=True)
    print("Result:",res)

    if res < baseline:
        print ("Improvement over baseline",str(baseline-res))
    else:
        print ("Performance worse than baseline by", str(res-baseline))



if __name__ == '__main__':
    start = time.time()
    atexit.register(total_time, start)
    np.set_printoptions(suppress=True)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    rs = None
    if rs is not None:
        np.random.seed(rs)

    check_nlp_improvement(fast=True)