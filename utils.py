from sklearn.base import clone
import numpy as np
from scipy import linalg
import sklearn.utils
import pandas as pd
import ctypes
import gc
from sklearn import metrics
from sklearn.cross_validation import StratifiedKFold, KFold
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, Normalizer, normalize, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression as LR
from sklearn.feature_extraction.text import CountVectorizer
import scipy.stats as stats
from scipy import sparse
import scipy.cluster.hierarchy as hcl
import multiprocessing as mp
from traceback import format_exc
import inspect, re
from text import NLP


def read_json(train_filename="../data/train.json", test_filename="../data/test.json", do_one_hot=False, do_descriptions=False):
    print ("Loading dataset...")
    train_df = pd.read_json(train_filename)
    test_df = pd.read_json(test_filename)
    print ("Raw dataset shapes:")
    print(train_df.shape)
    print(test_df.shape)

    print ("Raw features:")
    print list(train_df)

    features_to_use = ["bathrooms", "bedrooms", "latitude", "longitude", "price"]

    train_df["num_photos"] = train_df["photos"].apply(len)
    test_df["num_photos"] = test_df["photos"].apply(len)

    # count of "features" #
    train_df["num_features"] = train_df["features"].apply(len)
    test_df["num_features"] = test_df["features"].apply(len)

    # count of words present in description column #
    # train_df["num_description_words"] = train_df["description"].apply(lambda x: len(x.split(" ")))
    # test_df["num_description_words"] = test_df["description"].apply(lambda x: len(x.split(" ")))

    # convert the created column to datetime object so as to extract more features
    train_df["created"] = pd.to_datetime(train_df["created"])
    test_df["created"] = pd.to_datetime(test_df["created"])

    # Let us extract some features like year, month, day, hour from date columns #
    train_df["created_year"] = train_df["created"].dt.year
    test_df["created_year"] = test_df["created"].dt.year
    train_df["created_month"] = train_df["created"].dt.month
    test_df["created_month"] = test_df["created"].dt.month
    train_df["created_day"] = train_df["created"].dt.day
    test_df["created_day"] = test_df["created"].dt.day
    train_df["created_hour"] = train_df["created"].dt.hour
    test_df["created_hour"] = test_df["created"].dt.hour

    # adding all these new features to use list #
    features_to_use.extend(["num_photos", "num_features", "created_year", "created_month", "created_day", "listing_id", "created_hour"])

    X_train = train_df[features_to_use]
    X_test = test_df[features_to_use]


    categorical = ["display_address", "manager_id", "building_id", "street_address"]
    # categorical = ["building_id", "street_address"]
    lbl = LabelEncoder()
    cat_part_train = []
    cat_part_test = []
    for f in categorical:
        lbl.fit(list(train_df[f].values) + list(test_df[f].values))
        label_train = lbl.transform(list(train_df[f].values))
        label_test = lbl.transform(list(test_df[f].values))
        # print label_train.shape,label_test.shape
        cat_part_train.append(vert(label_train))
        cat_part_test.append(vert(label_test))

    cat_train = np.hstack(cat_part_train)
    cat_test = np.hstack(cat_part_test)
    if do_one_hot:
        ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
        cat_train = ohe.fit_transform(cat_train)
        cat_test = ohe.transform(cat_test)
    X_train = np.hstack((X_train,cat_train))
    X_test = np.hstack((X_test, cat_test))


            # features_to_use.append(f)

    train_features = train_df["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))
    test_features = test_df["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))
    tfidf = CountVectorizer(stop_words='english', max_features=200)
    train_features = tfidf.fit_transform(train_features).todense()
    test_features = tfidf.transform(test_features).todense()
    # train_X = sparse.hstack([train_df[features_to_use], tr_sparse]).tocsr()
    # test_X = sparse.hstack([test_df[features_to_use], te_sparse]).tocsr()
    X_train = np.hstack((X_train, train_features))
    X_test = np.hstack((X_test, test_features))

    if do_descriptions:
        nlp = NLP(train_df,test_df)
        train_desc, test_desc = nlp.convert_descriptions()
        assert len(train_desc) == len(X_train), "Length of training dataset is different from the length of provided descriptions list"
        assert len(test_desc) == len(X_test), "Length of test dataset is different from the length of provided descriptions list"
        X_train = np.hstack((X_train,train_desc))
        X_test = np.hstack((X_test, test_desc))



    target_num_map = {'high': 2, 'medium': 1, 'low': 0}
    train_y = np.array(train_df['interest_level'].apply(lambda x: target_num_map[x]))
    print("Loaded; dataset shapes:", X_train.shape, X_test.shape)
    return X_train, train_y, X_test, test_df["listing_id"].values




def cclone(clf):
    return clone(clf)






def stretch(input, offset=0, middle=None, off_center=False):
    if middle is not None:
        move = middle-input.mean()
        input = input + move
    if off_center is False:
        if offset is not None:
            input = (input - input.min())/(input.max()-input.min())*(1-2*offset) + offset
    else:
        #stdev = np.std(input)

        mean = np.mean(input)
        input = mean + (input - mean)*(1-offset*2)


        #dev = np.fabs(input-mean)/stdev

    return input

def scale(X_train,X_test,additional_sets=None, scaler=MinMaxScaler()):
    X_train = scaler.fit_transform(X_train)
    if X_test is not None:
        X_test = scaler.transform(X_test)

    rv = [X_train, X_test]

    # if X_live is not None:
    #     X_live = scaler.transform(X_live)
    if additional_sets is not None:
        for X_add in additional_sets:
            X_add = scaler.transform(X_add)
            rv.append(X_add)
    return rv

def vert(y):
    return np.expand_dims(y,axis=1)


def read_batch(startend, shared_x, shared_res,shape):
    start = startend[0]
    end = startend[1]
    #shared_res.append(np.asarray([np.array(shared_x[l_idx].split(",")).astype('float') for l_idx in range(start,end)]))
    res = np.frombuffer(shared_res, dtype='float32').reshape(shape)
    for l_idx in range(start, end):
        res[l_idx,:] = np.array(shared_x[l_idx].split(",")).astype('float')
    print ".",

def mp_readfile(input_file, n_jobs=24, batch_size=1000):
    print "Doing multiprocessing read of "+str(input_file)+", batch_size:"+str(batch_size)
    x = open(input_file, 'r')
    mgr = mp.Manager()
    shared_x = mgr.list()
    for l in x:
        shared_x.append(l)

    first_line = shared_x[0].split(",")
    shared_res = mp.Array(ctypes.c_float, len(shared_x) * len(first_line), lock=False)


    tasks = mp.JoinableQueue()
    batch_count = int(np.floor(len(shared_x) / batch_size)) + 1
    worker_count = min(n_jobs,batch_count)
    print str(worker_count)+" workers"
    num_jobs = 0
    for batch_idx in range(batch_count):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(shared_x))
        tasks.put([start, end])
        num_jobs += 1

    for w in range(worker_count):
        tasks.put(None)

    for w in range(worker_count):
        p = Consumer(tasks, read_batch, [shared_x, shared_res, (len(shared_x), len(first_line))])
        p.start()

    tasks.join()

    X = np.frombuffer(shared_res, dtype='float32').reshape(len(shared_x), len(first_line))
    del shared_x
    gc.collect()
    print ""
    return X

class Consumer(mp.Process):
    def __init__(self, task_queue, function, arguments, results_queue=None):
        mp.Process.__init__(self)
        self.task_queue = task_queue
        self.function = function
        self.arguments = arguments
        self.results_queue = results_queue

    def run(self):
        while True:
            idx = self.task_queue.get()
            if idx is None:
                self.task_queue.task_done()
                break
            try:
                res = self.function(idx, *self.arguments)
                if (self.results_queue != None):
                    self.results_queue.put(res)
            except Exception:
                #e = exc_info()
                print("\nFailed in mp on "+str(idx)+". Problem:\n"+format_exc())

            self.task_queue.task_done()
        return


class Helper(BaseEstimator, ClassifierMixin):
    def __init__(self, classifier, features=None, stretch_offset=None, verbose=False, platt=False, to_dense=False, keep_originals_when_transforming=True):
        self.classifier = classifier
        self.features = features
        self.stretch_offset = stretch_offset
        self.verbose = verbose
        self.platt = platt
        self.to_dense = to_dense
        self.keep_originals_when_transforming = keep_originals_when_transforming

    def fit(self,X,y):
        if self.features is not None:
            #print self.features
            X1n = X[:,self.features]
        else:
            X1n = np.copy(X)
        if self.to_dense and sparse.issparse(X1n):
            X1n = X1n.todense()
        self.classifier.fit(X1n,y)
        if self.platt:
            p_train = self.classifier.predict(X1n)
            self.platt_scaler = LR()
            self.platt_scaler.fit(p_train.reshape( -1, 1 ), y)
        return self

    def predict(self,X):
        if self.features is not None:
            X1n = X[:,self.features]
        else:
            X1n = np.copy(X)
        if self.to_dense and sparse.issparse(X1n):
            X1n = X1n.todense()
        res = self.classifier.predict(X1n)
        if self.platt:
            res = self.platt_scaler.predict_proba(res.reshape( -1, 1 ))[:,1]
        return res

    def predict_proba(self,X):
        if hasattr(self.classifier, 'predict_proba'):
            if self.features is not None:
                X1n = X[:,self.features]
            else:
                X1n = np.copy(X)
            if self.to_dense and sparse.issparse(X1n):
                X1n = X1n.todense()
            p = self.classifier.predict_proba(X1n)[:,1]
            if self.platt:
                p = self.platt_scaler.predict_proba(p.reshape( -1, 1 ))[:,1]
        else:
            p = self.predict(X)
            if np.max(p) > 1 or np.min(p) < 0:
                print("Stretching required",np.min(p),np.max(p))
                p = stretch(p)

        if self.stretch_offset is not None:
            p = stretch(p, offset=self.stretch_offset)
        p = vert(p)
        p = np.hstack((1-p,p))
        return p

    def transform(self,X):
        if self.features is not None:
            X1n = X[:,self.features]
        else:
            X1n = np.copy(X)
        if self.to_dense and sparse.issparse(X1n):
            X1n = X1n.todense()
        try:
            transformed = self.classifier.transform(X1n)
        except AttributeError:
            transformed = self.classifier.fit_transform(X1n)
        #print "transformed",transformed
        if self.keep_originals_when_transforming:
            res = np.hstack((X,transformed))
        else:
            res = transformed
        if self.verbose:
            print "transformed",res.shape,res
        return res

    def get_params(self, deep=True):
        return {"classifier": self.classifier, "features":self.features, "stretch_offset":self.stretch_offset, "verbose":self.verbose,
                "platt":self.platt, "to_dense":self.to_dense, "keep_originals_when_transforming":self.keep_originals_when_transforming}


    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self,parameter, value)
        return self




def mp_fit_wrapper(pair, shared_X, shared_y, sample_weight):
    clf_idx = pair[0]
    clf = pair[1]
    X_feature_count = int(len(shared_X)/len(shared_y))
    X = np.frombuffer(shared_X, dtype='float32').reshape((len(shared_y),X_feature_count))
    if "transform" in dir(clf): #if can't transform, no need to fit
        clf.fit(X, shared_y, sample_weight=sample_weight)
    print "<",
    return (clf_idx,clf)

def mp_transform_wrapper(pair, shared_X, X_Shape):
    clf_idx = pair[0]
    clf = pair[1]

    X = np.frombuffer(shared_X, dtype='float32').reshape(X_Shape)
    res = cpredict(clf,X)
    print ">",
    return (clf_idx,res)

class MultiPredictor(BaseEstimator):
    def __init__(self, classifiers, platt=False, keep=False, verbose=True, n_jobs=False, multi_transform=False):
        self.classifiers = classifiers
        self.platt=platt
        self.keep = keep
        self.verbose=verbose
        self.multi_transform = multi_transform
        if n_jobs is False:
            self.n_jobs = n_jobs
        else:
            self.n_jobs = min(n_jobs,len(classifiers))



    def fit(self,X,y, sample_weight = None):
        self.fit_classifiers=[]
        if self.n_jobs is False:
            for clf in self.classifiers:
                cclf = clone(clf)
                #cclf = Helper(cclf,platt=self.platt)
                if sample_weight is not None:
                    cclf.fit(X, y, sample_weight = sample_weight)
                else:
                    cclf.fit(X, y)
                self.fit_classifiers.append(cclf)
                if self.verbose:
                    print "<",
        else:
            print "n_jobs:"+str(self.n_jobs)
            tasks = mp.JoinableQueue()
            results = mp.Queue()
            for clf_idx in range(len(self.classifiers)):
                tasks.put((clf_idx,clone(self.classifiers[clf_idx])))
            for w in range(self.n_jobs):
                tasks.put(None)

            shared_X = mp.Array('f', np.reshape(X, (X.shape[0] * X.shape[1],)), lock=False)
            shared_y = mp.Array('f', y, lock=False)
            for w in range(self.n_jobs):
                p = Consumer(tasks, mp_fit_wrapper, [shared_X, shared_y], sample_weight, results_queue=results)
                p.start()

            tasks.join()
            self.fit_classifiers = [0]*len(self.classifiers)
            for clf in self.classifiers:
                pair = results.get()
                self.fit_classifiers[pair[0]] = pair[1]

        return self

    def transform(self,X):
        #predictions = np.full((len(self.fit_classifiers),X.shape[0]),-1.)
        predictions = [None]*len(self.classifiers)
        if self.n_jobs is False or self.multi_transform is False:
            for idx in range(len(self.fit_classifiers)):
                predictions[idx] = cpredict(self.fit_classifiers[idx],X)
                if self.verbose:
                    print ">",
        else:
            #rint "n_jobs:" + str(self.n_jobs)
            tasks = mp.JoinableQueue()
            results = mp.Queue()
            for clf_idx in range(len(self.fit_classifiers)):
                tasks.put((clf_idx, self.fit_classifiers[clf_idx], X.shape[1]))
            for w in range(self.n_jobs):
                tasks.put(None)
            shared_X = mp.Array('f', np.reshape(X, (X.shape[0] * X.shape[1],)), lock=False)
            for w in range(self.n_jobs):
                p = Consumer(tasks, mp_transform_wrapper, [shared_X, X.shape], results_queue=results)
                p.start()
            tasks.join()
            for clf in self.classifiers:
                pair = results.get()
                predictions[pair[0]] = pair[1]

        predictions = np.vstack(predictions)
        predictions = predictions.T
        if self.keep:
            return np.hstack((X,predictions))
        else:
            return predictions
        # print "Voter predictions",predictions
        # return np.average(predictions,weights=self.weights,axis=1)

    def fit_transform(self,X,y):
        self.fit(X,y)
        X = self.transform(X)
        return X

    def get_params(self, deep=True):
        return {"classifiers": self.classifiers, "platt":self.platt, "keep":self.keep, "verbose":self.verbose, "n_jobs":self.n_jobs, "multi_transform":self.multi_transform}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self,parameter, value)
        return self

def unique_rows(a):
    print ",,,"
    a = np.ascontiguousarray(a)
    print ",,,,"
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    print ",,,,,"
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

def remove_stupid_columns(X, verbose = True):
    if verbose:
        print "Removing stupid columns"
    X = X.T
    if verbose:
        print "T", X.shape, len(X)
    res = X
    print ","
    zerostd_count = 0
    progress_count = 0
    # for row in X:
    #     stdev = np.std(row)
    #     print stdev
    #     if stdev == 0:
    #         zerostd_count += 1
    #         continue
    #     res.append(row)
    #     progress_count += 1
    #     if verbose and progress_count % 10 == 0:
    #         print ".",
    # pre_unique_len = len(res)
    # print "pre_unique_len",pre_unique_len
    # res = unique_rows(np.array(res))
    pre_unique_len = len(res)
    print ",,"
    res = unique_rows(res)
    print ",,,,,,"
    dupes_count = pre_unique_len - len(res)
    if verbose:
        print ""
        print "Removed: all the same element columns:"+str(zerostd_count)
        print "Removed: duplicate columns:"+str(dupes_count)
    return np.array(res).T


class DumbAverage(BaseEstimator, ClassifierMixin):
    def __init__(self,weights=None, method='average'):
        self.weights = weights
        self.method = method

    def fit(self,X,y):
        return self

    def predict(self,X):
        if self.weights is not None and len(self.weights) == X.shape[1]:
            weights = self.weights
        else:
            print "DumbAverage: using uniform weights"
            weights = np.ones((X.shape[1],))
        if self.method == 'average':
            res = np.average(X,weights=weights,axis=1)


        if self.method == 'rankaverage':
            T = X.T
            ranks = np.empty(T.shape)
            for f in range(X.shape[1]):
                ranked = stats.rankdata(T[f],method='dense')
                ranks[f] = ranked
            ranks = ranks.T
            res = np.average(ranks,weights=weights,axis=1)
            res = stretch(res)

        return res


    def get_params(self, deep=True):
        return {"weights": self.weights, "method": self.method}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self,parameter, value)
        return self


def pshuffle(arrays):
    len = arrays[0].shape[0]
    rnd = np.arange(len)
    np.random.shuffle(rnd)
    retars = []
    for ar in arrays:
        retars.append(ar[rnd])
    return retars


def pconcat(arrays,axis=None):
    res = arrays[0]
    for idx in range(1,len(arrays)):
        if arrays[idx] is not None:
            res = np.concatenate((res,arrays[idx]),axis=axis)




def upsample(X, y, rank, mult=0, base=1, noise_std=0.015,verbose=False,features=21):
    if mult == 0 and base == 1:
        return X,y

    print "^",
    start_size = X.shape[0]
    multiplier = np.floor(rank * mult+base).astype('int32')
    nonzeros = multiplier[multiplier > 0].shape[0]
    tests = rank[rank < 0].shape[0]
    multiplier = np.where(rank >= 0,multiplier,1)

    X = np.repeat(X, multiplier, axis=0)
    if verbose:
        print "upsampling, start size",start_size-tests,"end size",X.shape[0]-tests,"test size",tests,"nonzeros",nonzeros
    #C = X
    if y is not None:
        y = np.repeat(y, multiplier)
        #C = np.hstack((C,vert(y)))


    if noise_std is not None and noise_std != 0:
        if features is None:
            features = X.shape[1]



        noise = np.random.normal(0,float(noise_std),X.shape[0]*features).reshape(X.shape[0],features)

        #noise = np.random.uniform(-noise_std,noise_std,C.shape[0]*features).reshape(C.shape[0],features)
        #noise_mult = np.repeat(np.clip(multiplier-1,0,100),multiplier)
        #noise_mult = np.repeat(np.clip(multiplier-1,0,1),multiplier)
        #noise_mult = np.repeat(np.clip(multiplier-1,0.1,1),multiplier)
        noise_mult = np.repeat(np.log(np.clip(multiplier,1,1000)),multiplier)
        noise = np.multiply(noise,vert(noise_mult))

        #print X
        X = np.hstack((X[:,:features] + noise,X[:,features:]))
        #X = np.hstack((np.clip((X[:,:features] + noise),0,1),X[:,features:]))
        #X = np.hstack((np.clip((X[:,:features] + noise),0,0.5),X[:,features:]))
        #print X


    #print ("Upsampled: "+str(start_size)+"->"+str(end_size))
    if y is not None:
        X,y=pshuffle([X,y])
        return X,y
    else:
        np.random.shuffle(X)
        return X






def cpredict(clf,X):
    if hasattr(clf, 'predict_proba'):
        # p = clf.predict_proba(X)[:,1]
        p = clf.predict_proba(X)
    else:
        p = clf.predict(X)
    #p = clf.predict(X)
    return p

def score(X_train,y_train,rank_train, X_test,y_test,rank_test,classifier, metric=metrics.roc_auc_score, force_binary=False,upsample_params=None, sample_weight_training=None, sample_weight_test=None, return_predictions=False, return_classifier=False, regression=False):
    clf = clone(classifier)
    if regression:
        metric = metrics.mean_absolute_error

    if rank_train is not None:
        if upsample_params is not None:
            X_train, y_train = upsample(X_train, y_train, rank_train, mult=upsample_params['mult'], base=upsample_params['base'], noise_std=upsample_params['noise_std'])
        else:
            X_train, y_train = upsample(X_train, y_train, rank_train)
        #X_test, y_test = upsample(X_test, y_test, rank_test)
    if sample_weight_training is not None:
        #print "Passing sample weights "+str(sample_weight_training)+" to classifier "+str(clf)
        clf.fit(X_train, y_train, sample_weight=sample_weight_training)
        #exit(0)
    else:
        clf.fit( X_train, y_train)
    # clf.fit(X_train, y_train)
    #print clf.feature_importances_
    if force_binary:
        p = clf.predict(X_test)
    else:
        p = cpredict(clf,X_test)
    if metric == metrics.accuracy_score:
        p = np.around(p).astype('int')
    if sample_weight_test is not None:
        #perc = np.percentile(sample_weight_test,95)
        #print perc, sample_weight_test, np.count_nonzero([sample_weight_test > perc[2]])/float(len(sample_weight_test))
        #exit(0)
        #p[sample_weight_test > perc] = 0.5
        pass
    # if sample_weight_test is not None:
    #     print sample_weight_test, np.average(sample_weight_test), np.median(sample_weight_test), np.min(sample_weight_test), np.max(sample_weight_test)
    #     exit(0)
    # if sample_weight_test is not None:
    #     avg_sample_weight_test = np.average(sample_weight_test)
    #     p = np.clip((p-0.5)*sample_weight_test/avg_sample_weight_test+0.5,0,1)
    if rank_train is not None:
        if upsample_params is not None:
            p, y_test = upsample(vert(p), y_test, rank_test, mult=upsample_params['mult'], base=upsample_params['base'], noise_std=None)
        else:
            p, y_test = upsample(vert(p), y_test, rank_test, noise_std=None)
    # print y_test
    # print p
    if sample_weight_test is not None:
        auc = metric(y_test, p, sample_weight=sample_weight_test)
    else:
        auc = metric(y_test,p)
    #exit(0)
    rv = auc
    if return_predictions or return_classifier:
        rv = [rv]
    if return_predictions:
        rv += [p]
    if return_classifier:
        rv += [clf]
    return rv

def get_xgb_imp(xgb, feat_names):
    from numpy import array
    imp_vals = xgb.booster().get_fscore()
    imp_dict = {feat_names[i]:float(imp_vals.get('f'+str(i),0.)) for i in range(len(feat_names))}
    total = array(imp_dict.values()).sum()
    return {k:v/total for k,v in imp_dict.items()}


def cv(X, y, rank, transformer, classifier, vector=None, folds=10, random_state=None, metric=metrics.roc_auc_score, verbose=False, force_binary = False,upsample_params=None, sample_weight=None, return_blend=False, return_classifiers = False):
    regression = False
    if len(np.unique(y)) > 100 or metric == metrics.mean_absolute_error:
        print "Regression task"
        kf = KFold(len(y), n_folds=folds, shuffle=True, random_state=random_state)
        regression = True
    else:
        kf = StratifiedKFold(y, n_folds=folds, shuffle=True, random_state=random_state)
    sumauc = 0
    rank_train = None
    rank_test = None
    sw_train = None
    sw_test = None
    blend = np.empty((X.shape[0],))
    classifiers = []
    for fold, (train_index, test_index) in enumerate(kf):
        X_train, X_test = X[train_index], X[test_index]
        #print(str(fold)+" "+str(test_index))
        y_train, y_test = y[train_index], y[test_index]
        if rank is not None:
            rank_train, rank_test = rank[train_index], rank[test_index]

        if sample_weight is not None:
            sw_train, sw_test = sample_weight[train_index], sample_weight[test_index]

        if transformer is not None:
            #print "App transformer"
            X_train = transformer.fit_transform(X_train)
            X_test = transformer.transform(X_test)
            # print "ds",X_train.shape, X_train[:,[0,21,42,63]]
            # print "dst", X_test.shape, X_test[:,[0,21,42,63]]
            # exit(0)



        if vector is not None:
            X_train = X_train[:,vector]
            X_test = X_test[:,vector]

        if return_blend:
            if return_classifiers:
                auc, p, clf = score(X_train, y_train, rank_train, X_test, y_test, rank_test, classifier, metric=metric, force_binary=force_binary, upsample_params=upsample_params, sample_weight_training=sw_train, sample_weight_test=sw_test, return_predictions=True, regression=regression, return_classifier=True)
                classifiers.append(clf)
            else:
                auc,p = score(X_train, y_train, rank_train, X_test, y_test, rank_test, classifier, metric=metric, force_binary=force_binary, upsample_params=upsample_params, sample_weight_training=sw_train, sample_weight_test=sw_test, return_predictions=True, regression=regression)
            blend[test_index] = p
        else:
            if return_classifiers:
                auc, clf = score(X_train,y_train,rank_train, X_test,y_test,rank_test, classifier, metric=metric, force_binary=force_binary,upsample_params=upsample_params, sample_weight_training=sw_train, sample_weight_test=sw_test, regression=regression, return_classifier=True)
                classifiers.append(clf)
            else:
                auc = score(X_train, y_train, rank_train, X_test, y_test, rank_test, classifier, metric=metric, force_binary=force_binary, upsample_params=upsample_params, sample_weight_training=sw_train, sample_weight_test=sw_test, regression=regression)
        sumauc += auc/folds
        if verbose:
            print ".",
    auc = sumauc
    if return_blend:
        if return_classifiers:
            return auc, blend, classifiers
        else:
            return auc, blend
    else:
        if return_classifiers:
            return auc, classifiers
        else:
            return auc


def correlations(X,y,ind_score_metric=metrics.log_loss,reference=None):
    if reference is None:
        reference = np.full(y.shape,1.)
    else:
        print "Reference specified in correlations", np.count_nonzero(reference), reference
    X = X[reference == 1]
    y = y[reference == 1]
    cov = np.corrcoef(X,y,rowvar=0)
    covsum = np.mean(np.fabs(cov[:-1,:-1]),axis=1)
    #print("Covariance matrix:"+str(cov))
    np.savetxt("covariances.csv", cov, fmt='%1.4f', delimiter=",")
    covtarg = cov[:-1,-1]

    print "covsum before scaling", covsum
    print "covtarg before scaling", covtarg

    if ind_score_metric is not None:
        score_ar = []
        for c in range(X.shape[1]):
            sc = ind_score_metric(y,X[:,c])
            score_ar.append(round(sc,6))
    print "individual feature scores", score_ar
    print "individual best:" +str(np.argmin(score_ar))+": "+str(np.min(score_ar))
    print "sorted :" + str(np.sort(score_ar))
    print "indexes:"+str(np.argsort(score_ar))



def groupby_avg(filename):
    data = pd.read_csv(filename)
    group = data.groupby(['t_id'], as_index=False)
    averages = group.aggregate(np.mean)
    print averages
    averages.to_csv('../predictions_avg.csv', float_format='%.5f', index=None)

class CorrClusterer(BaseEstimator):
    def __init__(self, group_size=3):
        self.group_size = group_size
        self.groups = None


    def fit(self,X,y=None):
        f_cnt = int(X.shape[1])
        Z = hcl.linkage(X.T, metric='euclidean', method='ward')
        # print Z
        clusters = []
        approved_clusters = []
        for idx in range(len(Z)):
            cluster = []
            raw_members = [Z[idx][0], Z[idx][1]]
            for m in raw_members:
                if m < f_cnt:
                    cluster.append(int(m))
                else:
                    cluster.extend(clusters[int(m)-f_cnt])
            clusters.append(cluster)
            if len(cluster) == self.group_size:
                approved_clusters.append(cluster)
        if len(approved_clusters) * self.group_size != f_cnt:
            print "Failed to cluster into groups of "+str(self.group_size)+", exiting"
            exit(1)
        self.groups = approved_clusters
        print approved_clusters
        return self

    def transform(self,X):
        res = np.empty((X.shape[0],len(self.groups)*3))
        idx = 0
        for group in self.groups:
            res[:,idx] = np.average(X[:,group],axis=1)
            res[:, idx+1] = np.min(X[:, group], axis=1)
            res[:, idx+2] = np.max(X[:, group], axis=1)
            idx = idx + 3
        return res

    def get_params(self, deep=True):
        return {"group_size": self.group_size}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self,parameter, value)
        return self










class Smoother(BaseEstimator):
    def __init__(self, threshold=0.1):
        self.threshold = threshold



    def fit(self,X,y=None):
        for i in range(X.shape[1]):
            uniq = np.unique(X[:, i])
            suniq = np.sort(uniq)
            diff = np.diff(suniq)
            sdiff = np.sort(diff)
            print i,np.min(suniq),np.max(suniq),np.average(suniq)," | ",np.min(sdiff),np.max(sdiff),np.average(sdiff),np.median(sdiff)#,np.percentile(sdiff,[10,20,30,40,50,60,70,80,90])
        exit(0)
        return self

    def transform(self,X):
        return X

    def get_params(self, deep=True):
        return {"threshold": self.threshold}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self,parameter, value)
        return self



def fix_predictions(ids, p):
    ids_adj = []
    p_adj = []
    max_id = np.max(ids)
    duplicates_count = 0
    missing_count = 0
    for i in range(len(ids)):
        id = ids[i]
        pr = p[i]
        if id not in ids_adj:
            ids_adj.append(id)
            p_adj.append(pr)
        else:
            duplicates_count +=1

    for id in range(1,max_id+1):
        if id not in ids_adj:
            ids_adj.append(id)
            p_adj.append(0.5)
            missing_count += 1
    if duplicates_count > 0 or missing_count > 0:
        print "FIXING PREDICTIONS. Duplicates:",duplicates_count," missing:",missing_count
    return np.array(ids_adj), np.array(p_adj)

def print_predictions(test_ids, p_test, filename="predictions.csv", additional_sets = None, extra_precision=False, skip_fix=False):
    test_ids = test_ids.astype('int')
    p_test = p_test[test_ids != -1]
    test_ids = (test_ids[test_ids != -1]).astype('int')
    if additional_sets is not None:
        for pair in additional_sets:
            ids = pair[0]
            preds = pair[1]
            test_ids = np.hstack((test_ids, ids))
            if preds is not None:
                p_test = np.hstack((p_test, preds))
            else:
                p_test = np.hstack((p_test, np.full(len(ids), 0.5)))

    # if train_ids is not None:
    #     test_ids = np.hstack((test_ids,train_ids))
    #     if train_preds is not None:
    #         p_test = np.hstack((p_test, train_preds))
    #     else:
    #         p_test = np.hstack((p_test,np.full(len(train_ids),0.5)))
    #
    # if live_ids is not None:
    #     test_ids = np.hstack((test_ids, live_ids))
    #     if live_preds is not None:
    #         p_test = np.hstack((p_test, live_preds))
    #     else:
    #         p_test = np.hstack((p_test, np.full(len(live_ids), 0.5)))
    if not skip_fix:
        test_ids, p_test = fix_predictions(test_ids,p_test)


    # pred = pd.DataFrame(test_ids, columns=['t_id'])
    # pred['probability'] = p_test
    # format = '%.5f'
    # if extra_precision:
    #     format = '%.17f'
    # pred.to_csv('../'+filename, columns=('t_id', 'probability'), float_format=format, index=None)

    f = open('../'+filename,'w')
    contents = ["t_id,probability\n"]
    for idx in range(len(test_ids)):
        p = p_test[idx]
        if (p == 0.5 or p == 0 or p == 1 or p == 0.4 or p == 0.6):
            contents.append(str(test_ids[idx])+","+str(p)+"\n")
        else:
            contents.append(str(test_ids[idx]) + "," + "{:.12f}".format(p) + "\n")
    f.writelines(contents)
    f.close()




def predict_on_blend(dataset, predictive_transformer, classifier, folds=5, random_state=None, shuffle=True, do_classifier_cv_folds=2, do_predict = True, save_l2=True):
    X_train, y_train, X_test, test_ids = dataset

    if shuffle:
        X_train, y_train = pshuffle([X_train,y_train])

    # if cheatless_split is not None:
    #     X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=cheatless_split, stratify=y_train)

    if predictive_transformer is not None:
        kf = StratifiedKFold(y_train, n_folds=folds, shuffle=False, random_state=random_state)
        print("Performing "+str(folds)+"-fold transformer CV on training set")
        y_blend = np.full((X_train.shape[0],), -1.)
        for fold, (train_index, test_index) in enumerate(kf):
            print "Fold "+str(fold),
            X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
            y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]


            predictive_transformer.fit( X_train_fold, y_train_fold )
            X_test_fold = predictive_transformer.transform( X_test_fold )

            if fold == 0:
                X_transformed_blend = np.full((X_train.shape[0], X_test_fold.shape[1]), -1.)
            X_transformed_blend[test_index] = X_test_fold
            y_blend[test_index] = y_test_fold

        if save_l2:
            np.savetxt("l2_train.csv", np.hstack((X_transformed_blend, vert(y_blend))), delimiter=",", fmt='%1.5f')
            print "\nFitting transformer on the whole train set"
            predictive_transformer.fit(X_train, y_train)
            X_test = predictive_transformer.transform(X_test)
            np.savetxt("l2_test.csv", np.hstack((vert(test_ids), X_test)), delimiter=",", fmt='%1.5f')
        # if cheatless_split is not None:
        #     X_val = transformer.transform(X_val)
        #     np.savetxt("l2_val.csv", np.hstack((X_val, vert(y_val))), delimiter=",", fmt='%1.5f')
    else:
        print "Transformer not provided, skipping transformer CV, assuming passed dataset is the blend"
        X_transformed_blend = X_train
        y_blend = y_train


    print "Train correlations"
    correlations(X_transformed_blend,y_blend)


    if do_classifier_cv_folds is not None:
        print "Performing "+str(do_classifier_cv_folds)+"-fold classfier CV on the blend"
        res = cv(X_transformed_blend,y_blend,None,None,classifier,folds=do_classifier_cv_folds,metric=metrics.log_loss)
        print "Classifier CV result: "+str(res)

    if do_predict:
        print "Fitting classifier on the blend"
        clf = cclone(classifier)
        clf.fit(X_transformed_blend, y_blend)
        p_test = cpredict(clf,X_test)


        pred = pd.DataFrame(test_ids,columns=['t_id'])
        pred['probability'] = p_test
        pred.to_csv( '../predictions.csv', columns = ( 't_id', 'probability' ), float_format='%.5f', index = None )

    return res