from beale import beale
import numpy as np

import sklearn.linear_model as lm
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score, GridSearchCV, RepeatedStratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn import svm
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

from multiprocessing.dummy import Pool as ThreadPool
from functools import partial
import pickle as pk
import warnings
# warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
# warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
# #############################################################################
# Classifier map
from config_submit import config, clsfrs, clsfrs_params

# Not used - use the version from Sci-kit learn.
#def logloss( preds, clsvec ):
#     nomeas      = clsvec.shape[0]
#     flt_limit   = config['float_limit']
#
#     tot_lgls = 0
#     for unk in range(nomeas):
#         y     = clsvec[unk]
#         p     = preds[unk]
#         yhat  = max( min(p, 1.0 - flt_limit), flt_limit )
#         term1 = y*np.log( yhat )
#         term2 = ( 1-y )*np.log( 1-yhat )
#         lgls  =  -(term1+term2)/nomeas
#         tot_lgls += lgls
#
#     return tot_lgls

# interface to beale regession reselection
def call_beale( descs, clsvec, n_selvar=None ):
    #
    #  augement with classification vector
    #
    clsarr   = clsvec.reshape(0-1,1)
    cspace   = np.hstack((descs,clsarr))
    cr       = np.corrcoef(cspace, rowvar=False)
    if n_selvar is None:
        n_selvar = config['n_selvar']

    # selvar  = process.beale( cr, n_selvar )
    selvar  = beale(cr, n_selvar)

    idx      = np.array( range( descs.shape[1] ) )

    return idx[selvar == 1]



# #############################################################################
# generate results - call on a leave one out basis
#
def train(models, dataset, cls):
    try:
        trn_mdls = list()
        for mdl in models:
            trn_mdls.append(mdl.fit(dataset, cls))

        return trn_mdls
    except Exception as e:
        print(e)

#################
def classifiers(models, unk_vec):
    try:
        resvec = [];
        for clsno in range(len(models)):
            probs = models[clsno].predict_proba(unk_vec)
            resvec.append(probs[:, 0])

        return resvec
    except Exception as e:
        print(e)

def leave_one_out(unk_idx, models, dataset, clsvec, results):  # dataset is a numpy n_array
    # train classfiers with a the dataset with the current unknown index removed
    # print("leave one out classification", unk_idx)
    # print(unk_idx)
    try:
        print('.', end='', flush=True)
        dta_copy = np.delete(dataset, unk_idx, axis=0)
        cls      = np.delete(clsvec, unk_idx)

        # train classifiers
        trained_models = train(models=models, dataset=dta_copy, cls=cls )

        # get classification results for the unknown index
        unk_vec_m = np.matrix(dataset[unk_idx, :]) # sort matrix stuff out later
        resvec    = classifiers(models=models, unk_vec=unk_vec_m)

        results[unk_idx,:] = resvec
    except Exception as e:
        print(e)

# function to be mapped over
def calc_parallel( part, it ):
    pool = ThreadPool(config["worker_pool_size"])
    results = pool.map( part, it )
    pool.close()
    pool.join()

    return results
# ############################################################################
#  process
#  train classifiers in the classifier vector on a leave one out basis
#  once the classifiers have been trained then apply them to classify the unknown index
# #
# def process( dataset, clsvec ):
#     try:
#         num_unks, nofeats = dataset.shape
#         res = np.zeros((num_unks,len(clsfrs)))
#
#         for unk_idx in range(num_unks):
#             res[unk_idx,:] = leave_one_out( unk_idx, dataset, clsvec, res )
#
#
#     except Exception as e:
#         print(e)
#
#     return res

def process( dataset, clsvec ):
    try:
        models = list()
        nunks, nofeats = dataset.shape
        results = np.zeros((nunks,len(clsfrs)))

        # estimate optimal model parameters
        n_fold = 20
        folds = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=42)
        for key, value in clsfrs.items():
            mdl   = eval( value )
            param_grid = clsfrs_params[key]
            if bool(param_grid):
                grid_srch = GridSearchCV(mdl, param_grid=param_grid, cv=folds, scoring='roc_auc', n_jobs=-1)
                grid_srch.fit(dataset,clsvec)
                print('Best score: {}'.format(grid_srch.best_score_))
                print('Best parameters: {}'.format(grid_srch.best_params_))
                mdl.set_params( **grid_srch.best_params_ )

            models.append(mdl)

        #
        for unk_idx in range(nunks):
            print( unk_idx )
            leave_one_out( unk_idx, models, dataset=dataset, clsvec=clsvec, results=results )

        # select most suitable classifiers
        BLLE = float('inf') # best log loss error
        selidx = []
        for n_selvar in range(3,len(clsfrs)):
            selidx = call_beale( results, clsvec, n_selvar )

            lnrg = lm.LinearRegression()
            lnrg.fit(results[:,selidx], clsvec.reshape(0-1, 1))
            linreg_pred = lnrg.predict(results[:,selidx])

            lgls = log_loss( clsvec, linreg_pred )
            print( lgls )
            if lgls < BLLE: # save best
                best_selidx = selidx
                BLLE  = lgls

        # train all seleccted models on all data
        np_mdls = np.array( models )
        models = train( np_mdls[ best_selidx ], dataset, clsvec  )

    except Exception as e:
        print(e)

    return results[:,best_selidx], models, best_selidx
#
#

for unk_idx in range(num_unks):
    print( unk_idx )
    leave_one_out( unk_idx, dataset, clsvec, results )

for ndim in range(10,nofeats,10):
    print( ndim )
    leave_one_out(0, dataset, clsvec, results)

#
# #
# - seems to have a problem.
# some of the sklearn packages use multiprocessing
#
# pool = ThreadPool( config["worker_pool_size"] )
# part = partial(leave_one_out, dataset=dataset, clsvec=clsvec, results=results )
# rc   = pool.map( part, range(nunks) )
# pool.close()
# pool.join()

# save some stuff
# np.save( 'results.npy', results)
# np.savetxt( 'results.csv', results, fmt="%12.8f", delimiter=",")
