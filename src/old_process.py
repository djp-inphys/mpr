from config_submit import config as config_submit

import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")

from scipy import linalg
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn import svm

from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool
from functools import partial

# #############################################################################
# Classifier map
clsfrs = {"cls0": "LinearDiscriminantAnalysis()",
          "cls1": "QuadraticDiscriminantAnalysis()",
          # svms
          "cls2": "svm.SVC( kernel=\"linear\", C = 1.0, shrinking=True, probability=True)",
          "cls3": "svm.SVC(kernel=\"poly\", C = 1.0, shrinking=True, probability=True)",
          "cls4": "svm.SVC(kernel=\"rbf\", C = 1.0, shrinking=True, probability=True)",
          "cls5": "svm.SVC(kernel=\"sigmoid\", C = 1.0, shrinking=True, probability=True)",
          # Create the knn model.
          "cls6": "KNeighborsClassifier(n_neighbors=1)",
          "cls7": "KNeighborsClassifier(n_neighbors=3)",
          "cls8": "KNeighborsClassifier(n_neighbors=5)",
          "cls9": "KNeighborsClassifier(n_neighbors=7)"}

# global stages - stores all the information from training - used to "compile" the run-time system
# stages = []
def sel_train(dataset, cls, sel_clsfrs = None):
    if (sel_clsfrs != None):
        clsfrs_avail = sel_clsfrs
    else:
        clsfrs_avail = range( len(clsfrs) )

    mdls = list()
    for c_idx in clsfrs_avail:
        key = "cls" + str(c_idx)
        clsfr = eval(clsfrs[key])

        mdls.append(clsfr.fit(dataset, cls))

    return mdls

# #############################################################################
# generate results - call on a leave one out basis
#
def train(dataset, cls):
    mdls = list()
    for c_idx in range( len(clsfrs) ):
        key = "cls" + str(c_idx)
        clsfr = eval(clsfrs[key])

        mdls.append(clsfr.fit(dataset, cls))

    return mdls


def classifiers(models, unk_vec):
    resvec = [];
    for clsno in range(len(models)):
        probs = models[clsno].predict_proba(unk_vec)
        resvec.append(probs[0, 0])

    return resvec

def leave_one_out(unk_idx, dataset, clsvec, results):  # dataset is a numpy n_array
    # train classfiers with a the dataset with the current unknown index removed
    print("leave one out classification", unk_idx)
    dta_copy = np.delete(dataset, unk_idx, 0)
    cls = np.delete(clsvec, unk_idx)

    # train classifiers
    models = train(dta_copy, cls)

    # get classification results for the unknown index
    unk_vec = np.matrix(dataset[unk_idx, :])
    resvec = classifiers(models, unk_vec)

    results[unk_idx,:] = resvec

# ############################################################################
#  process
#  train classifiers in the classifier vector on a leave one out basis
#  once the classifiers have been trained then apply them to classify the unknown index
#
def process( dataset, clsvec ):
    # global stages
    try:
        num_unks, nofeats = dataset.shape
        results = np.zeros((num_unks,len(clsfrs)))

        models  = train(dataset, clsvec)
        print( models )

        pool = ThreadPool( config_submit["worker_pool_size"] )
        partial_loo = partial( leave_one_out, dataset=dataset, clsvec=clsvec, results=results )

        N = len( clsvec )
        # N = 5
        _ = pool.map( partial_loo, range(N) )

        pool.close()
        pool.join()

        # select most suitable classifiers
        # selidx = call_beale( results, clsvec )
        # # train all seleccted models in all data
        # models =  train( dataset, clsvec, selidx )
        # stages.append( models )
        #
    except Exception as e:
        print(e)
    # resturn stage results
    # return results[:,selidx]
    return results

#
# def process( dataset, clsvec ):
#     try:
#         num_unks, nofeats = dataset.shape
#         res = np.zeros((num_unks,len(clsfrs)))
#
#         for unk_idx in range(num_unks):
#             res[unk_idx,:] = leave_one_out(unk_idx, dataset, clsvec)
#
#
#     except Exception as e:
#         print(e)
#
#     return res
#
#
#     unk_idx = 0
#     res = np.zeros( (dataset.shape[0], len(clsfrs)) )
#     for idx, each in enumerate( os.listdir(".") ):
#         if each.endswith(".npy") and each.startswith(str(unk_idx)):
#             resvec = np.load(each)
#             res[unk_idx,:]  = resvec
#             unk_idx         += 1


# # ############################################################################
# #  process
# #  train classifiers in the classifier vector on a leave one out basis
# #  once the classifiers have been trained then apply them to classify the unknown index
# #
# def leave_one_out( dataset, clsvec, unk_idx ): # dataset is a numpy n_array
#     nomeas, nofeats = dataset.shape
#     results  = np.zeros( (nomeas, len(clsfrs)) )
#     # step through data removing
#     corr_count = 0
#     for unk_idx in range(nomeas):
#         correct = False
#         # train classfiers with a the dataset with the current unknown index removed
#         dta_copy = np.delete( dataset, unk_idx, 0 )
#         cls      = np.delete( clsvec, unk_idx )
#         models   = train( dta_copy, cls )
#
#         # get classification results for the unknown index
#         unk_vec = np.matrix( dataset[unk_idx, :] )
#         resvec  = classifiers( models, unk_vec )
#
#         results[unk_idx, :] = resvec
#         mnres = np.mean( resvec )
#         if ((mnres > 0.5) and (clsvec[unk_idx]==1)):
#             corr_count+=1
#             correct = True
#         elif ((mnres <= 0.5) and (clsvec[unk_idx] == 0)):
#             corr_count+=1
#             correct = True
#
#         print("leave-one-out classification ", unk_idx, mnres )
#
#     return results
#
#
# pool = Pool(config_submit["worker_pool_size"])
# partial_loo = partial(leave_one_out, dataset=dataset, clsvec=clsvec)
#
# N = len(clsvec)
# _ = pool.map(partial_loo, range(N))
#
# pool.close()
# pool.join()
#
# unk_idx = 0
# res = np.zeros((dataset.shape[0], len(clsfrs)))
# for idx, each in enumerate(os.listdir(".")):
#     if each.endswith(".npy") and each.startswith(str(unk_idx)):
#         resvec = np.load(each)
#         res[unk_idx, :] = resvec
#         unk_idx += 1
#
