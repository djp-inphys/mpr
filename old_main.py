from config_submit import config as config_submit
from process import process, ky

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

import sklearn.linear_model as lm

# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def plot3( feats, clsvec, title):
    cancer = feats[clsvec == 1, :]
    benign = feats[clsvec == 0, :]
    # plot out first 3 components
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(cancer[:,0], cancer[:,1], cancer[:,2], 'r.')
    ax.plot(benign[:,0], benign[:,1], benign[:,2], 'g.')
    ax.set_title( title )

    plt.show()


def logloss( preds, clsvec ):
    nomeas      = clsvec.shape[0]
    flt_limit   = config_submit['float_limit']

    tot_lgls = 0
    for unk in range(nomeas):
        y     = clsvec[unk]
        p     = preds[unk]
        yhat  = max( min(p, 1.0 - flt_limit), flt_limit )
        term1 = y*np.log( yhat )
        term2 = ( 1-y )*np.log( 1-yhat )
        lgls  =  -(term1+term2)/nomeas
        tot_lgls += lgls

    return tot_lgls

def ky_data( descs, cls ):
    try:
        pca     = PCA(n_components=config_submit[ 'n_components' ]).fit(descs)
        kl_scrs = pca.fit_transform( descs )
        # separate out all cancers and benign
        cancer    = kl_scrs[cls==1,:]
        benign    = kl_scrs[cls==0,:]
        # plot out first 3 components
        plot3( kl_scrs, cls, "kl_scores" )
        # calculate ky scores
        cls = np.concatenate( (cls[cls==1], cls[cls==0]) )
        ky_coeffs, ky_scrs = ky.ky(np.concatenate((cancer, benign)), cls)

        # plot out first 3 components
        plot3(ky_scrs, cls, "ky_scores")

    except Exception as e:
        print(e + "\n")


    return ky_scrs, cls

if __name__ == '__main__':
    # read data
    filename = config_submit[ 'datapath' ]
    filename += "/feats.csv"  # these are all the features from no 2 solution
    descs  = np.genfromtxt(filename, delimiter='\t')
    # extract class label - assumes finsl column
    clsvec = descs[:, -1]
    #
    # selidx = np.loadtxt( "selvar.csv", delimiter=",")
    # selidx = selidx.astype(int)
    ky_scrs, clsvec = ky_data(descs, clsvec)
    for stage in range(0,10):
        if stage == 0:
            res = np.loadtxt("res4.csv", delimiter = ",")
        else:
            pca   = PCA(n_components=5).fit(descs)
            feats = pca.fit_transform(descs)
            res, stages = process(feats, clsvec)
        #
        outfname = "res" + str(stage) + ".csv"
        np.savetxt( outfname, res, delimiter = ",", fmt = "%10.8f" )
        plot3( res, clsvec, "selected results" )

        lnrg = lm.LinearRegression()
        lnrg.fit( res, clsvec.reshape(0-1,1) )
        linreg_pred = lnrg.predict(res)

        print( logloss(linreg_pred, clsvec) )
        # replace next gen of descriptors
        descs = res
