from config_submit import config as config
from process import process, logloss, classifiers, ky

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import log_loss, roc_auc_score
import sklearn.linear_model as lm
import pickle as pk

# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def plot3( feats, clsvec, title):
    if hasattr(clsvec, "shape"): # classifier label list is not empty: hence, this is a training set
        bad = feats[clsvec == 1, :]
        good = feats[clsvec == 0, :]
        # plot out first 3 components
        fig = plt.figure(dpi=200)
        ax = fig.gca(projection='3d')
        ax.plot(bad[:,0], bad[:,1], bad[:,2], 'r.')
        ax.plot(good[:,0], good[:,1], good[:,2], 'g.')
        title += " train"
        ax.set_title( title )
    else: # conversely this must be a test set
        fig = plt.figure(dpi=200)
        ax = fig.gca(projection='3d')
        ax.plot(feats[:, 0], feats[:, 1], feats[:, 2], 'b.')
        title += " test"
        ax.set_title(title)

    plt.show()


def ky_data( descs, cls ):
    try:
        #pca     = PCA(n_components=20).fit(descs)
        pca = PCA().fit(descs)
        kl_scrs = pca.fit_transform( descs )
        # separate out all bads and good
        bad    = kl_scrs[cls==1,:]
        good    = kl_scrs[cls==0,:]
        # plot out first 3 components
        plot3( kl_scrs, cls, "kl_scores" )

        # calculate ky scores
        cls = np.concatenate( (cls[cls==1], cls[cls==0]) )
        ky_coeffs, ky_scrs = ky(np.concatenate((bad, good)), cls)

        # plot out first 3 components
        plot3(ky_scrs, cls, "ky_scores")

    except Exception as e:
        print(e)

    return ky_scrs, cls, ky_coeffs

if __name__ == '__main__':
    models = list()
    stages = list()

    if False:
        # read data
        descs_trndf  = pd.read_csv(config["datapath"] + "train.csv")
        print( descs_trndf.head() )

        # extract class label - assumes final column
        descs_trn  = descs_trndf.drop(['id','target'], axis=1)
        clsvec     = descs_trndf['target'].as_matrix()
        # read in test data
        filename_tst = config["datapath"] + "test.csv"
        descs_tstdf  = pd.read_csv( filename_tst )

        descs_tst    = descs_tstdf.drop(['id'], axis=1)
        # transform it to
        descs       = np.concatenate((descs_trn, descs_tst), axis=0)
        pca         = PCA().fit(descs)

        kl_scrs_trn = pca.fit_transform(descs_trn)
        kl_scrs_tst = pca.fit_transform(descs_tst)

        np.save( config["datapath"] + config["fname_classes"], clsvec)
        np.save( config["datapath"] + config["fname_train"], kl_scrs_trn)
        np.save( config["datapath"] + config["fname_test"], kl_scrs_tst)
    else: # read data
        kl_scrs_trn = np.load(config["datapath"] + config["fname_train"])
        kl_scrs_tst = np.load(config["datapath"] + config["fname_test"])
        clsvec      = np.load(config["datapath"] + config["fname_classes"])

    if config["run_type"] == "build":
        try:
            for stage in range(config['n_stages']):
                # representation stage
                if not (stage == 0):
                    descs = np.concatenate((descs_trn,descs_tst), axis = 0)
                    pca = PCA().fit(descs)
                    # save pca
                    stages.append(pca)
                    kl_scrs_trn = pca.fit_transform(descs_trn)
                    kl_scrs_tst = pca.fit_transform(descs_tst)

                # plot out first 3 components
                plot3(kl_scrs_trn, clsvec, "kl_scores")
                plot3(kl_scrs_tst, [], "kl_scores")

                # main process looop - train and classify
                trn_results, mdl, best_selidx = process(kl_scrs_trn[:, 0:config["n_dim"]], clsvec)

                c_res = classifiers(mdl, kl_scrs_tst[:, 0:config["n_dim"]])
                tst_results = np.array( c_res, order="C" ).transpose()

                # save model
                models.append(mdl)

                # save results
                # assess results
                lnrg = lm.LinearRegression()
                lnrg.fit(trn_results, clsvec.reshape(0 - 1, 1))
                linreg_pred = lnrg.predict(trn_results)
                print(roc_auc_score(clsvec, linreg_pred))

                # trial submission
                tst_pred = lnrg.predict(tst_results)
                submission = pd.read_csv(config["datapath"] + "sample_submission.csv")
                submission['target'] = tst_pred.reshape(len(tst_pred))
                submission.to_csv("submission" + str(stage) + ".csv", index=False)

                # replace next gen of descriptors
                descs_trn = trn_results
                descs_tst = tst_results
                if (stage == 5):
                    print(stage)

        except Exception as e:
            print(e)

        # execution model
    print(stages)

    # pk.dump("system", stages)
    pk.dump(stages, open("representation.pkl", "wb"))
    pk.dump(models, open("models.pkl", "wb"))

    if config["run_type"] == "run":
        try:
            if config['BUILD_or_RUN'] == 'run':
                # load models
                stages = pk.load(open("reps.pkl", "rb"))
                models = pk.load(open("mdls.pkl", "rb"))
                #
                # prediict and individual classification result
                #
                for stage, mdl in zip(stages, models):
                    if (type(stage[0]) == PCA):
                        print("pca")
                        scrs = stage[0].fit_transform(descs)
                        trn_results = classifiers(mdl[1], classifiers(mdl[0], scrs[:, 0:stage[1]]))
                    elif (type(stage[0]) == np.ndarray):
                        print("ky")
                        scrs = np.matmul(descs, stage[0])
                        trn_results = classifiers(mdl[1], classifiers(mdl[0], scrs[:, 0:stage[1]]))
                    elif (type(stage[0]) == lm.LinearRegression):
                        print("lr")
                        break
                    else:  # shouldnt get here
                        assert False

                    descs = trn_results  # for next iteration

                if type(stage[0]) == lm.LinearRegression:
                    probs = stage[0].predict(trn_results)
                    #write_submit(patient_ids, stage[0].predict(results), "final_submision.csv")
                else:
                    assert False

        except Exception as e:
            print(e)



    # ky_feats, clsvec = ky_data(descs, clsvec)
    #
    # pca = PCA(n_components=5).fit( ky_feats )
    # stages.append( pca )
    # feats  = stages[0].fit_transform(descs)
    # results, models = process( feats, clsvec )
    # pk_out = open( "sys.pkl", "wb" )
    # pk.dump( stages )
    # pk_out.close()
    #
    # pk_in = open( "sys,pkl", "rb" )
    # stgs = pk.load( pk_in )
    # pk_in.close()
    #
    # feats = stgs[0].fit_transfrom( descs )
    #
    # models = stgs[1]
    #
