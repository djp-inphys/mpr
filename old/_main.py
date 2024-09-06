from config_submit import config as config
from process import process, logloss, classifiers, ky

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

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

    if config[ "save" ]:
        # read data
        trn_filename = config[ "datapath" ] + "train.csv"
        trn_descs  = np.genfromtxt(trn_filename, delimiter=',') #descriminators

        # extract class label - assumes final column
        clsvec = trn_descs[:, 1]
        trn_descs  = trn_descs[:,2:-1] # exclude the class affiliation

        if config[ "kittler and young" ]:
            if hasattr( clsvec, "shape" ):
                ky_feats, clsvec, ky_coeffs = ky_data( trn_descs, clsvec )
                trn_descs = ky_feats
                np.save( config["datapath"]+"trn_descs.npy", trn_descs)
            else:
                exit(-1)

        # read in test data
        tst_filename = config["datapath"] + "test.csv"
        tst_descs    = np.genfromtxt( tst_filename, delimiter="," )
        tst_descs    = tst_descs[:,1:-1] # zap initial column
        # transform it to ky
        tst_descs    = np.matmul( tst_descs, ky_coeffs ) # these are features - called descriminators
        np.save( config["datapath"] + "tst_descs.npy", tst_descs)
    else: # read data
        trn_filename = config["datapath"] + "train.npy"
        trn_descs    = np.load(trn_filename)

        tst_filename = config["datapath"] + "test.npy"
        tst_descs    = np.load(trn_filename)

    if config["run_type"] == "build":
        try:
            for stage in range(config['n_stages']):
                # representation stage
                pca = PCA().fit(trn_descs)
                # save pca
                stages.append(pca)
                kl_scrs_train = pca.fit_transform(trn_descs)
                kl_scrs_test  = pca.fit_transform(tst_descs)

                # plot out first 3 components
                plot3(kl_scrs_train, clsvec, "kl_scores")
                plot3(kl_scrs_test, [], "kl_scores")

                # main process looop - train and classify
                results, mdl = process(kl_scrs_train[:, 0:config.dim], clsvec)

                # save model
                models.append(mdl)

                # save results
                outfname = "res" + str(stage) + ".csv"
                np.savetxt(outfname, results, delimiter=",", fmt="%10.8f")

                outfname = "res" + str(stage) + "_2.csv"
                np.savetxt(outfname, results_stage2, delimiter=",", fmt="%10.8f")

                # assess results
                lnrg = lm.LinearRegression()
                lnrg.fit(results, clsvec.reshape(0 - 1, 1))
                linreg_pred = lnrg.predict(results)

                print(logloss(linreg_pred, clsvec))
                # replace next gen of descriptors
                descs = results
                descs_stage2 = results_stage2
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
                        results = classifiers(mdl[1], classifiers(mdl[0], scrs[:, 0:stage[1]]))
                    elif (type(stage[0]) == np.ndarray):
                        print("ky")
                        scrs = np.matmul(descs, stage[0])
                        results = classifiers(mdl[1], classifiers(mdl[0], scrs[:, 0:stage[1]]))
                    elif (type(stage[0]) == lm.LinearRegression):
                        print("lr")
                        break
                    else:  # shouldnt get here
                        assert False

                    descs = results  # for next iteration

                if type(stage[0]) == lm.LinearRegression:
                    probs = stage[0].predict(results)
                    #write_submit(patient_ids, stage[0].predict(results), "final_submision.csv")
                else:
                    assert False

        except Exception as e:
            print(e)
