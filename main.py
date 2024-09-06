# stanard imports
import pandas as pd
import numpy as np
import sklearn.linear_model as lm
import pickle as pk
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, log_loss
import matplotlib.pyplot as plt
# local imports
from process import process, classifiers, ky
from config_submit import config


def load_data(file_path, delimiter=",", columns=None):
    """
    Load data from a file and return as a numpy array.
    """
    try:
        data = np.genfromtxt(file_path, skip_header=1, delimiter=delimiter)
        if columns:
            data = data[:, columns]
        return data
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None


def save_data(file_path, data):
    """
    Save numpy data to a file.
    """
    np.save(file_path, data)


def perform_pca(descriptors, n_components=None):
    """
    Perform PCA on the dataset and return transformed data.
    """
    pca = PCA(n_components=n_components)
    return pca.fit_transform(descriptors), pca


def plot_3d(features, labels, title):
    """
    Plot a 3D scatter plot of the first 3 PCA components or classification scores.
    """
    fig = plt.figure(dpi=200)
    ax = fig.gca(projection='3d')
    
    if hasattr(labels, "shape"):
        bad = features[labels == 1]
        good = features[labels == 0]
        ax.scatter(bad[:, 0], bad[:, 1], bad[:, 2], color='r', label='Bad')
        ax.scatter(good[:, 0], good[:, 1], good[:, 2], color='g', label='Good')
    else:
        ax.scatter(features[:, 0], features[:, 1], features[:, 2], color='b')
    
    ax.set_title(title)
    plt.show()


def ky_transformation(descriptors, labels):
    """
    Perform KY transformation on the dataset and plot results.
    """
    try:
        pca_scores, pca = perform_pca(descriptors)
        plot_3d(pca_scores, labels, "PCA Scores")
        
        bad, good = pca_scores[labels == 1], pca_scores[labels == 0]
        combined_data = np.concatenate((bad, good))
        ky_coeffs, ky_scores = ky(combined_data, labels)
        
        plot_3d(ky_scores, labels, "KY Scores")
        return ky_scores, ky_coeffs
    except Exception as e:
        print(f"Error during KY transformation: {e}")
        return None, None


def process_training_data():
    """
    Load, process, and return training data descriptors and class labels.
    """
    trn_filename = config["datapath"] + "train.csv"
    trn_descs = load_data(trn_filename)
    
    if trn_descs is not None:
        clsvec = trn_descs[:, 1]
        trn_descs = trn_descs[:, 2:-1]
        
        if config.get("kittler and young"):
            if hasattr(clsvec, "shape"):
                trn_descs, ky_coeffs = ky_transformation(trn_descs, clsvec)
                save_data(config["datapath"] + "trn_descs.npy", trn_descs)
            else:
                raise ValueError("Class vector does not have the correct shape.")
        
        return trn_descs, clsvec
    return None, None


def process_test_data(ky_coeffs):
    """
    Load and transform test data descriptors.
    """
    tst_filename = config["datapath"] + "test.csv"
    tst_descs = load_data(tst_filename, columns=slice(1, -1))
    
    if tst_descs is not None and ky_coeffs is not None:
        tst_descs = np.matmul(tst_descs, ky_coeffs)
        save_data(config["datapath"] + "tst_descs.npy", tst_descs)
    
    return tst_descs


def run_pipeline(trn_descs, clsvec, tst_descs):
    """
    Run the machine learning pipeline: PCA, classification, and result saving.
    """
    models, stages = [], []
    
    for stage in range(config['n_stages']):
        pca_features, pca = perform_pca(trn_descs, n_components=config.get("dim"))
        stages.append(pca)

        # Plot results
        plot_3d(pca_features, clsvec, "PCA Scores (Train)")
        plot_3d(pca.transform(tst_descs), [], "PCA Scores (Test)")
        
        # Train model
        results, mdl = process(pca_features[:, :config["dim"]], clsvec)
        models.append(mdl)
        
        # Save intermediate results
        np.savetxt(f"res_{stage}.csv", results, delimiter=",", fmt="%10.8f")

        # Assess model performance
        linreg = lm.LinearRegression()
        linreg.fit(results, clsvec.reshape(-1, 1))
        linreg_pred = linreg.predict(results)
        print(log_loss(linreg_pred, clsvec))

        trn_descs = results  # Use results as new descriptors for next stage

    # Save stages and models
    pk.dump(stages, open("representation.pkl", "wb"))
    pk.dump(models, open("models.pkl", "wb"))


def main():
    try:
        trn_descs, clsvec = process_training_data()
        tst_descs = process_test_data(None)
        
        if config["run_type"] == "build":
            run_pipeline(trn_descs, clsvec, tst_descs)
        elif config["run_type"] == "run":
            stages = pk.load(open("representation.pkl", "rb"))
            models = pk.load(open("models.pkl", "rb"))
            descs = tst_descs
            for stage, mdl in zip(stages, models):
                if isinstance(stage, PCA):
                    descs = stage.transform(descs)
                descs = classifiers(mdl, descs[:, :config["dim"]])
            # Further classification and predictions can go here
    except Exception as e:
        print(f"Error in main: {e}")


if __name__ == '__main__':
    main()
