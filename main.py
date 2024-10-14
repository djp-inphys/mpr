import warnings
import pandas as pd
import numpy as np
import sklearn.linear_model as lm
import pickle as pk
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, log_loss
from mpl_toolkits.mplot3d import Axes3D
import logging
from process import process, classifiers, ky
from config_submit import config

import matplotlib.pyplot as plt
import matplotlib

from src.ky import KYAnalyser
# Suppress unnecessary warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_data(file_path):
    """
    Load data from a file and return as a numpy array.
    """
    try:
        data = np.load(file_path)
        logging.info(f"Data loaded successfully from {file_path}.")

        return data
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {e}")
        raise


def save_data(file_path, data):
    """
    Save numpy data to a file.
    """
    try:
        np.save(file_path, data)
        logging.info(f"Data saved to {file_path}.")
    except Exception as e:
        logging.error(f"Error saving data to {file_path}: {e}")
        raise


def perform_pca(descriptors, n_components=None):
    """
    Perform PCA on the dataset and return transformed data.
    """
    try:
        pca = PCA(n_components=n_components)
        transformed_data = pca.fit_transform(descriptors)
        logging.info(f"PCA performed with {n_components} components.")
        return transformed_data, pca
    except Exception as e:
        logging.error(f"Error performing PCA: {e}")
        raise


def plot_3d(features, labels=None, title="3D Plot"):
    """
    Plot a 3D scatter plot of the first 3 PCA components or classification scores.

    """
    matplotlib.use('TkAgg')
    try:
        fig = plt.figure(dpi=200)
        ax = fig.add_subplot(111, projection='3d')
        
        if labels is not None and hasattr(labels, "shape"):
            ax.scatter(features[labels == 1, 0], features[labels == 1, 1], features[labels == 1, 2], c='r', label='Class 1')
            ax.scatter(features[labels == 0, 0], features[labels == 0, 1], features[labels == 0, 2], c='g', label='Class 0')
        else:
            ax.scatter(features[:, 0], features[:, 1], features[:, 2], c='b', label='Data')
        
        ax.set_title(title)
        plt.legend()
        plt.show()
        logging.info(f"3D plot created: {title}.")
    except Exception as e:
        logging.error(f"Error creating 3D plot: {e}")
        raise


def ky_transformation(descriptors, labels):
    """
    Perform KY transformation on the dataset and plot results.
    """
    try:
        pca_scores, _ = perform_pca(descriptors)
        plot_3d(pca_scores, labels, "PCA Scores")

        ky_analyser = KYAnalyser()
        ky_coeffs, ky_scores   = ky_analyser.fit_transform(descriptors, labels)

        plot_3d(ky_scores, labels, "KY Scores")
        return ky_scores, ky_coeffs
    except Exception as e:
        logging.error(f"Error during KY transformation: {e}")
        raise


def process_training_data():
    """
    Load, process, and return training data descriptors and class labels.
    """
    try:
        trn_filename = f"{config['datapath']}{config['fname_train']}"
        trn_descs = load_data(trn_filename)

        clsvec = np.load( config['classes.npy'] )
    
        if config.get("kittler and young"):
            if hasattr(clsvec, "shape"):
                trn_descs, ky_coeffs = ky_transformation(trn_descs, clsvec)
                save_data(f"{config['datapath']}trn_descs.npy", trn_descs)
            else:
                raise ValueError("Class vector does not have the correct shape.")

        return trn_descs, ky_coeffs, clsvec
    
    except Exception as e:
        logging.error(f"Error processing training data: {e}")
        raise


def process_test_data(ky_coeffs=None):
    """
    Load and transform test data descriptors.
    """
    try:
        tst_filename = f"{config['datapath']}test.csv"
        tst_descs = load_data(tst_filename, columns=slice(1, -1))

        if tst_descs is not None and ky_coeffs is not None:
            tst_descs = np.dot(tst_descs, ky_coeffs)
            save_data(f"{config['datapath']}tst_descs.npy", tst_descs)
        return tst_descs
    except Exception as e:
        logging.error(f"Error processing test data: {e}")
        raise


def run_pipeline(trn_descs, clsvec, tst_descs):
    """
    Run the machine learning pipeline: PCA, classification, and result saving.
    """
    try:
        models, stages = [], []

        for stage in range(config['n_stages']):
            pca_features, pca = perform_pca(trn_descs, n_components=config.get("dim"))
            stages.append(pca)

            # Plot results
            plot_3d(pca_features, clsvec, f"PCA Scores (Train Stage {stage})")
            plot_3d(pca.transform(tst_descs), [], f"PCA Scores (Test Stage {stage})")

            # Train model
            results, mdl = process(pca_features[:, :config["dim"]], clsvec)
            models.append(mdl)

            # Save intermediate results
            np.savetxt(f"res_{stage}.csv", results, delimiter=",", fmt="%10.8f")

            # Assess model performance
            linreg = lm.LinearRegression()
            linreg.fit(results, clsvec.reshape(-1, 1))
            linreg_pred = linreg.predict(results)
            logging.info(f"Log loss at stage {stage}: {log_loss(clsvec, linreg_pred)}")

            trn_descs = results  # Use results as new descriptors for the next stage

        # Save stages and models
        pk.dump(stages, open("representation.pkl", "wb"))
        pk.dump(models, open("models.pkl", "wb"))
        logging.info("Pipeline run completed and models saved.")
    except Exception as e:
        logging.error(f"Error running pipeline: {e}")
        raise


def main():
    try:
        trn_descs, ky_coeffs, clsvec = process_training_data()
        tst_descs = process_test_data(ky_coeffs)

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
        logging.info("Main execution completed successfully.")
    except Exception as e:
        logging.error(f"Error in main: {e}")
        raise


if __name__ == '__main__':
    main()
