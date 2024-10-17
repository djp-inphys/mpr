import logging
import pickle as pk
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.metrics import log_loss, roc_auc_score
import sklearn.linear_model as lm
from sklearn.preprocessing import MinMaxScaler


from config_submit import config
from src.ky import KYAnalyser
from src.process import process_none_parallel, classifiers

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


def plot_3d(features, labels=None, title="3D Plot", equal=False):
    """
    Plot a 3D scatter plot of the first 3 PCA components or classification scores.
    If equal=True, display the 3 axes with equal scaling.
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

        if equal:
            # Get the limits of the data
            max_range = np.array([features[:, 0].max()-features[:, 0].min(),
                                  features[:, 1].max()-features[:, 1].min(),
                                  features[:, 2].max()-features[:, 2].min()]).max() / 2.0

            mid_x = (features[:, 0].max()+features[:, 0].min()) * 0.5
            mid_y = (features[:, 1].max()+features[:, 1].min()) * 0.5
            mid_z = (features[:, 2].max()+features[:, 2].min()) * 0.5
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)

        plt.show()
        logging.info(f"3D plot created: {title}. Equal scaling: {equal}")
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
        ky_coeffs, ky_scores, ky_lambda   = ky_analyser.fit_transform(descriptors, labels)

        plot_3d(ky_scores, labels, "KY Scores")
        return ky_scores, ky_coeffs, ky_lambda
    except Exception as e:
        logging.error(f"Error during KY transformation: {e}")
        raise


def process_training_data():
    """
    Load, process, and return training data descriptors and class labels.
    """
    try:
        trn_filename = f"{config['datapath']}train.csv"
        trn_descs = pd.read_csv(trn_filename).values
        trn_descs = trn_descs[:, 2:-1]

        cls_filename = f"{config['datapath']}{config['fname_classes']}"
        clsvec = np.load( cls_filename )
    
        if config.get("kittler and young"):
            if hasattr(clsvec, "shape"):
                trn_descs, ky_coeffs, latent = ky_transformation(trn_descs, clsvec)
                # 
                # Uncomment to plot latent roots (eigenvalues) - provides a good idea of how many components to keep
                # 
                # # Plot latent roots (eigenvalues)
                # plt.figure(figsize=(10, 6))
                # plt.plot(range(1, len(latent) + 1), latent, 'bo-')
                # plt.title('Scree Plot: Latent Roots (Eigenvalues) from KY Transformation')
                # plt.xlabel('Component Number')
                # plt.ylabel('Eigenvalue')
                # plt.grid(True)
                # plt.show()
                # logging.info("Scree plot of latent roots (eigenvalues) created.")
                # # Determine suitable dimensionality using the latent roots
                # cumulative_variance = np.cumsum(latent) / np.sum(latent)
                suitable_dim = 2
                
                logging.info(f"Suitable dimensionality determined: {suitable_dim}")
                
                # Update config with the new dimensionality
                config['n_dim'] = min(suitable_dim, config.get('n_dim', suitable_dim))
                
                logging.info(f"Updated n_dim in config: {config['n_dim']}")
                
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
        tst_filename = f"{config['datapath']}{config['fname_test']}"
        tst_descs = load_data(tst_filename)

        if tst_descs is not None and ky_coeffs is not None:
            tst_descs = tst_descs @ ky_coeffs
            save_data(f"{config['datapath']}tst_descs.npy", tst_descs)
        return tst_descs
    except Exception as e:
        logging.error(f"Error processing test data: {e}")
        raise


def run_pipeline(trn_descs, clsvec, tst_descs):
    """
    Run the machine learning pipeline: PCA, classification, and result saving.
    """
    def estimate_intrinsic_dimensionality(data):
        """
        Estimate the intrinsic dimensionality of the data based on PCA eigenvalues.
        """
        # Perform PCA
        pca = PCA()
        pca.fit(data)
        
        eigenvalues = pca.explained_variance_
        total_variance = np.sum(eigenvalues)
        cumulative_variance_ratio = np.cumsum(eigenvalues) / total_variance
        
        # Method 1: Variance explained threshold (e.g., 95%)
        intrinsic_dim_95 = np.argmax(cumulative_variance_ratio >= 0.95) + 1
        
        # Method 2: Elbow method (find the elbow in the scree plot)
        diffs = np.diff(eigenvalues)
        elbow_index = np.argmax(diffs[:-1] - diffs[1:]) + 1
        intrinsic_dim_elbow = elbow_index + 1
        
        # Optionally, plot the scree plot
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, 'bo-')
        plt.xlabel('Principal Component')
        plt.ylabel('Eigenvalue')
        plt.title('Scree Plot')
        plt.savefig('scree_plot.png')
        plt.close()
        
        return intrinsic_dim_95, intrinsic_dim_elbow


    try:
        models, stages = [], []

        for stage in range(config['n_stages']):
            if stage != 0:
                features, pca = perform_pca(trn_descs, n_components=config.get("dim"))
                stages.append(pca)
                scaler = MinMaxScaler()
                stages.append(scaler)
                pca_features_scaled = scaler.fit_transform(features)

                # Update pca_features with the scaled version
                features = pca_features_scaled
                logging.info(f"Applied Min-Max scaling to PCA features for stage {stage}")
            else:
                features = trn_descs

            # Plot results
            plot_3d(features, clsvec, f"PCA Scores (Train Stage {stage})", equal=True)

            # Train model
            dim = 3 # TODO: Use intrinsic dimensionality
            pca_feats = features[:, :dim]
            # Check for NaN or Infinite values in the data
            mask = np.isfinite(pca_feats).all(axis=1) # note isinfinite checks for inf and nan
            if not np.all(mask):
                logging.warning(f"Found {np.sum(~mask)} rows with NaN or Infinite values. Removing these rows.")
                pca_feats = pca_feats[mask]
                clsvec = clsvec[mask]
                
                if len(pca_feats) == 0:
                    raise ValueError("All data rows contain NaN or Infinite values. Cannot proceed with empty dataset.")
                
                logging.info(f"Remaining data shape after removal: {pca_feats.shape}")
            
            results, mdl = process_none_parallel(pca_feats, clsvec)
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
